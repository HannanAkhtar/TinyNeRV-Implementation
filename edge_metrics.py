# edge_metrics.py
import argparse
import os
import re
import time
from typing import Dict, Tuple, Any, List

import torch
import torch.nn as nn

from model_nerv import Generator
from utils import PositionalEncoding


def _parse_arch_from_ckpt_path(ckpt_path: str) -> dict:
    """
    Infer Generator(**kargs) fields from the checkpoint path / parent folder name.
    Example substring:
    embed1.25_40_512_1_fc_9_16_26__exp1.0_reduce2_low128_blk2_..._Strd5,2,2,2,2_SinRes_actswish_
    """
    base = os.path.basename(os.path.dirname(ckpt_path)) or os.path.basename(ckpt_path)

    def must(pattern: str, name: str) -> str:
        m = re.search(pattern, base)
        if not m:
            raise ValueError(f"Could not parse `{name}` from: {base}")
        return m.group(1)

    # embed like 1.25_40
    embed = must(r"embed([0-9.]+_[0-9]+)", "embed")
    stem_dim_num = must(r"embed[0-9.]+_[0-9]+_([0-9]+_[0-9]+)_fc_", "stem_dim_num")
    fc_hw_dim = must(r"_fc_([0-9]+_[0-9]+_[0-9]+)__", "fc_hw_dim")
    expansion = float(must(r"__exp([0-9.]+)_", "expansion"))
    reduction = int(must(r"_reduce([0-9]+)_", "reduction"))
    lower_width = int(must(r"_low([0-9]+)_", "lower_width"))
    num_blocks = int(must(r"_blk([0-9]+)_", "num_blocks"))

    strides_s = must(r"_Strd([0-9,]+)_", "stride_list")
    stride_list = [int(x) for x in strides_s.split(",")]

    sin_res = ("_SinRes" in base)
    sigmoid = False  # matches your training logs/args
    norm = "none"    # matches your commands
    bias = True      # Generator expects it

    act = "swish"
    m_act = re.search(r"_act([a-zA-Z0-9]+)_", base)
    if m_act:
        act = m_act.group(1)

    conv_type = "conv"
    m_conv = re.search(r"_b[0-9]+_([a-zA-Z0-9]+)_lr", base)
    if m_conv:
        conv_type = m_conv.group(1)

    pe = PositionalEncoding(embed)
    embed_length = pe.embed_length

    return dict(
        embed_length=embed_length,
        stem_dim_num=stem_dim_num,
        fc_hw_dim=fc_hw_dim,
        expansion=expansion,
        reduction=reduction,
        stride_list=stride_list,
        num_blocks=num_blocks,
        lower_width=lower_width,
        bias=bias,
        norm=norm,
        act=act,
        conv_type=conv_type,
        sin_res=sin_res,
        sigmoid=sigmoid,
    )


def _load_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "generator"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_size_mb(model: nn.Module) -> float:
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return nbytes / (1024 ** 2)


@torch.no_grad()
def measure_fps(model: nn.Module, dummy_embed: torch.Tensor, iters: int, warmup: int) -> float:
    model.eval()
    device = dummy_embed.device

    # warmup
    for _ in range(warmup):
        _ = model(dummy_embed)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(dummy_embed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    secs = max(t1 - t0, 1e-9)
    return iters / secs


def _format_shape(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_format_shape(t) for t in x) + "]"
    return str(type(x))


def compute_macs_via_hooks(model: nn.Module, dummy_embed: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    """
    Count MACs by observing real tensor shapes during forward.
    Counts:
      - nn.Conv2d
      - nn.Linear
    Returns:
      total_macs, breakdown dict
    """
    breakdown: Dict[str, float] = {"conv2d_macs": 0.0, "linear_macs": 0.0}

    hooks: List[Any] = []

    def conv2d_hook(mod: nn.Conv2d, inp, out):
        # inp[0]: (N, Cin, Hin, Win), out: (N, Cout, Hout, Wout)
        x = inp[0]
        if not isinstance(x, torch.Tensor) or not isinstance(out, torch.Tensor):
            return
        n = out.shape[0]
        cout = out.shape[1]
        hout = out.shape[2]
        wout = out.shape[3]
        cin = mod.in_channels
        kh, kw = mod.kernel_size
        groups = mod.groups

        # MACs per output element: (Cin/groups) * Kh * Kw
        macs = n * cout * hout * wout * (cin / groups) * kh * kw
        breakdown["conv2d_macs"] += float(macs)

    def linear_hook(mod: nn.Linear, inp, out):
        x = inp[0]
        if not isinstance(x, torch.Tensor) or not isinstance(out, torch.Tensor):
            return
        # x: (..., in_features), out: (..., out_features)
        # MACs per output element: in_features
        # total output elements = out.numel()
        in_features = mod.in_features
        macs = out.numel() * in_features
        breakdown["linear_macs"] += float(macs)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    with torch.no_grad():
        _ = model(dummy_embed)

    for h in hooks:
        h.remove()

    total = breakdown["conv2d_macs"] + breakdown["linear_macs"]
    return total, breakdown


def measure_peak_vram_mb(model: nn.Module, dummy_embed: torch.Tensor, device: torch.device, reps: int = 1) -> Tuple[float, float]:
    """
    Measure peak VRAM during inference for `reps` forwards.
    Returns (peak_allocated_mb, peak_reserved_mb).
    """
    assert device.type == "cuda"
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(reps):
            _ = model(dummy_embed)
        torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_resv = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return float(peak_alloc), float(peak_resv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to NeRV checkpoint (.pth)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--vram_reps", type=int, default=5, help="How many forwards to use for peak VRAM measurement")
    args = ap.parse_args()

    arch_args = _parse_arch_from_ckpt_path(args.ckpt)
    state_dict = _load_state_dict(args.ckpt)

    device = torch.device(args.device)
    model = Generator(**arch_args).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_embed = torch.randn(1, arch_args["embed_length"], device=device)

    # sanity forward (also useful to ensure both models actually output same HxW)
    with torch.no_grad():
        y = model(dummy_embed)

    print("===== DEVICE =====")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"output type/shape: {_format_shape(y)}")

    print("\n===== ARCH (INFERRED) =====")
    for k in [
        "embed_length", "stem_dim_num", "fc_hw_dim", "expansion", "reduction",
        "lower_width", "num_blocks", "stride_list", "conv_type", "act", "sin_res", "sigmoid"
    ]:
        print(f"{k:12s}: {arch_args[k]}")

    print("\n===== MODEL SIZE =====")
    nparams = count_params(model)
    print(f"Parameters      : {nparams/1e6:.3f} M")
    print(f"Model size (MB) : {param_size_mb(model):.2f} MB")

    print("\n===== INFERENCE SPEED =====")
    # Ensure FPS and VRAM are comparable: FPS does not affect VRAM measurement below
    fps = measure_fps(model, dummy_embed, iters=args.iters, warmup=args.warmup)
    print(f"FPS (single-frame) : {fps:.2f}")

    print("\n===== GPU MEMORY (INFERENCE) =====")
    if device.type == "cuda":
        peak_alloc, peak_resv = measure_peak_vram_mb(model, dummy_embed, device=device, reps=args.vram_reps)
        print(f"Peak allocated : {peak_alloc:.2f} MB")
        print(f"Peak reserved  : {peak_resv:.2f} MB")
        print(f"(measured over {args.vram_reps} forward passes, after empty_cache + reset_peak)")
    else:
        print("Peak allocated : N/A (CPU device)")
        print("Peak reserved  : N/A (CPU device)")

    print("\n===== COMPUTE =====")
    # Use hook-based MACs to avoid thop miscounting custom modules / PixelShuffle
    macs, breakdown = compute_macs_via_hooks(model, dummy_embed)
    print(f"Conv2d MACs : {breakdown['conv2d_macs'] / 1e6:.2f} M")
    print(f"Linear MACs : {breakdown['linear_macs'] / 1e6:.2f} M")
    print(f"Total MACs  : {macs / 1e6:.2f} M")
    print(f"FLOPs       : {2 * macs / 1e6:.2f} M  (approx = 2 * MACs)")

    # Optional: still try THOP, but clearly label it as "best-effort"
    print("\n===== COMPUTE (THOP, BEST-EFFORT) =====")
    try:
        from thop import profile
        macs_thop, _ = profile(model, inputs=(dummy_embed,), verbose=False)
        print(f"THOP MACs  : {macs_thop / 1e6:.2f} M")
        print(f"THOP FLOPs : {2 * macs_thop / 1e6:.2f} M")
        print("[NOTE] THOP may under/over-count with CustomConv/PixelShuffle; use hook-based totals for comparison.")
    except ImportError:
        print("[INFO] thop not installed. Skipping.")
    except Exception as e:
        print(f"[INFO] THOP failed: {e}")


if __name__ == "__main__":
    main()
