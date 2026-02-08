# train_nerv_kd_feature.py
# Feature KD + 1x1 adapters (progressive internal distillation) for NeRV.
#
# What this does:
#   - Student: trains with GT loss (Fusion6 by default) on final RGB output
#   - Teacher: frozen
#   - Feature KD: matches intermediate feature maps at the end of each stage
#       KD_feat = mean_i  L1( Adapter_i(S_feat_i),  T_feat_i )
#     where Adapter_i is a learnable 1x1 conv to map student channels -> teacher channels.
#
# Notes:
#   - Works with your existing model_nerv.py without modifying it (uses forward hooks).
#   - Supports num_blocks >= 1 by distilling only "end-of-stage" block outputs.
#   - Assumes teacher & student have the same stride_list and num_blocks (typical NeRV-T vs NeRV-M).
#
# Place next to: train_nerv.py, model_nerv.py, utils.py
#
# Example:
#   python train_nerv_kd_feature.py \
#     --dataset bunny --frame_gap 1 --test_gap 1 \
#     --student_weight /path/to/nerv_t/model_val_best.pth \
#     --teacher_weight /path/to/nerv_m/model_val_best.pth \
#     --epochs 300 --lr 0.0005 \
#     --lambda_kd_feat 0.2 --kd_feat_ramp_epochs 30 \
#     --outf bunny_KD_feat
#
# (Tune lambda_kd_feat; good starting range: 0.05 ~ 0.3)

from __future__ import print_function

import argparse
import hashlib
import os
import random
import re
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from model_nerv import CustomDataSet, Generator
from utils import (
    PositionalEncoding,
    loss_fn,
    psnr_fn,
    msssim_fn,
    adjust_lr,
    RoundTensor,
    worker_init_fn,
)

# -------------------------
# Checkpoint helpers
# -------------------------
def _load_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "generator"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k], ckpt
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt, ckpt
    return ckpt, None

def _strip_module_prefix_if_needed(sd: dict, model) -> dict:
    if len(sd) == 0:
        return sd
    first_key = next(iter(sd.keys()))
    if first_key.startswith("module.") and not hasattr(model, "module"):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _parse_arch_from_ckpt_path(ckpt_path: str) -> dict:
    """
    Infer Generator(**kargs) fields from checkpoint parent folder naming.
    """
    base = os.path.basename(os.path.dirname(ckpt_path)) or os.path.basename(ckpt_path)

    def must(pattern: str, name: str) -> str:
        m = re.search(pattern, base)
        if not m:
            raise ValueError(f"Could not parse `{name}` from: {base}")
        return m.group(1)

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
    sigmoid = False
    norm = "none"
    bias = True

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
        embed=embed,
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

# -------------------------
# Feature hooker
# -------------------------
class StageFeatureHook:
    """
    Captures features at the end of each stage (i.e., after the last block of the stage).
    Works for num_blocks >= 1.
    """
    def __init__(self, model: Generator, stride_list, num_blocks: int):
        self.model = model
        self.stride_list = list(stride_list)
        self.num_blocks = int(num_blocks)
        self.handles = []
        self.feats = []

        # Determine layer indices corresponding to "end of stage"
        # In model_nerv.py, layers are appended stage-by-stage, and each stage has num_blocks blocks.
        # Stage i end index = (i+1)*num_blocks - 1
        self.stage_end_indices = [(i + 1) * self.num_blocks - 1 for i in range(len(self.stride_list))]

        if len(self.model.layers) < max(self.stage_end_indices) + 1:
            raise ValueError(
                f"Model has {len(self.model.layers)} blocks, but expected at least {max(self.stage_end_indices)+1} "
                f"for stride_list={self.stride_list}, num_blocks={self.num_blocks}."
            )

    def _hook_fn(self, module, inp, out):
        # out is the feature map after NeRVBlock: (N,C,H,W)
        self.feats.append(out)

    def install(self):
        self.remove()
        self.feats = []
        for idx in self.stage_end_indices:
            h = self.model.layers[idx].register_forward_hook(self._hook_fn)
            self.handles.append(h)

    def clear(self):
        self.feats = []

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

# -------------------------
# Adapters
# -------------------------
class FeatureAdapters(nn.Module):
    """
    One 1x1 conv per stage to map student feature channels -> teacher feature channels.
    """
    def __init__(self, s_ch_list, t_ch_list):
        super().__init__()
        assert len(s_ch_list) == len(t_ch_list)
        self.adapters = nn.ModuleList()
        for s_ch, t_ch in zip(s_ch_list, t_ch_list):
            conv = nn.Conv2d(s_ch, t_ch, kernel_size=1, stride=1, padding=0, bias=True)
            # Init so it's "near identity" when s_ch == t_ch, otherwise small weights.
            if s_ch == t_ch:
                nn.init.eye_(conv.weight.data.view(t_ch, s_ch))
                nn.init.zeros_(conv.bias.data)
            else:
                nn.init.kaiming_uniform_(conv.weight, a=1.0)
                nn.init.zeros_(conv.bias)
            self.adapters.append(conv)

    def forward(self, feat_list):
        return [ad(f) for ad, f in zip(self.adapters, feat_list)]

def _linear_ramp(cur_epoch: int, start_epoch: int, ramp_epochs: int, max_w: float) -> float:
    if max_w <= 0:
        return 0.0
    if cur_epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_w)
    t = (cur_epoch - start_epoch) / float(ramp_epochs)
    t = max(0.0, min(1.0, t))
    return float(max_w) * t

@torch.no_grad()
def evaluate(student, val_loader, PE, device, args):
    student.eval()
    psnr_list, msssim_list = [], []
    for data, norm_idx in val_loader:
        data = data.to(device, non_blocking=True)
        embed_input = PE(norm_idx).to(device, non_blocking=True)
        out_list = student(embed_input)
        s_final = out_list[-1]
        gt = F.adaptive_avg_pool2d(data, s_final.shape[-2:])
        psnr_list.append(psnr_fn([s_final], [gt]))
        msssim_list.append(msssim_fn([s_final], [gt]))
    val_psnr = torch.mean(torch.cat(psnr_list, dim=0), dim=0)
    val_msssim = torch.mean(torch.cat(msssim_list, dim=0).float(), dim=0)
    student.train()
    return val_psnr, val_msssim

def main():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--vid", default=[None], type=int, nargs="+")
    parser.add_argument("--frame_gap", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="bunny")
    parser.add_argument("--test_gap", type=int, default=1)

    # student architecture (used if student_weight doesn't encode/parse; keep same as train_nerv_kd_final)
    parser.add_argument("--embed", type=str, default="1.25_40")
    parser.add_argument("--stem_dim_num", type=str, default="256_1")
    parser.add_argument("--fc_hw_dim", type=str, default="9_16_16")
    parser.add_argument("--expansion", type=float, default=1.0)
    parser.add_argument("--reduction", type=int, default=2)
    parser.add_argument("--strides", type=int, nargs="+", default=[5, 2, 2, 2, 2])
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--lower-width", type=int, default=32)
    parser.add_argument("--single_res", action="store_true")
    parser.add_argument("--conv_type", default="conv", type=str, choices=["conv", "deconv", "bilinear"])
    parser.add_argument("--norm", default="none", type=str, choices=["none", "bn", "in"])
    parser.add_argument(
        "--act",
        type=str,
        default="swish",
        choices=["relu", "leaky", "leaky01", "relu6", "gelu", "swish", "softplus", "hardswish"],
    )
    parser.add_argument("--sigmoid", action="store_true")

    # training
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("-b", "--batchSize", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=300)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_type", type=str, default="cosine")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--loss_type", type=str, default="Fusion6")
    parser.add_argument("--lw", type=float, default=1.0)
    parser.add_argument("--manualSeed", type=int, default=1)
    parser.add_argument("-p", "--print-freq", default=50, type=int)

    # KD (feature)
    parser.add_argument("--teacher_weight", type=str, required=True, help="Path to teacher checkpoint (.pth)")
    parser.add_argument("--student_weight", type=str, default="None", help="Optional init student checkpoint (.pth)")

    parser.add_argument("--lambda_kd_feat", type=float, default=0.2, help="Max feature KD weight after ramp")
    parser.add_argument(
        "--kd_feat_start_epoch",
        type=int,
        default=-1,
        help="Epoch to start feature KD ramp. -1 means start at warmup (=warmup*epochs).",
    )
    parser.add_argument("--kd_feat_ramp_epochs", type=int, default=30, help="Ramp epochs to reach lambda_kd_feat")

    parser.add_argument(
        "--feat_kd_type",
        type=str,
        default="l1",
        choices=["l1", "attn", "l1_attn"],
        help="Feature KD variant: L1 on adapted features, attention-transfer, or both.",
    )
    parser.add_argument("--feat_attn_w", type=float, default=1.0, help="Weight for attention-transfer term (if used)")
    parser.add_argument("--feat_l1_w", type=float, default=1.0, help="Weight for feature L1 term (if used)")

    # output
    parser.add_argument("--outf", default="bunny_KD_feat", help="output folder under ./output/")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # -------------------------
    # Reproducibility
    # -------------------------
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Output folder
    # -------------------------
    out_dir = os.path.join("./output", args.outf)
    if os.path.exists(out_dir):
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Output folder exists: {out_dir}. Use --overwrite to replace.")
    os.makedirs(out_dir, exist_ok=True)

    # Save args
    with open(os.path.join(out_dir, "args.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tensorboard"))

    # -------------------------
    # Build teacher from checkpoint-parsed arch (recommended)
    # -------------------------
    t_arch = _parse_arch_from_ckpt_path(args.teacher_weight)
    PE = PositionalEncoding(t_arch["embed"])
    t_kargs = dict(
        embed_length=t_arch["embed_length"],
        stem_dim_num=t_arch["stem_dim_num"],
        fc_hw_dim=t_arch["fc_hw_dim"],
        expansion=t_arch["expansion"],
        reduction=t_arch["reduction"],
        stride_list=t_arch["stride_list"],
        num_blocks=t_arch["num_blocks"],
        lower_width=t_arch["lower_width"],
        bias=t_arch["bias"],
        norm=t_arch["norm"],
        act=t_arch["act"],
        conv_type=t_arch["conv_type"],
        sin_res=t_arch["sin_res"],
        sigmoid=t_arch["sigmoid"],
    )
    teacher = Generator(**t_kargs).to(device)

    t_sd, _ = _load_state_dict(args.teacher_weight)
    t_sd = _strip_module_prefix_if_needed(t_sd, teacher)
    missing, unexpected = teacher.load_state_dict(t_sd, strict=False)
    print(f"[Teacher] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # -------------------------
    # Build student (either from args, or parse from checkpoint path if provided)
    # -------------------------
    if args.student_weight != "None" and os.path.exists(args.student_weight):
        s_arch = _parse_arch_from_ckpt_path(args.student_weight)
        # override args to match checkpoint (safer)
        args.embed = s_arch["embed"]
        args.stem_dim_num = s_arch["stem_dim_num"]
        args.fc_hw_dim = s_arch["fc_hw_dim"]
        args.expansion = s_arch["expansion"]
        args.reduction = s_arch["reduction"]
        args.strides = s_arch["stride_list"]
        args.num_blocks = s_arch["num_blocks"]
        args.lower_width = s_arch["lower_width"]
        args.act = s_arch["act"]
        args.conv_type = s_arch["conv_type"]
        # IMPORTANT: single_res is implied by sin_res in their naming convention
        args.single_res = bool(s_arch["sin_res"])
        args.sigmoid = bool(s_arch["sigmoid"])

    s_pe = PositionalEncoding(args.embed)
    if s_pe.embed_length != PE.embed_length:
        # Ensure the positional encoding used matches the student embed string;
        # teacher PE is only used to build teacher arch; embed_length must be consistent in training.
        PE = s_pe

    s_kargs = dict(
        embed_length=PE.embed_length,
        stem_dim_num=args.stem_dim_num,
        fc_hw_dim=args.fc_hw_dim,
        expansion=args.expansion,
        reduction=args.reduction,
        stride_list=args.strides,
        num_blocks=args.num_blocks,
        lower_width=args.lower_width,
        bias=True,
        norm=args.norm,
        act=args.act,
        conv_type=args.conv_type,
        sin_res=args.single_res,
        sigmoid=args.sigmoid,
    )
    student = Generator(**s_kargs).to(device)

    if args.student_weight != "None" and os.path.exists(args.student_weight):
        s_sd, _ = _load_state_dict(args.student_weight)
        s_sd = _strip_module_prefix_if_needed(s_sd, student)
        missing, unexpected = student.load_state_dict(s_sd, strict=False)
        print(f"[Student] init loaded. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[Student] training from scratch (no --student_weight).")

    # -------------------------
    # Sanity check: stage definitions must match for feature KD
    # -------------------------
    if list(t_kargs["stride_list"]) != list(s_kargs["stride_list"]) or int(t_kargs["num_blocks"]) != int(s_kargs["num_blocks"]):
        raise ValueError(
            "Feature KD requires teacher and student to have the same stride_list and num_blocks.\n"
            f"Teacher strides={t_kargs['stride_list']} num_blocks={t_kargs['num_blocks']}\n"
            f"Student strides={s_kargs['stride_list']} num_blocks={s_kargs['num_blocks']}\n"
        )

    # -------------------------
    # Install hooks and build adapters by probing shapes
    # -------------------------
    s_hook = StageFeatureHook(student, s_kargs["stride_list"], s_kargs["num_blocks"])
    t_hook = StageFeatureHook(teacher, t_kargs["stride_list"], t_kargs["num_blocks"])
    s_hook.install()
    t_hook.install()

    with torch.no_grad():
        dummy_t = torch.tensor([0.1234], device=device)  # (1,)
        dummy_embed = PE(dummy_t).to(device)
        _ = teacher(dummy_embed)
        _ = student(dummy_embed)
        t_feats = list(t_hook.feats)
        s_feats = list(s_hook.feats)

    if len(t_feats) != len(s_feats):
        raise RuntimeError(f"Hook mismatch: teacher feats={len(t_feats)}, student feats={len(s_feats)}")

    t_ch_list = [f.shape[1] for f in t_feats]
    s_ch_list = [f.shape[1] for f in s_feats]

    adapters = FeatureAdapters(s_ch_list, t_ch_list).to(device)
    print("[Adapters] stage channels (student -> teacher):")
    for i, (sc, tc) in enumerate(zip(s_ch_list, t_ch_list)):
        print(f"  stage{i}: {sc} -> {tc}")

    # -------------------------
    # Optimizer
    # -------------------------
    params = list(student.parameters()) + list(adapters.parameters())
    optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta, 0.999))

    # -------------------------
    # Data loaders
    # -------------------------
    img_transforms = transforms.ToTensor()
    train_data_dir = f"./data/{args.dataset.lower()}"
    val_data_dir = f"./data/{args.dataset.lower()}"

    train_dataset = CustomDataSet(train_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.frame_gap)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_dataset = CustomDataSet(val_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    data_size = len(train_dataset)

    # -------------------------
    # KD schedule
    # -------------------------
    warmup_epoch = int(args.warmup * args.epochs)
    kd_start_epoch = args.kd_feat_start_epoch
    if kd_start_epoch < 0:
        kd_start_epoch = warmup_epoch

    print(f"[Schedule] warmup_epoch={warmup_epoch}, kd_feat_start_epoch={kd_start_epoch}, kd_feat_ramp_epochs={args.kd_feat_ramp_epochs}")

    # -------------------------
    # Training loop
    # -------------------------
    best_val_psnr = torch.tensor([-1.0], device=device)
    best_train_psnr = torch.tensor([-1.0], device=device)

    def save_ckpt(name, epoch, val_psnr, val_msssim, train_psnr=None, train_msssim=None):
        ckpt = dict(
            epoch=epoch,
            state_dict=student.state_dict(),
            adapters=adapters.state_dict(),
            optimizer=optimizer.state_dict(),
            val_best_psnr=best_val_psnr.detach().cpu(),
            val_best_msssim=val_msssim.detach().cpu(),
            train_best_psnr=(best_train_psnr.detach().cpu() if train_psnr is not None else None),
            train_best_msssim=(train_msssim.detach().cpu() if train_msssim is not None else None),
            args=vars(args),
        )
        torch.save(ckpt, os.path.join(out_dir, name))

    global_step = 0
    start_time = datetime.now()

    for epoch in range(args.epochs):
        student.train()
        adapters.train()
        teacher.eval()

        kd_w = _linear_ramp(epoch, kd_start_epoch, args.kd_feat_ramp_epochs, args.lambda_kd_feat)

        epoch_psnr_list, epoch_msssim_list = [], []
        epoch_loss_list = []
        epoch_kd_list = []

        for it, (data, norm_idx) in enumerate(train_loader):
            global_step += 1

            data = data.to(device, non_blocking=True)
            embed_input = PE(norm_idx).to(device, non_blocking=True)

            adjust_lr(optimizer, epoch, it, data_size, args)

            # clear hook buffers
            s_hook.clear()
            t_hook.clear()

            # forward
            out_list = student(embed_input)
            s_final = out_list[-1]

            with torch.no_grad():
                _ = teacher(embed_input)
                t_final = teacher(embed_input)[-1]  # safe even if single_res; last output exists

            # GT loss on final output
            gt = F.adaptive_avg_pool2d(data, s_final.shape[-2:])
            gt_loss = args.lw * loss_fn(s_final, gt, args)

            # feature KD loss
            # hooks captured stage features from the forward passes above
            s_feats = list(s_hook.feats)
            t_feats = list(t_hook.feats)

            # adapters map student feats to teacher channel space
            s_feats_adapt = adapters(s_feats)

            kd_feat_loss = torch.tensor(0.0, device=device)
            if kd_w > 0:
                num_stages = len(s_feats_adapt)
                # L1 feature loss
                if args.feat_kd_type in ["l1", "l1_attn"]:
                    l1_sum = 0.0
                    for sf, tf in zip(s_feats_adapt, t_feats):
                        l1_sum = l1_sum + F.l1_loss(sf, tf.detach())
                    kd_feat_loss = kd_feat_loss + args.feat_l1_w * (l1_sum / float(num_stages))

                # Attention transfer: match spatial energy maps (channel-agnostic)
                # AT(f) = normalize(mean_c(f^2))  => shape (N,1,H,W)
                if args.feat_kd_type in ["attn", "l1_attn"]:
                    at_sum = 0.0
                    for sf_raw, tf_raw in zip(s_feats, t_feats):
                        # Use raw feats (before adapter) to avoid adapter trivially learning scaling;
                        # AT is channel-agnostic anyway.
                        s_map = (sf_raw ** 2).mean(dim=1, keepdim=True)
                        t_map = (tf_raw ** 2).mean(dim=1, keepdim=True)
                        # normalize per-sample
                        s_map = s_map / (s_map.mean(dim=(2, 3), keepdim=True) + 1e-6)
                        t_map = t_map / (t_map.mean(dim=(2, 3), keepdim=True) + 1e-6)
                        at_sum = at_sum + F.l1_loss(s_map, t_map.detach())
                    kd_feat_loss = kd_feat_loss + args.feat_attn_w * (at_sum / float(num_stages))

            total_loss = gt_loss + kd_w * kd_feat_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # metrics (train)
            with torch.no_grad():
                p = psnr_fn([s_final], [gt])  # shape (B,1)
                m = msssim_fn([s_final], [gt])  # shape (B,1)

            epoch_psnr_list.append(p.detach().cpu())
            epoch_msssim_list.append(m.detach().cpu())
            epoch_loss_list.append(total_loss.detach().cpu().view(1))
            epoch_kd_list.append((kd_feat_loss.detach().cpu().view(1)))

            if (it + 1) % args.print_freq == 0:
                cur_psnr = torch.mean(torch.cat(epoch_psnr_list, dim=0), dim=0)
                cur_msssim = torch.mean(torch.cat(epoch_msssim_list, dim=0).float(), dim=0)
                cur_loss = torch.mean(torch.cat(epoch_loss_list, dim=0), dim=0).item()
                cur_kd = torch.mean(torch.cat(epoch_kd_list, dim=0), dim=0).item()

                print(
                    f"Epoch[{epoch+1}/{args.epochs}] Iter[{it+1}/{len(train_loader)}] "
                    f"loss={cur_loss:.4f} gt={gt_loss.item():.4f} kdW={kd_w:.4f} kdFeat={cur_kd:.4f} "
                    f"PSNR={cur_psnr.item():.3f} MSSSIM={cur_msssim.item():.4f}"
                )

                writer.add_scalar("train/loss", cur_loss, global_step)
                writer.add_scalar("train/kd_weight", kd_w, global_step)
                writer.add_scalar("train/kd_feat", cur_kd, global_step)
                writer.add_scalar("train/psnr", cur_psnr.item(), global_step)
                writer.add_scalar("train/msssim", cur_msssim.item(), global_step)

        # end epoch train stats
        train_psnr = torch.mean(torch.cat(epoch_psnr_list, dim=0), dim=0)
        train_msssim = torch.mean(torch.cat(epoch_msssim_list, dim=0).float(), dim=0)
        writer.add_scalar("epoch/train_psnr", train_psnr.item(), epoch)
        writer.add_scalar("epoch/train_msssim", train_msssim.item(), epoch)

        # validate
        val_psnr, val_msssim = evaluate(student, val_loader, PE, device, args)
        writer.add_scalar("epoch/val_psnr", val_psnr.item(), epoch)
        writer.add_scalar("epoch/val_msssim", val_msssim.item(), epoch)

        print(
            f"[VAL] Epoch {epoch+1}: PSNR={val_psnr.item():.3f}, MSSSIM={val_msssim.item():.4f} "
            f"(train PSNR={train_psnr.item():.3f})"
        )

        # save latest
        save_ckpt("model_latest.pth", epoch, val_psnr, val_msssim, train_psnr, train_msssim)

        # save best val
        if val_psnr.item() > best_val_psnr.item():
            best_val_psnr = val_psnr.detach()
            save_ckpt("model_val_best.pth", epoch, val_psnr, val_msssim, train_psnr, train_msssim)
            print(f"[CKPT] New best val PSNR: {best_val_psnr.item():.3f}")

        # save best train
        if train_psnr.item() > best_train_psnr.item():
            best_train_psnr = train_psnr.detach()
            save_ckpt("model_train_best.pth", epoch, val_psnr, val_msssim, train_psnr, train_msssim)

    end_time = datetime.now()
    print(f"Done. Start: {start_time}, End: {end_time}, Output: {out_dir}")

    # cleanup hooks
    s_hook.remove()
    t_hook.remove()

if __name__ == "__main__":
    main()
