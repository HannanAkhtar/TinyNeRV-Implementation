# train_nerv_kd_temporal.py
# Temporal-derivative KD for NeRV: distill frame-to-frame *changes* (Δframe behavior)
#
# Core idea:
#   S(t) matches GT(t) (normal reconstruction loss)
#   and additionally:  (S(t) - S(t-Δ))  ≈  (T(t) - T(t-Δ))
#
# This targets temporal behavior (motion / flicker / stability) without feature hooks/adapters.
#
# Usage example (fine-tune T from a checkpoint with L as teacher):
#   python train_nerv_kd_temporal.py \
#     --dataset bunny --frame_gap 1 --test_gap 1 \
#     --teacher_weight /.../NeRV-L/.../model_val_best.pth \
#     --student_weight /.../NeRV-T/.../model_val_best.pth \
#     --epochs 300 --lr 0.0005 --warmup 0.25 \
#     --lambda_kd_temp 0.15 --kd_temp_ramp_epochs 50 \
#     --outf bunny_NeRV-T_KDtemp_L --overwrite

from __future__ import print_function

import argparse
import os
import random
import re
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
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
    Matches your existing naming convention.
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

    # training
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("-b", "--batchSize", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=300)
    parser.add_argument("--warmup", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_type", type=str, default="cosine")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--loss_type", type=str, default="Fusion6")
    parser.add_argument("--lw", type=float, default=1.0)
    parser.add_argument("--manualSeed", type=int, default=1)
    parser.add_argument("-p", "--print-freq", default=50, type=int)

    # KD temporal
    parser.add_argument("--teacher_weight", type=str, required=True)
    parser.add_argument("--student_weight", type=str, default="None", help="Optional init student checkpoint (.pth)")
    parser.add_argument("--lambda_kd_temp", type=float, default=0.15, help="Max temporal KD weight after ramp")
    parser.add_argument(
        "--kd_temp_start_epoch",
        type=int,
        default=-1,
        help="Epoch to start temporal KD ramp. -1 means start at warmup (=warmup*epochs).",
    )
    parser.add_argument("--kd_temp_ramp_epochs", type=int, default=50)

    parser.add_argument(
        "--delta_mode",
        type=str,
        default="gap",
        choices=["gap", "one"],
        help="How to choose the time step Δ: 'gap' uses frame_gap/N, 'one' uses 1/N.",
    )
    parser.add_argument(
        "--temporal_loss",
        type=str,
        default="l1",
        choices=["l1", "smoothl1"],
        help="Loss on temporal differences (Δframes).",
    )
    parser.add_argument(
        "--detach_t_prev",
        action="store_true",
        help="If set, compute teacher Δ using T(t) and T(t-Δ) separately detached. (Usually not needed; teacher is no_grad.)",
    )

    # output
    parser.add_argument("--outf", default="bunny_KD_temp", help="output folder under ./output/")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # -------------------------
    # Reproducibility
    # -------------------------
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

    with open(os.path.join(out_dir, "args.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tensorboard"))

    # -------------------------
    # Build teacher from checkpoint-parsed arch
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
    # Build student from checkpoint-parsed arch if provided (recommended), else error
    # -------------------------
    if args.student_weight != "None" and os.path.exists(args.student_weight):
        s_arch = _parse_arch_from_ckpt_path(args.student_weight)
        # Make sure PE matches student embed string
        PE = PositionalEncoding(s_arch["embed"])

        s_kargs = dict(
            embed_length=s_arch["embed_length"],
            stem_dim_num=s_arch["stem_dim_num"],
            fc_hw_dim=s_arch["fc_hw_dim"],
            expansion=s_arch["expansion"],
            reduction=s_arch["reduction"],
            stride_list=s_arch["stride_list"],
            num_blocks=s_arch["num_blocks"],
            lower_width=s_arch["lower_width"],
            bias=s_arch["bias"],
            norm=s_arch["norm"],
            act=s_arch["act"],
            conv_type=s_arch["conv_type"],
            sin_res=s_arch["sin_res"],
            sigmoid=s_arch["sigmoid"],
        )
    else:
        raise ValueError("Please provide --student_weight path (this script is intended for FT; easy to extend for scratch).")

    student = Generator(**s_kargs).to(device)
    s_sd, _ = _load_state_dict(args.student_weight)
    s_sd = _strip_module_prefix_if_needed(s_sd, student)
    missing, unexpected = student.load_state_dict(s_sd, strict=False)
    print(f"[Student] init loaded. missing={len(missing)} unexpected={len(unexpected)}")

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = optim.Adam(student.parameters(), lr=args.lr, betas=(args.beta, 0.999))

    # -------------------------
    # Data loaders
    # -------------------------
    img_transforms = transforms.ToTensor()
    data_dir = f"./data/{args.dataset.lower()}"

    train_dataset = CustomDataSet(data_dir, img_transforms, vid_list=args.vid, frame_gap=args.frame_gap)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_dataset = CustomDataSet(data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    # Temporal step size in normalized time
    # CustomDataSet uses frame_idx = k / N (NOT (N-1)), where N = total frames.
    total_frames = len(train_dataset.frame_path)
    if total_frames <= 0:
        raise RuntimeError("Could not determine total_frames from dataset.")
    base_delta = 1.0 / float(total_frames)
    if args.delta_mode == "gap":
        delta = float(args.frame_gap) * base_delta
    else:
        delta = base_delta
    print(f"[Temporal] total_frames={total_frames}, base_delta={base_delta:.8f}, delta_mode={args.delta_mode}, delta={delta:.8f}")

    # KD schedule
    warmup_epoch = int(args.warmup * args.epochs)
    kd_start_epoch = args.kd_temp_start_epoch
    if kd_start_epoch < 0:
        kd_start_epoch = warmup_epoch
    print(f"[Schedule] warmup_epoch={warmup_epoch}, kd_temp_start_epoch={kd_start_epoch}, kd_temp_ramp_epochs={args.kd_temp_ramp_epochs}")

    # Loss fn for temporal part
    if args.temporal_loss == "l1":
        temp_loss_fn = lambda a, b: F.l1_loss(a, b)
    else:
        temp_loss_fn = lambda a, b: F.smooth_l1_loss(a, b)

    # -------------------------
    # Training loop
    # -------------------------
    best_val_psnr = torch.tensor([-1.0], device=device)
    best_train_psnr = torch.tensor([-1.0], device=device)

    def save_ckpt(name, epoch, val_psnr, val_msssim, train_psnr=None, train_msssim=None):
        ckpt = dict(
            epoch=epoch,
            state_dict=student.state_dict(),
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
        teacher.eval()

        kd_w = _linear_ramp(epoch, kd_start_epoch, args.kd_temp_ramp_epochs, args.lambda_kd_temp)

        epoch_psnr_list, epoch_msssim_list = [], []
        epoch_loss_list = []
        epoch_kd_list = []

        for it, (data, norm_idx) in enumerate(train_loader):
            global_step += 1

            data = data.to(device, non_blocking=True)
            norm_idx = norm_idx.to(device, non_blocking=True)  # (B,)
            embed_t = PE(norm_idx)

            # Previous time (clamped)
            norm_prev = torch.clamp(norm_idx - delta, min=0.0)
            embed_prev = PE(norm_prev)

            adjust_lr(optimizer, epoch, it, len(train_dataset), args)

            # Student outputs
            out_t = student(embed_t)[-1]        # S(t)
            out_prev = student(embed_prev)[-1]  # S(t-Δ)

            # GT loss on current frame
            gt_t = F.adaptive_avg_pool2d(data, out_t.shape[-2:])
            gt_loss = args.lw * loss_fn(out_t, gt_t, args)

            # Teacher outputs (no grad)
            with torch.no_grad():
                t_t = teacher(embed_t)[-1]        # T(t)
                t_prev = teacher(embed_prev)[-1]  # T(t-Δ)

            # Temporal-derivative KD
            s_delta = out_t - out_prev
            if args.detach_t_prev:
                # teacher already no_grad; this is just explicit
                t_delta = (t_t.detach() - t_prev.detach())
            else:
                t_delta = (t_t - t_prev)
            kd_temp_loss = temp_loss_fn(s_delta, t_delta)

            total_loss = gt_loss + kd_w * kd_temp_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # Metrics on current frame
            with torch.no_grad():
                p = psnr_fn([out_t], [gt_t])
                m = msssim_fn([out_t], [gt_t])

            epoch_psnr_list.append(p.detach().cpu())
            epoch_msssim_list.append(m.detach().cpu())
            epoch_loss_list.append(total_loss.detach().cpu().view(1))
            epoch_kd_list.append(kd_temp_loss.detach().cpu().view(1))

            if (it + 1) % args.print_freq == 0:
                cur_psnr = torch.mean(torch.cat(epoch_psnr_list, dim=0), dim=0)
                cur_msssim = torch.mean(torch.cat(epoch_msssim_list, dim=0).float(), dim=0)
                cur_loss = torch.mean(torch.cat(epoch_loss_list, dim=0), dim=0).item()
                cur_kd = torch.mean(torch.cat(epoch_kd_list, dim=0), dim=0).item()

                print(
                    f"Epoch[{epoch+1}/{args.epochs}] Iter[{it+1}/{len(train_loader)}] "
                    f"loss={cur_loss:.4f} gt={gt_loss.item():.4f} kdW={kd_w:.4f} kdTemp={cur_kd:.4f} "
                    f"PSNR={cur_psnr.item():.3f} MSSSIM={cur_msssim.item():.4f}"
                )

                writer.add_scalar("train/loss", cur_loss, global_step)
                writer.add_scalar("train/kd_weight", kd_w, global_step)
                writer.add_scalar("train/kd_temp", cur_kd, global_step)
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

if __name__ == "__main__":
    main()
