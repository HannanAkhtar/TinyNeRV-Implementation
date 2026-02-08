# train_nerv_kd_freqfocal.py
# Multi-scale Output KD + Frequency-band (Laplacian) + Focal weighting (FFD-style) for NeRV.
#
# Loss:
#   L = GT(Fusion6) + kd_w * mean_scales( L_low + alpha_high * L_high_focal )
# where:
#   low(x)  = GaussianBlur(x)
#   high(x) = x - low(x)
#   L_low   = L1(low(S), low(T))
#   L_high_focal = mean( w * |high(S) - high(T)| )
#   w = normalize(mean_c(|T - S|))^gamma   (detach)
#
# Designed for FINETUNING a pretrained student (NeRV-T) from a checkpoint.
#
# Saves:
#   model_latest.pth, model_val_best.pth, model_train_best.pth
#
# Place next to: train_nerv.py, model_nerv.py, utils.py

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
# KD helpers: pooling, gaussian blur, laplacian split, focal mask
# -------------------------
def _pool_to_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return x
    h, w = x.shape[-2], x.shape[-1]
    th = max(1, h // scale)
    tw = max(1, w // scale)
    return F.adaptive_avg_pool2d(x, (th, tw))


def _gaussian_kernel2d(ks: int, sigma: float, device, dtype):
    # Create a normalized 2D Gaussian kernel (ks x ks)
    # ks must be odd
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def _gaussian_blur(x: torch.Tensor, ks: int, sigma: float) -> torch.Tensor:
    # Depthwise conv2d blur for NCHW
    if ks <= 1:
        return x
    assert ks % 2 == 1, "kernel_size must be odd"
    b, c, h, w = x.shape
    kernel2d = _gaussian_kernel2d(ks, sigma, x.device, x.dtype)  # (ks, ks)
    weight = kernel2d.view(1, 1, ks, ks).repeat(c, 1, 1, 1)      # (C,1,ks,ks)
    pad = ks // 2
    return F.conv2d(x, weight, bias=None, stride=1, padding=pad, groups=c)


def _laplacian_split(x: torch.Tensor, ks: int, sigma: float):
    low = _gaussian_blur(x, ks, sigma)
    high = x - low
    return low, high


def _focal_weight_from_diff(t: torch.Tensor, s: torch.Tensor, gamma: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Build per-pixel weight w from teacher-student disagreement:
      diff = mean_c(|T - S|)  => shape (N,1,H,W)
      w = normalize(diff)^gamma, detached
    """
    diff = (t - s).abs().mean(dim=1, keepdim=True)  # (N,1,H,W)
    # normalize per-sample
    d_min = diff.amin(dim=(2, 3), keepdim=True)
    d_max = diff.amax(dim=(2, 3), keepdim=True)
    w = (diff - d_min) / (d_max - d_min + eps)
    if gamma != 1.0:
        w = w.pow(gamma)
    return w.detach()


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

    # dataset parameters
    parser.add_argument("--vid", default=[None], type=int, nargs="+")
    parser.add_argument("--frame_gap", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="bunny")
    parser.add_argument("--test_gap", type=int, default=1)

    # student architecture
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
    parser.add_argument("--warmup", type=float, default=0.2)  # kept for compatibility; not used if kd_start_epoch set
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_type", type=str, default="cosine")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--loss_type", type=str, default="Fusion6")
    parser.add_argument("--manualSeed", type=int, default=1)
    parser.add_argument("-p", "--print-freq", default=50, type=int)

    # KD paths
    parser.add_argument("--teacher_weight", type=str, required=True)
    parser.add_argument("--student_weight", type=str, default="None")

    # KD schedule
    parser.add_argument("--lambda_kd", type=float, default=0.3, help="Target KD weight (max after ramp)")
    parser.add_argument("--kd_start_epoch", type=int, default=0, help="For finetune: start at 0")
    parser.add_argument("--kd_ramp_epochs", type=int, default=10, help="Short ramp recommended for finetuning")

    # KD multi-scale
    parser.add_argument("--kd_scales", type=int, nargs="+", default=[1, 2, 4])

    # KD frequency/focal knobs
    parser.add_argument("--blur_ks", type=int, default=5, help="Gaussian blur kernel size (odd)")
    parser.add_argument("--blur_sigma", type=float, default=1.0, help="Gaussian blur sigma")
    parser.add_argument("--alpha_high", type=float, default=3.0, help="Weight multiplier for high-frequency KD")
    parser.add_argument("--focal_gamma", type=float, default=1.0, help="Gamma for focal mask (1~2 typical)")

    # output/logging
    parser.add_argument("--outf", default="bunny_KD_freqfocal", help="output folder under ./output/")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # seeds
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # short exp id
    teacher_tag = os.path.basename(os.path.dirname(args.teacher_weight))[:32]
    student_tag = f"{args.stem_dim_num}_fc{args.fc_hw_dim}_low{args.lower_width}"
    scales_tag = "x".join([str(s) for s in args.kd_scales])
    h = hashlib.md5(
        f"{args.teacher_weight}|{args.student_weight}|{args.embed}|{student_tag}|"
        f"{args.lambda_kd}|{args.kd_start_epoch}|{args.kd_ramp_epochs}|{args.kd_scales}|"
        f"{args.blur_ks}|{args.blur_sigma}|{args.alpha_high}|{args.focal_gamma}|{args.lr}|{args.epochs}".encode()
    ).hexdigest()[:8]
    exp_id = (
        f"{args.dataset}/KDff_{student_tag}_T{teacher_tag}_lam{args.lambda_kd}"
        f"_st{args.kd_start_epoch}_r{args.kd_ramp_epochs}_sc{scales_tag}"
        f"_ks{args.blur_ks}_sig{args.blur_sigma}_ah{args.alpha_high}_g{args.focal_gamma}_{h}"
    )

    out_dir = os.path.join("output", args.outf, exp_id)
    if args.overwrite and os.path.isdir(out_dir):
        print("Will overwrite existing output dir:", out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # PE
    PE = PositionalEncoding(args.embed)
    args.embed_length = PE.embed_length

    # student
    student = Generator(
        embed_length=args.embed_length,
        stem_dim_num=args.stem_dim_num,
        fc_hw_dim=args.fc_hw_dim,
        expansion=args.expansion,
        num_blocks=args.num_blocks,
        norm=args.norm,
        act=args.act,
        bias=True,
        reduction=args.reduction,
        conv_type=args.conv_type,
        stride_list=args.strides,
        sin_res=args.single_res,
        lower_width=args.lower_width,
        sigmoid=args.sigmoid,
    ).to(device)

    if args.student_weight != "None":
        sd, _ = _load_state_dict(args.student_weight)
        sd = _strip_module_prefix_if_needed(sd, student)
        student.load_state_dict(sd, strict=True)
        print(f"=> loaded student init from {args.student_weight}")

    # teacher
    t_arch = _parse_arch_from_ckpt_path(args.teacher_weight)
    t_sd, _ = _load_state_dict(args.teacher_weight)

    if t_arch["embed_length"] != args.embed_length:
        raise ValueError(
            f"Teacher embed_length={t_arch['embed_length']} != Student embed_length={args.embed_length}. "
            f"Teacher embed inferred: {t_arch['embed']} ; Student embed: {args.embed}"
        )

    teacher = Generator(
        embed_length=t_arch["embed_length"],
        stem_dim_num=t_arch["stem_dim_num"],
        fc_hw_dim=t_arch["fc_hw_dim"],
        expansion=t_arch["expansion"],
        num_blocks=t_arch["num_blocks"],
        norm=t_arch["norm"],
        act=t_arch["act"],
        bias=True,
        reduction=t_arch["reduction"],
        conv_type=t_arch["conv_type"],
        stride_list=t_arch["stride_list"],
        sin_res=t_arch["sin_res"],
        lower_width=t_arch["lower_width"],
        sigmoid=t_arch["sigmoid"],
    ).to(device)

    t_sd = _strip_module_prefix_if_needed(t_sd, teacher)
    teacher.load_state_dict(t_sd, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"=> loaded teacher from {args.teacher_weight}")

    optimizer = optim.Adam(student.parameters(), betas=(args.beta, 0.999))
    writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))

    # data
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

    # best trackers
    train_best_psnr = torch.tensor(0.0, device=device)
    train_best_msssim = torch.tensor(0.0, device=device)
    val_best_psnr = torch.tensor(0.0, device=device)
    val_best_msssim = torch.tensor(0.0, device=device)

    start = datetime.now()
    for epoch in range(args.epochs):
        student.train()
        psnr_list, msssim_list = [], []

        for i, (data, norm_idx) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            embed_input = PE(norm_idx).to(device, non_blocking=True)

            # student forward
            s_out_list = student(embed_input)
            s_final = s_out_list[-1]

            # GT loss at full res
            gt = F.adaptive_avg_pool2d(data, s_final.shape[-2:])
            gt_loss = loss_fn(s_final, gt, args)

            # KD weight (short ramp for finetuning)
            if epoch < args.kd_start_epoch:
                kd_w = 0.0
            else:
                ramp_t = (epoch - args.kd_start_epoch) / max(args.kd_ramp_epochs, 1)
                ramp_t = float(min(max(ramp_t, 0.0), 1.0))
                kd_w = args.lambda_kd * ramp_t

            kd_loss_total = torch.tensor(0.0, device=device)
            kd_low_total = torch.tensor(0.0, device=device)
            kd_high_total = torch.tensor(0.0, device=device)

            if kd_w > 0.0:
                with torch.no_grad():
                    t_out_list = teacher(embed_input)
                    t_final = t_out_list[-1]

                per_scale_losses = []
                per_scale_low = []
                per_scale_high = []

                for sc in args.kd_scales:
                    s_sc = _pool_to_scale(s_final, sc)
                    t_sc = _pool_to_scale(t_final, sc)

                    # laplacian split
                    s_low, s_high = _laplacian_split(s_sc, args.blur_ks, args.blur_sigma)
                    t_low, t_high = _laplacian_split(t_sc, args.blur_ks, args.blur_sigma)

                    # low-frequency match (global structure)
                    l_low = F.l1_loss(s_low, t_low)

                    # focal weight based on disagreement at this scale
                    w = _focal_weight_from_diff(t_sc, s_sc, gamma=args.focal_gamma)  # (N,1,H,W)

                    # high-frequency focal match (details)
                    high_diff = (s_high - t_high).abs()
                    l_high = (w * high_diff).mean()

                    l = l_low + args.alpha_high * l_high

                    per_scale_losses.append(l)
                    per_scale_low.append(l_low.detach())
                    per_scale_high.append(l_high.detach())

                kd_loss_total = torch.stack(per_scale_losses).mean()
                kd_low_total = torch.stack(per_scale_low).mean()
                kd_high_total = torch.stack(per_scale_high).mean()

            loss = gt_loss + kd_w * kd_loss_total

            lr = adjust_lr(optimizer, epoch, i, data_size, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            psnr_list.append(psnr_fn([s_final], [gt]))
            msssim_list.append(msssim_fn([s_final], [gt]))

            if i % args.print_freq == 0 or i == len(train_loader) - 1:
                train_psnr = torch.mean(torch.cat(psnr_list, dim=0), dim=0)
                train_msssim = torch.mean(torch.cat(msssim_list, dim=0).float(), dim=0)
                print(
                    f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
                    f"Epoch[{epoch+1}/{args.epochs}] Step[{i+1}/{len(train_loader)}] "
                    f"lr:{lr:.2e} PSNR:{RoundTensor(train_psnr,2,False)} MSSSIM:{RoundTensor(train_msssim,4,False)} "
                    f"GT:{gt_loss.item():.4f} KD:{kd_loss_total.item():.4f} "
                    f"KDlow:{kd_low_total.item():.4f} KDhigh:{kd_high_total.item():.4f} kd_w:{kd_w:.3f}",
                    flush=True,
                )

        # epoch summaries
        train_psnr = torch.mean(torch.cat(psnr_list, dim=0), dim=0)
        train_msssim = torch.mean(torch.cat(msssim_list, dim=0).float(), dim=0)

        writer.add_scalar("Train/PSNR", train_psnr[-1].item(), epoch + 1)
        writer.add_scalar("Train/MSSSIM", train_msssim[-1].item(), epoch + 1)
        writer.add_scalar("Train/lr", lr, epoch + 1)
        writer.add_scalar("Train/kd_w_epoch_end", float(kd_w), epoch + 1)
        writer.add_scalar("Train/kd_loss_epoch_end", kd_loss_total.item(), epoch + 1)
        writer.add_scalar("Train/kd_low_epoch_end", kd_low_total.item(), epoch + 1)
        writer.add_scalar("Train/kd_high_epoch_end", kd_high_total.item(), epoch + 1)

        # train best
        is_train_best = train_psnr[-1] > train_best_psnr
        if is_train_best:
            train_best_psnr = train_psnr[-1].detach()
            train_best_msssim = train_msssim[-1].detach()

        # validation
        val_psnr, val_msssim = evaluate(student, val_loader, PE, device, args)
        writer.add_scalar("Val/PSNR", val_psnr[-1].item(), epoch + 1)
        writer.add_scalar("Val/MSSSIM", val_msssim[-1].item(), epoch + 1)

        # val best
        is_val_best = val_psnr[-1] > val_best_psnr
        if is_val_best:
            val_best_psnr = val_psnr[-1].detach()
            val_best_msssim = val_msssim[-1].detach()

        # checkpoint
        save_checkpoint = {
            "epoch": epoch + 1,
            "state_dict": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_best_psnr": train_best_psnr.detach().cpu(),
            "train_best_msssim": train_best_msssim.detach().cpu(),
            "val_best_psnr": val_best_psnr.detach().cpu(),
            "val_best_msssim": val_best_msssim.detach().cpu(),
            "teacher_weight": args.teacher_weight,
            "student_init_weight": args.student_weight,
            "lambda_kd": args.lambda_kd,
            "kd_start_epoch": args.kd_start_epoch,
            "kd_ramp_epochs": args.kd_ramp_epochs,
            "kd_scales": args.kd_scales,
            "blur_ks": args.blur_ks,
            "blur_sigma": args.blur_sigma,
            "alpha_high": args.alpha_high,
            "focal_gamma": args.focal_gamma,
            "teacher_arch": t_arch,
            "student_arch": dict(
                embed=args.embed,
                stem_dim_num=args.stem_dim_num,
                fc_hw_dim=args.fc_hw_dim,
                expansion=args.expansion,
                reduction=args.reduction,
                strides=args.strides,
                num_blocks=args.num_blocks,
                lower_width=args.lower_width,
                single_res=args.single_res,
                conv_type=args.conv_type,
                act=args.act,
                norm=args.norm,
                sigmoid=args.sigmoid,
            ),
        }

        torch.save(save_checkpoint, os.path.join(out_dir, "model_latest.pth"))
        if is_train_best:
            torch.save(save_checkpoint, os.path.join(out_dir, "model_train_best.pth"))
        if is_val_best:
            torch.save(save_checkpoint, os.path.join(out_dir, "model_val_best.pth"))

        print(
            f"Eval Epoch{epoch+1}: PSNR {val_psnr[-1].item():.2f} MSSSIM {val_msssim[-1].item():.4f} "
            f"{'(VAL BEST)' if is_val_best else ''} {( 'TRAIN BEST' ) if is_train_best else ''}",
            flush=True,
        )

    writer.close()
    print("Training complete in:", datetime.now() - start)


if __name__ == "__main__":
    main()
