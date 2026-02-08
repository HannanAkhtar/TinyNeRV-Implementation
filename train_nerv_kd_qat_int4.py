from __future__ import print_function

import argparse
import os
import random
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
from tqdm import tqdm

from model_nerv import CustomDataSet, Generator
from utils import (
    PositionalEncoding,
    adjust_lr,
    worker_init_fn,
    loss_fn,
    psnr_fn,
    quantize_per_tensor,
    RoundTensor,
)

# -------------------------
# INT4 weights-only QAT (STE fake-quant) using SAME rule as utils.quantize_per_tensor
# -------------------------
def fake_quant_minmax_ste(w: torch.Tensor, bit: int = 4, axis: int = 0) -> torch.Tensor:
    """
    Fake-quantize weights using the SAME min-max uniform rule as utils.quantize_per_tensor(),
    but keep gradients via STE: w_q = w + (deq(w) - w).detach()

    axis:
      -1 : per-tensor
       0 : per-out-channel (recommended for Conv2d/Linear)
       1 : per-in-channel
    """
    if bit < 0:
        return w

    if axis == -1:
        valid = (w != 0)
        if valid.sum() == 0:
            return w
        w_min = w[valid].min()
        w_max = w[valid].max()
        scale = (w_max - w_min) / (2 ** bit)
        deq = w_min + scale * torch.round((w - w_min) / (scale + 1e-19))
        return w + (deq - w).detach()

    if axis == 0:
        ch = w.size(0)
        min_max_list = []
        for i in range(ch):
            valid = (w[i] != 0)
            if valid.sum():
                min_max_list.append([w[i][valid].min(), w[i][valid].max()])
            else:
                min_max_list.append([torch.tensor(0.0, device=w.device, dtype=w.dtype),
                                     torch.tensor(0.0, device=w.device, dtype=w.dtype)])
        mm = torch.stack([torch.stack(x) for x in min_max_list], dim=0)  # (ch,2)
        w_min = mm[:, 0]
        w_max = mm[:, 1]
        scale = (w_max - w_min) / (2 ** bit)

        if w.dim() == 4:
            w_min = w_min[:, None, None, None]
            scale = scale[:, None, None, None]
        elif w.dim() == 2:
            w_min = w_min[:, None]
            scale = scale[:, None]
        else:
            w_min = w_min.view(-1, *([1] * (w.dim() - 1)))
            scale = scale.view(-1, *([1] * (w.dim() - 1)))

        deq = w_min + scale * torch.round((w - w_min) / (scale + 1e-19))
        return w + (deq - w).detach()

    if axis == 1:
        ch = w.size(1)
        min_max_list = []
        for i in range(ch):
            valid = (w[:, i] != 0)
            if valid.sum():
                min_max_list.append([w[:, i][valid].min(), w[:, i][valid].max()])
            else:
                min_max_list.append([torch.tensor(0.0, device=w.device, dtype=w.dtype),
                                     torch.tensor(0.0, device=w.device, dtype=w.dtype)])
        mm = torch.stack([torch.stack(x) for x in min_max_list], dim=0)  # (ch,2)
        w_min = mm[:, 0]
        w_max = mm[:, 1]
        scale = (w_max - w_min) / (2 ** bit)

        if w.dim() == 4:
            w_min = w_min[None, :, None, None]
            scale = scale[None, :, None, None]
        elif w.dim() == 2:
            w_min = w_min[None, :]
            scale = scale[None, :]
        else:
            w_min = w_min.view(*([1] + [-1] + [1] * (w.dim() - 2)))
            scale = scale.view(*([1] + [-1] + [1] * (w.dim() - 2)))

        deq = w_min + scale * torch.round((w - w_min) / (scale + 1e-19))
        return w + (deq - w).detach()

    raise ValueError(f"Unsupported axis={axis}. Use -1, 0, or 1.")


class QATConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, bit: int, axis: int):
        super().__init__()
        self.conv = conv
        self.bit = bit
        self.axis = axis

    def forward(self, x):
        wq = fake_quant_minmax_ste(self.conv.weight, self.bit, self.axis)
        return F.conv2d(
            x, wq, self.conv.bias,
            self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
        )


class QATConvTranspose2d(nn.Module):
    def __init__(self, deconv: nn.ConvTranspose2d, bit: int, axis: int):
        super().__init__()
        self.deconv = deconv
        self.bit = bit
        self.axis = axis

    def forward(self, x):
        wq = fake_quant_minmax_ste(self.deconv.weight, self.bit, self.axis)
        return F.conv_transpose2d(
            x, wq, self.deconv.bias,
            self.deconv.stride, self.deconv.padding, self.deconv.output_padding,
            self.deconv.groups, self.deconv.dilation
        )


class QATLinear(nn.Module):
    def __init__(self, fc: nn.Linear, bit: int, axis: int):
        super().__init__()
        self.fc = fc
        self.bit = bit
        self.axis = axis

    def forward(self, x):
        wq = fake_quant_minmax_ste(self.fc.weight, self.bit, self.axis)
        return F.linear(x, wq, self.fc.bias)


def apply_int_qat_wrappers(module: nn.Module, bit: int = 4, axis: int = 0):
    """Recursively replace Conv/Linear layers with weights-only QAT wrappers."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QATConv2d(child, bit=bit, axis=axis))
        elif isinstance(child, nn.ConvTranspose2d):
            setattr(module, name, QATConvTranspose2d(child, bit=bit, axis=axis))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QATLinear(child, bit=bit, axis=axis))
        else:
            apply_int_qat_wrappers(child, bit=bit, axis=axis)


def _load_ckpt_strict(model: nn.Module, path: str, tag: str):
    ckt = torch.load(path, map_location='cpu')
    sd = ckt['state_dict']
    if len(sd) and list(sd.keys())[0].startswith('module.'):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    print(f"=> loaded {tag} (epoch {ckt.get('epoch', 'NA')})")


@torch.no_grad()
def evaluate_weights_only_int(model: nn.Module, val_loader, PE: nn.Module, quant_bit: int = 4, quant_axis: int = 0):
    """
    Evaluation-time weights-only quantization:
    quantize weights in state_dict -> dequant weights -> forward -> PSNR on final stage.
    """
    model.eval()
    sd = model.state_dict()
    backup = {k: v.clone() for k, v in sd.items()}

    # Quantize weights (skip bias/norm, only 2D/4D tensors)
    for k, v in sd.items():
        if v.dim() in (2, 4) and ('weight' in k) and ('norm' not in k) and ('bias' not in k):
            _, new_v = quantize_per_tensor(v, bit=quant_bit, axis=quant_axis)
            sd[k] = new_v
    model.load_state_dict(sd, strict=False)

    psnr_list = []
    for data, norm_idx in val_loader:
        data = data.cuda(non_blocking=True)
        embed_input = PE(norm_idx).cuda(non_blocking=True)

        output_list = model(embed_input)
        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        ps = psnr_fn(output_list, target_list)  # (B, num_stage)
        psnr_list.append(ps)

    psnr_all = torch.cat(psnr_list, dim=0)     # (N, num_stage)
    psnr_mean = torch.mean(psnr_all, dim=0)    # (num_stage)
    final_psnr = float(psnr_mean[-1].item())

    model.load_state_dict(backup, strict=True)
    return final_psnr


def main():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--vid', default=[None], type=int, nargs='+')
    parser.add_argument('--frame_gap', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='bunny')
    parser.add_argument('--test_gap', type=int, default=1)

    # student model
    parser.add_argument('--embed', type=str, default='1.25_80')
    parser.add_argument('--stem_dim_num', type=str, default='1024_1')
    parser.add_argument('--fc_hw_dim', type=str, default='9_16_128')
    parser.add_argument('--expansion', type=float, default=8)
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 3, 2, 2, 2])
    parser.add_argument('--num-blocks', type=int, default=1)
    parser.add_argument('--norm', default='none', type=str, choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu',
                        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower-width', type=int, default=32)
    parser.add_argument("--single_res", action='store_true')
    parser.add_argument("--conv_type", default='conv', type=str, choices=['conv', 'deconv', 'bilinear'])
    parser.add_argument('--sigmoid', action='store_true')

    # teacher model (only specify what differs; defaults fallback to student values)
    parser.add_argument('--teacher_stem_dim_num', type=str, default=None)
    parser.add_argument('--teacher_fc_hw_dim', type=str, default=None)
    parser.add_argument('--teacher_expansion', type=float, default=None)
    parser.add_argument('--teacher_reduction', type=int, default=None)
    parser.add_argument('--teacher_strides', type=int, nargs='+', default=None)
    parser.add_argument('--teacher_num_blocks', type=int, default=None)
    parser.add_argument('--teacher_lower_width', type=int, default=None)

    # training
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=0, help='warmup epochs (integer, like train_nerv.py)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_type', type=str, default='cosine')
    parser.add_argument('--lr_steps', default=[], type=float, nargs="+")
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='Fusion6')
    parser.add_argument('--lw', type=float, default=1.0)

    # QAT config
    parser.add_argument('--qat_bit', type=int, default=4)
    parser.add_argument('--qat_axis', type=int, default=0, help='0=per-out-channel, -1=per-tensor')
    parser.add_argument('--eval_quant_axis', type=int, default=0, help='0=per-channel, -1=per-tensor')
    parser.add_argument('--eval_freq', type=int, default=10)

    # KD config (final-output KD)
    parser.add_argument('--kd_w', type=float, default=1.0)
    parser.add_argument('--kd_type', type=str, default='mse', choices=['mse', 'l1'])

    # io/logging
    parser.add_argument('--manualSeed', type=int, default=1)
    parser.add_argument('--student_weight', default='None', type=str, help='FP32 STUDENT checkpoint to start from (required)')
    parser.add_argument('--teacher_weight', default='None', type=str, help='FP32 TEACHER checkpoint (required)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--outf', default='kd_qat_int4', help='output subfolder under output/')
    parser.add_argument('--suffix', default='')

    args = parser.parse_args()

    # fill teacher defaults from student if not provided
    if args.teacher_stem_dim_num is None:
        args.teacher_stem_dim_num = args.stem_dim_num
    if args.teacher_fc_hw_dim is None:
        args.teacher_fc_hw_dim = args.fc_hw_dim
    if args.teacher_expansion is None:
        args.teacher_expansion = args.expansion
    if args.teacher_reduction is None:
        args.teacher_reduction = args.reduction
    if args.teacher_strides is None:
        args.teacher_strides = args.strides
    if args.teacher_num_blocks is None:
        args.teacher_num_blocks = args.num_blocks
    if args.teacher_lower_width is None:
        args.teacher_lower_width = args.lower_width

    if args.student_weight == 'None':
        raise ValueError("Pass --student_weight <FP32 student checkpoint>.")
    if args.teacher_weight == 'None':
        raise ValueError("Pass --teacher_weight <FP32 teacher checkpoint>.")

    # seeds
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)
    cudnn.benchmark = True

    # output dir
    base_out = os.path.join('output', args.outf)
    if args.overwrite and os.path.isdir(base_out):
        shutil.rmtree(base_out)
    os.makedirs(base_out, exist_ok=True)

    # positional encoding
    PE = PositionalEncoding(args.embed)
    args.embed_length = PE.embed_length

    # build STUDENT
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
        sigmoid=args.sigmoid
    )

    # build TEACHER (can be different)
    teacher = Generator(
        embed_length=args.embed_length,
        stem_dim_num=args.teacher_stem_dim_num,
        fc_hw_dim=args.teacher_fc_hw_dim,
        expansion=args.teacher_expansion,
        num_blocks=args.teacher_num_blocks,
        norm=args.norm,
        act=args.act,
        bias=True,
        reduction=args.teacher_reduction,
        conv_type=args.conv_type,
        stride_list=args.teacher_strides,
        sin_res=args.single_res,
        lower_width=args.teacher_lower_width,
        sigmoid=args.sigmoid
    )

    print(f"=> loading FP32 student checkpoint: {args.student_weight}")
    _load_ckpt_strict(student, args.student_weight, tag="student")

    print(f"=> loading FP32 teacher checkpoint: {args.teacher_weight}")
    _load_ckpt_strict(teacher, args.teacher_weight, tag="teacher")

    # Apply weights-only QAT wrappers to STUDENT AFTER loading FP32
    apply_int_qat_wrappers(student, bit=args.qat_bit, axis=args.qat_axis)

    student = student.cuda()
    teacher = teacher.cuda()
    PE = PE.cuda()

    # freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(student.parameters(), betas=(args.beta, 0.999))

    # data
    img_transforms = transforms.ToTensor()
    data_dir = f'./data/{args.dataset.lower()}'
    train_dataset = CustomDataSet(data_dir, img_transforms, vid_list=args.vid, frame_gap=args.frame_gap)
    val_dataset = CustomDataSet(data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchSize, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        worker_init_fn=worker_init_fn
    )

    data_size = len(train_loader)

    exp_name = (
        f'{args.dataset}/KDQATw{args.qat_bit}_axis{args.qat_axis}_evalAxis{args.eval_quant_axis}/'
        f'stud_fc{args.fc_hw_dim}_low{args.lower_width}_teach_fc{args.teacher_fc_hw_dim}_low{args.teacher_lower_width}_'
        f'embed{args.embed}_{args.stem_dim_num}_exp{args.expansion}_red{args.reduction}_'
        f'e{args.epochs}_b{args.batchSize}_{args.conv_type}_lr{args.lr}_{args.lr_type}_'
        f'{args.loss_type}_act{args.act}_kdw{args.kd_w}_{args.suffix}'
    )
    out_dir = os.path.join(base_out, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir)

    best_int_psnr = -1e9
    start = datetime.now()

    for epoch in range(args.epochs):
        student.train()

        psnr_list = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=140)

        for i, (data, norm_idx) in pbar:
            data = data.cuda(non_blocking=True)
            embed_input = PE(norm_idx).cuda(non_blocking=True)

            # student forward (QAT weights)
            s_out_list = student(embed_input)
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in s_out_list]

            # teacher forward (FP32 frozen)
            with torch.no_grad():
                t_out_list = teacher(embed_input)

            # GT multi-scale loss (same as train_nerv_qat_int4.py)
            loss_list = [loss_fn(output, target, args) for output, target in zip(s_out_list, target_list)]
            loss_list = [loss_list[k] * (args.lw if k < len(loss_list) - 1 else 1.0) for k in range(len(loss_list))]
            loss_gt = sum(loss_list)

            # KD loss on final output
            if args.kd_type == 'mse':
                loss_kd = F.mse_loss(s_out_list[-1], t_out_list[-1])
            else:
                loss_kd = F.l1_loss(s_out_list[-1], t_out_list[-1])

            loss_sum = loss_gt + args.kd_w * loss_kd

            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            psnr_list.append(psnr_fn(s_out_list, target_list))
            if i % 50 == 0 or i == len(train_loader) - 1:
                train_psnr = torch.cat(psnr_list, dim=0)
                train_psnr = torch.mean(train_psnr, dim=0)
                pbar.set_description(
                    f"Epoch[{epoch+1}/{args.epochs}] Step[{i+1}/{len(train_loader)}] "
                    f"lr:{lr:.2e} PSNR:{RoundTensor(train_psnr, 2, False)} "
                    f"gt:{loss_gt.item():.4f} kd:{loss_kd.item():.4f} tot:{loss_sum.item():.4f}"
                )

        # epoch end train psnr
        train_psnr = torch.cat(psnr_list, dim=0)
        train_psnr = torch.mean(train_psnr, dim=0)
        writer.add_scalar('train/psnr_final_fp32forward', float(train_psnr[-1].item()), epoch + 1)
        writer.add_scalar('train/lr', lr, epoch + 1)
        writer.add_scalar('train/loss_gt', float(loss_gt.item()), epoch + 1)
        writer.add_scalar('train/loss_kd', float(loss_kd.item()), epoch + 1)

        # eval INT4 weights-only (PTQ-style eval)
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            int_psnr = evaluate_weights_only_int(
                student, val_loader, PE,
                quant_bit=args.qat_bit,
                quant_axis=args.eval_quant_axis
            )
            writer.add_scalar(f'val/psnr_int{args.qat_bit}_weights_only', int_psnr, epoch + 1)

            is_best = int_psnr > best_int_psnr
            best_int_psnr = max(best_int_psnr, int_psnr)

            save_obj = {
                'epoch': epoch + 1,
                'state_dict': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_int_psnr': best_int_psnr,
                'args': vars(args),
            }
            torch.save(save_obj, os.path.join(out_dir, 'model_latest.pth'))
            if is_best:
                torch.save(save_obj, os.path.join(out_dir, 'model_int_best.pth'))

            print(f"[Eval] epoch {epoch+1}: INT{args.qat_bit} weights-only PSNR = {int_psnr:.4f} | best = {best_int_psnr:.4f}")

    print("Training complete in:", datetime.now() - start)
    print(f"Best INT{args.qat_bit} weights-only PSNR: {best_int_psnr:.4f} dB")
    print("Outputs saved to:", out_dir)


if __name__ == '__main__':
    main()
