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

    # model
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

    # io/logging
    parser.add_argument('--manualSeed', type=int, default=1)
    parser.add_argument('--weight', default='None', type=str, help='FP32 checkpoint to start from (required)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--outf', default='qat_int4', help='output subfolder under output/')
    parser.add_argument('--suffix', default='')

    args = parser.parse_args()

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

    # model
    model = Generator(
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

    if args.weight == 'None':
        raise ValueError("Pass --weight <FP32 checkpoint> for QAT-only fine-tuning.")

    print(f"=> loading FP32 checkpoint: {args.weight}")
    ckt = torch.load(args.weight, map_location='cpu')
    sd = ckt['state_dict']
    # match train_nerv.py behavior: strip 'module.' if needed
    if len(sd) and list(sd.keys())[0].startswith('module.'):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    print(f"=> loaded (epoch {ckt.get('epoch', 'NA')})")

    # Apply weights-only QAT wrappers AFTER loading FP32
    apply_int_qat_wrappers(model, bit=args.qat_bit, axis=args.qat_axis)

    model = model.cuda()
    PE = PE.cuda()

    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))

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
        f'{args.dataset}/QATw{args.qat_bit}_axis{args.qat_axis}_evalAxis{args.eval_quant_axis}/'
        f'embed{args.embed}_{args.stem_dim_num}_fc_{args.fc_hw_dim}_exp{args.expansion}_'
        f'red{args.reduction}_low{args.lower_width}_blk{args.num_blocks}_gap{args.frame_gap}_'
        f'e{args.epochs}_b{args.batchSize}_{args.conv_type}_lr{args.lr}_{args.lr_type}_'
        f'{args.loss_type}_act{args.act}_{args.suffix}'
    )
    out_dir = os.path.join(base_out, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir)

    best_int_psnr = -1e9
    start = datetime.now()

    for epoch in range(args.epochs):
        model.train()

        psnr_list = []
        msssim_dummy = []  # not used, but keep structure similar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=140)

        for i, (data, norm_idx) in pbar:
            data = data.cuda(non_blocking=True)
            embed_input = PE(norm_idx).cuda(non_blocking=True)

            output_list = model(embed_input)
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]

            loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
            # follow train_nerv: weight lower stages by args.lw, final stage weight 1
            loss_list = [loss_list[k] * (args.lw if k < len(loss_list) - 1 else 1.0) for k in range(len(loss_list))]
            loss_sum = sum(loss_list)

            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            psnr_list.append(psnr_fn(output_list, target_list))
            if i % 50 == 0 or i == len(train_loader) - 1:
                train_psnr = torch.cat(psnr_list, dim=0)  # (B, num_stage)
                train_psnr = torch.mean(train_psnr, dim=0)  # (num_stage)
                pbar.set_description(
                    f"Epoch[{epoch+1}/{args.epochs}] Step[{i+1}/{len(train_loader)}] "
                    f"lr:{lr:.2e} PSNR:{RoundTensor(train_psnr, 2, False)} loss:{loss_sum.item():.4f}"
                )

        # epoch end train psnr
        train_psnr = torch.cat(psnr_list, dim=0)
        train_psnr = torch.mean(train_psnr, dim=0)  # (num_stage)
        writer.add_scalar('train/psnr_final_fp32forward', float(train_psnr[-1].item()), epoch + 1)
        writer.add_scalar('train/lr', lr, epoch + 1)

        # eval INT4 weights-only (PTQ-style eval)
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            int_psnr = evaluate_weights_only_int(
                model, val_loader, PE,
                quant_bit=args.qat_bit,
                quant_axis=args.eval_quant_axis
            )
            writer.add_scalar(f'val/psnr_int{args.qat_bit}_weights_only', int_psnr, epoch + 1)

            is_best = int_psnr > best_int_psnr
            best_int_psnr = max(best_int_psnr, int_psnr)

            save_obj = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
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
