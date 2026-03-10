#!/usr/bin/env python3
"""
flicker_measure_tPSNR_fixed.py

Standalone script to measure flickering / temporal incoherence from reconstructed images.

Metrics:
  1) Temporal MSE between frame-differences:   MSE( (GT_t - GT_{t-1}) , (REC_t - REC_{t-1}) )
  2) Temporal PSNR (T-PSNR) from temporal MSE
  3) tSSIM between frame-differences:         SSIM( (GT_t - GT_{t-1}) , (REC_t - REC_{t-1}) )
  4) Optional LPIPS-Temporal:
       - Compare REC temporal perceptual change vs GT temporal perceptual change
       - Reports mean LPIPS(rec_t, rec_{t-1}), mean LPIPS(gt_t, gt_{t-1}), and their ratio.

Requirements:
  - opencv-python
  - numpy
  - scikit-image
Optional:
  - lpips + torch

Examples:
  python flicker_measure_tPSNR_fixed.py --gt path/to/gt --rec path/to/rec
  python flicker_measure_tPSNR_fixed.py --gt ... --rec ... --lpips --lpips_net alex
"""

import argparse
import os
import re
import math
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_LPIPS_CACHE = {}  # key: (net, device_str) -> (model, torch_device)
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*The parameter 'pretrained' is deprecated.*",
    category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=".*Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
    category=UserWarning)


def numerical_sort(files: List[str]) -> List[str]:
    def key_fn(f: str) -> Tuple[int, str]:
        nums = re.findall(r"\d+", f)
        return (int(nums[0]) if nums else -1, f)
    return sorted(files, key=key_fn)


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    files = numerical_sort(files)
    if not files:
        raise ValueError(f"No image files found in folder: {folder}")
    return files


def read_img_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    # drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # grayscale -> 3ch
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_to_match(rec_rgb: np.ndarray, gt_rgb: np.ndarray) -> np.ndarray:
    """Resize rec to gt spatial resolution (W,H)."""
    if rec_rgb.shape == gt_rgb.shape:
        return rec_rgb
    h, w = gt_rgb.shape[0], gt_rgb.shape[1]
    return cv2.resize(rec_rgb, (w, h), interpolation=cv2.INTER_LINEAR)


def temporal_mse(d_gt: np.ndarray, d_rec: np.ndarray) -> float:
    diff = d_gt - d_rec
    return float(np.mean(diff * diff))


def temporal_psnr_from_mse(mse: float, peak: float = 255.0) -> float:
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def tssim(d_gt: np.ndarray, d_rec: np.ndarray, data_range: float = 255.0) -> float:
    """
    Temporal SSIM between temporal difference images.
    We clamp to [-255,255] then shift to [0,510] so SSIM is well-defined.
    Includes compatibility fallback for older scikit-image.
    """
    d_gt_c = np.clip(d_gt, -255.0, 255.0)
    d_rec_c = np.clip(d_rec, -255.0, 255.0)
    d_gt_s = d_gt_c + 255.0
    d_rec_s = d_rec_c + 255.0

    try:
        return float(sk_ssim(d_gt_s, d_rec_s, data_range=2 * data_range, channel_axis=-1))
    except TypeError:
        # older scikit-image
        return float(sk_ssim(d_gt_s, d_rec_s, data_range=2 * data_range, multichannel=True))


_LPIPS_CACHE = {}  # key: (net, device_str) -> (model, torch_device)

def try_init_lpips(net: str = "alex", device: str = "cuda", auto_install: bool = True):
    """
    Initialize LPIPS. Loads the LPIPS network only once per (net, device) per process.
    If lpips is missing and auto_install=True, attempts to install it.
    """
    import sys

    def _install_lpips():
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install", "lpips"]
        return subprocess.run(cmd, capture_output=True, text=True)

    # 1) torch import
    try:
        import torch
    except Exception as e:
        return None, None, f"PyTorch not available. Error: {e}"

    # 2) Resolve device
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    cache_key = (net, str(dev))

    # 3) Return cached model if available
    if cache_key in _LPIPS_CACHE:
        model, cached_dev = _LPIPS_CACHE[cache_key]
        return model, cached_dev, None

    # 4) Import lpips (and optionally install)
    try:
        import lpips
    except Exception as e:
        if not auto_install:
            return None, None, f"LPIPS not available. Install with: python -m pip install lpips. Error: {e}"

        res = _install_lpips()
        if res.returncode != 0:
            msg = (
                "LPIPS not available and auto-install failed.\n"
                f"Command: {sys.executable} -m pip install lpips\n"
                f"STDOUT:\n{res.stdout}\n"
                f"STDERR:\n{res.stderr}\n"
                f"Original import error: {e}"
            )
            return None, None, msg

        try:
            import lpips
        except Exception as e2:
            return None, None, f"LPIPS install completed but import still failed: {e2}"

    # 5) Create + cache model
    model = lpips.LPIPS(net=net).to(dev)
    model.eval()
    _LPIPS_CACHE[cache_key] = (model, dev)

    return model, dev, None




def to_lpips_tensor(img_rgb_uint8: np.ndarray, device) -> "torch.Tensor":
    import torch
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = x * 2.0 - 1.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    return x


def lpips_pair(lpips_model, device, img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    import torch
    with torch.no_grad():
        t1 = to_lpips_tensor(img1_rgb, device)
        t2 = to_lpips_tensor(img2_rgb, device)
        val = lpips_model(t1, t2)
        return float(val.item())


def compute_metrics(
    gt_folder: str,
    rec_folder: str,
    max_frames: Optional[int] = None,
    lpips_enable: bool = False,
    lpips_net: str = "alex",
    lpips_device: str = "cuda",
    verbose: bool = False,
    resize_rec_to_gt: bool = True,
    strict_shapes: bool = False,
) -> dict:
    gt_files = list_images(gt_folder)
    rec_files = list_images(rec_folder)

    n = min(len(gt_files), len(rec_files))
    if max_frames is not None:
        n = min(n, max_frames)

    if n < 2:
        raise ValueError("Need at least 2 frames in both folders to compute temporal metrics.")

    # LPIPS init (optional)
    lpips_model = None
    lpips_dev = None
    if lpips_enable:
        lpips_model, lpips_dev, lpips_err = try_init_lpips(net=lpips_net, device=lpips_device)
        if lpips_model is None:
            print(lpips_err, file=sys.stderr)
            lpips_enable = False

    t_mse_list, t_psnr_list, tssim_list = [], [], []
    lpips_rec_list, lpips_gt_list = [], []

    # Read first pair
    gt0_path = os.path.join(gt_folder, gt_files[0])
    rec0_path = os.path.join(rec_folder, rec_files[0])
    gt_prev = read_img_rgb(gt0_path)
    rec_prev = read_img_rgb(rec0_path)

    if resize_rec_to_gt and gt_prev.shape != rec_prev.shape:
        rec_prev = resize_to_match(rec_prev, gt_prev)

    if strict_shapes and gt_prev.shape != rec_prev.shape:
        raise ValueError(f"Shape mismatch at first frame: {gt_files[0]} {gt_prev.shape} vs {rec_files[0]} {rec_prev.shape}")

    for i in range(1, n):
        gt_path = os.path.join(gt_folder, gt_files[i])
        rec_path = os.path.join(rec_folder, rec_files[i])

        try:
            gt_curr = read_img_rgb(gt_path)
            rec_curr = read_img_rgb(rec_path)
        except Exception:
            print("\nFAILED reading frames:")
            print("GT :", gt_path)
            print("REC:", rec_path)
            raise

        if resize_rec_to_gt and gt_curr.shape != rec_curr.shape:
            rec_curr = resize_to_match(rec_curr, gt_curr)

        if strict_shapes and gt_curr.shape != rec_curr.shape:
            raise ValueError(f"Shape mismatch: {gt_files[i]} {gt_curr.shape} vs {rec_files[i]} {rec_curr.shape}")

        d_gt = gt_curr.astype(np.float32) - gt_prev.astype(np.float32)
        d_rec = rec_curr.astype(np.float32) - rec_prev.astype(np.float32)

        mse = temporal_mse(d_gt, d_rec)
        t_mse_list.append(mse)
        t_psnr_list.append(temporal_psnr_from_mse(mse))
        tssim_list.append(tssim(d_gt, d_rec))

        if lpips_enable:
            lp_rec = lpips_pair(lpips_model, lpips_dev, rec_curr, rec_prev)
            lp_gt = lpips_pair(lpips_model, lpips_dev, gt_curr, gt_prev)
            lpips_rec_list.append(lp_rec)
            lpips_gt_list.append(lp_gt)

        if verbose and (i % 50 == 0 or i == n - 1):
            print(f"[{i:04d}/{n-1:04d}] tMSE={mse:.4f}, tPSNR={t_psnr_list[-1]:.2f}, tSSIM={tssim_list[-1]:.4f}")

        gt_prev = gt_curr
        rec_prev = rec_curr

    out = {
        "num_frames_used": n,
        "num_temporal_pairs": n - 1,
        "tMSE_mean": float(np.mean(t_mse_list)),
        "tMSE_std": float(np.std(t_mse_list)),
        "tPSNR_mean": float(np.mean(t_psnr_list)),
        "tPSNR_std": float(np.std(t_psnr_list)),
        "tSSIM_mean": float(np.mean(tssim_list)),
        "tSSIM_std": float(np.std(tssim_list)),
    }

    if lpips_enable and lpips_rec_list and lpips_gt_list:
        rec_mean = float(np.mean(lpips_rec_list))
        gt_mean = float(np.mean(lpips_gt_list))
        ratio = float(rec_mean / (gt_mean + 1e-12))
        out.update({
            "lpips_temporal_rec_mean": rec_mean,
            "lpips_temporal_rec_std": float(np.std(lpips_rec_list)),
            "lpips_temporal_gt_mean": gt_mean,
            "lpips_temporal_gt_std": float(np.std(lpips_gt_list)),
            "lpips_temporal_ratio_rec_over_gt": ratio,
            "lpips_net": lpips_net,
            "lpips_device_used": str(lpips_dev),
        })

    return out


def main():
    parser = argparse.ArgumentParser(description="Compute temporal flicker / incoherence metrics from frame folders.")
    parser.add_argument("--gt", required=True, help="Path to ground-truth frames folder.")
    parser.add_argument("--rec", required=True, help="Path to reconstructed frames folder.")
    parser.add_argument("--max_frames", type=int, default=132, help="Use only the first N frames (after sorting).")

    parser.add_argument("--resize_rec_to_gt", action="store_true",
                        help="Resize reconstructed frames to GT resolution when shapes mismatch (recommended).")
    parser.add_argument("--no_resize_rec_to_gt", action="store_true",
                        help="Disable resizing (strict matching by index/shape).")
    parser.add_argument("--strict_shapes", action="store_true",
                        help="Raise error if shapes still mismatch (after optional resizing).")

    parser.add_argument("--lpips", action="store_true", help="Enable LPIPS-Temporal metrics (requires lpips + torch).")
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
                        help="LPIPS backbone (alex fastest; vgg strongest).")
    parser.add_argument("--lpips_device", type=str, default="cuda", help="cuda or cpu (cuda used if available).")

    parser.add_argument("--verbose", action="store_true", help="Print intermediate progress.")
    args = parser.parse_args()

    resize_flag = True
    if args.no_resize_rec_to_gt:
        resize_flag = False
    elif args.resize_rec_to_gt:
        resize_flag = True

    try:
        metrics = compute_metrics(
            gt_folder=args.gt,
            rec_folder=args.rec,
            max_frames=args.max_frames,
            lpips_enable=args.lpips,
            lpips_net=args.lpips_net,
            lpips_device=args.lpips_device,
            verbose=args.verbose,
            resize_rec_to_gt=resize_flag,
            strict_shapes=args.strict_shapes,
        )
    except Exception as e:
        print("\n=== ERROR ===")
        print(f"GT folder : {args.gt}")
        print(f"REC folder: {args.rec}")
        print(f"Exception : {type(e).__name__}: {e}")
        raise

    print("\n=== Temporal Metrics ===")
    print(f"REC path:             {args.rec}")
    #print(f"Frames used:          {metrics['num_frames_used']}")
    #print(f"Temporal MSE:         {metrics['tMSE_mean']:.6f}  (std {metrics['tMSE_std']:.6f})")
    print(f"Temporal PSNR:        {metrics['tPSNR_mean']:.3f} dB (std {metrics['tPSNR_std']:.3f})")
    print(f"tSSIM (diff-SSIM):    {metrics['tSSIM_mean']:.6f}  (std {metrics['tSSIM_std']:.6f})")

    if "lpips_temporal_rec_mean" in metrics:
        #print("\n=== LPIPS-Temporal ===")
        #print(f"LPIPS net:            {metrics['lpips_net']}")
        #print(f"Device:               {metrics['lpips_device_used']}")
        print(f"LPIPS(rec_t, rec_t-1): {metrics['lpips_temporal_rec_mean']:.6f} (std {metrics['lpips_temporal_rec_std']:.6f})")
        print(f"LPIPS(gt_t, gt_t-1):   {metrics['lpips_temporal_gt_mean']:.6f} (std {metrics['lpips_temporal_gt_std']:.6f})")
        print(f"Ratio rec/gt:         {metrics['lpips_temporal_ratio_rec_over_gt']:.6f}")
        print("Interpretation: ratio > 1 suggests more perceptual temporal variation (potential flicker) than GT.")

if __name__ == "__main__":
    main()
