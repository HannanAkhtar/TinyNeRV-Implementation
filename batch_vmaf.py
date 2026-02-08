#!/usr/bin/env python3
"""
Batch VMAF computation for NeRV dump folders (Windows-safe, auto model_path).

Layout:
  ROOT/
    RUN_NAME/
      visualize/
        gt_*.png
        pred_*.png

Requires:
  - ffmpeg on PATH
  - ffmpeg with libvmaf enabled
  - a VMAF model file available locally (e.g., vmaf_v0.6.1.pkl)
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


GT_RE = re.compile(r"^gt_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
PR_RE = re.compile(r"^pred_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, p.stdout


def check_ffmpeg_and_libvmaf() -> None:
    code, out = run_cmd(["ffmpeg", "-version"])
    if code != 0:
        raise RuntimeError("ffmpeg not found on PATH.")

    code, out = run_cmd(["ffmpeg", "-hide_banner", "-filters"])
    if code != 0 or "libvmaf" not in out:
        raise RuntimeError("ffmpeg does not have libvmaf enabled (ffmpeg -filters must show libvmaf).")


def find_visualize_dir(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "visualize"
    if direct.is_dir():
        return direct
    for m in run_dir.rglob("visualize"):
        if m.is_dir():
            return m
    return None


def index_files(vis_dir: Path) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    gt: Dict[int, Path] = {}
    pr: Dict[int, Path] = {}
    for p in vis_dir.iterdir():
        if not p.is_file():
            continue
        m = GT_RE.match(p.name)
        if m:
            gt[int(m.group(1))] = p
            continue
        m = PR_RE.match(p.name)
        if m:
            pr[int(m.group(1))] = p
            continue
    return gt, pr


def make_padded_sequence(
    gt_map: Dict[int, Path],
    pr_map: Dict[int, Path],
    temp_dir: Path,
) -> Tuple[int, int, str]:
    """
    Copy frames into temp_dir with consecutive, zero-padded names:
      gt_0000.png, pred_0000.png, ...
    Returns: (num_frames, pad_width, extension)
    """
    common = sorted(set(gt_map.keys()) & set(pr_map.keys()))
    if not common:
        raise RuntimeError("No common frame indices found between gt_* and pred_*.")

    n = len(common)
    pad = max(4, len(str(n - 1)))
    ext = gt_map[common[0]].suffix.lower()

    for i, idx in enumerate(common):
        gt_src = gt_map[idx]
        pr_src = pr_map[idx]
        gt_dst = temp_dir / f"gt_{i:0{pad}d}{ext}"
        pr_dst = temp_dir / f"pred_{i:0{pad}d}{ext}"
        shutil.copy2(gt_src, gt_dst)
        shutil.copy2(pr_src, pr_dst)

    return n, pad, ext


def encode_mp4_from_sequence(
    temp_dir: Path,
    prefix: str,
    pad: int,
    ext: str,
    fps: int,
    out_mp4: Path,
) -> None:
    # IMPORTANT: use relative paths (Windows + ffmpeg filter parsing safety)
    pattern = f"{prefix}_%0{pad}d{ext}"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-pix_fmt",
        "yuv420p",
        out_mp4.name,
    ]
    code, out = run_cmd(cmd, cwd=temp_dir)
    if code != 0:
        raise RuntimeError(f"ffmpeg encode failed for {prefix}:\n{out}")


def parse_vmaf_mean_from_json(vmaf_json: Path) -> float:
    data = json.loads(vmaf_json.read_text(encoding="utf-8"))
    pooled = data.get("pooled_metrics", {})
    vmaf = pooled.get("vmaf", {})
    mean = vmaf.get("mean", None)
    if mean is None:
        raise RuntimeError("Could not find pooled VMAF mean in VMAF JSON output.")
    return float(mean)


def find_vmaf_model_file(user_model_path: Optional[str]) -> Path:
    """
    Find a VMAF model .pkl locally.

    Priority:
      1) user provided --model_path
      2) search common conda/windows locations for vmaf_v0.6.1.pkl (or similar)
    """
    if user_model_path:
        p = Path(user_model_path).expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"--model_path provided but file not found: {p}")
        return p

    candidates = [
        "vmaf_v0.6.1.pkl",
        "vmaf_v0.6.1.json",  # unlikely, but keep
        "vmaf_v0.6.0.pkl",
        "vmaf_v0.6.0.json",
    ]

    search_roots: List[Path] = []

    # Conda env root (most likely)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        cp = Path(conda_prefix).resolve()
        search_roots.extend([
            cp / "Library" / "share" / "model",
            cp / "share" / "model",
            cp / "Library" / "share",
            cp / "share",
        ])

    # Also search common Program Files locations (if you installed ffmpeg/libvmaf system-wide)
    for base in [os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", "")]:
        if base:
            b = Path(base)
            search_roots.extend([b / "VMAF", b / "libvmaf", b / "ffmpeg", b / "ffmpeg" / "share"])

    # Deduplicate
    seen = set()
    uniq_roots = []
    for r in search_roots:
        rr = str(r)
        if rr not in seen:
            seen.add(rr)
            uniq_roots.append(r)

    # Look in those directories
    for root in uniq_roots:
        if not root.exists():
            continue
        for name in candidates:
            p = root / name
            if p.exists():
                return p

        # If not found directly, do a bounded recursive search for *.pkl
        # (avoid searching entire disk; keep it within these roots)
        for pkl in root.rglob("vmaf*.pkl"):
            return pkl

    raise RuntimeError(
        "Could not find a VMAF model file (e.g., vmaf_v0.6.1.pkl).\n"
        "Fix options:\n"
        "  - Install libvmaf models into your conda env (often under %CONDA_PREFIX%\\Library\\share\\model)\n"
        "  - Or rerun with: --model_path \"C:\\path\\to\\vmaf_v0.6.1.pkl\""
    )


def compute_vmaf(temp_dir: Path, pred_mp4: Path, gt_mp4: Path, model_path: Path) -> float:
    """
    Run libvmaf using ONLY RELATIVE PATHS in the filter string.
    We pass model_path as a relative file placed in temp_dir to avoid ':' parsing issues.
    """
    # Copy model into temp_dir so we can reference it WITHOUT absolute Windows drive paths
    model_local = temp_dir / model_path.name
    if not model_local.exists():
        shutil.copy2(model_path, model_local)

    vmaf_log_name = "vmaf.json"

    # IMPORTANT: all filter args are relative (no "C:\...") to avoid ':' parsing issues
    libvmaf_expr = f"libvmaf=model=path={model_local.name}:log_fmt=json:log_path={vmaf_log_name}"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        pred_mp4.name,
        "-i",
        gt_mp4.name,
        "-lavfi",
        libvmaf_expr,
        "-f",
        "null",
        "-",
    ]
    code, out = run_cmd(cmd, cwd=temp_dir)
    if code != 0:
        raise RuntimeError(f"ffmpeg libvmaf failed:\n{out}")

    vmaf_json = temp_dir / vmaf_log_name
    if not vmaf_json.exists():
        raise RuntimeError("VMAF JSON log was not created (vmaf.json missing).")

    return parse_vmaf_mean_from_json(vmaf_json)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory containing run folders")
    ap.add_argument("--out", type=str, default="vmaf_results.csv", help="Output CSV file path")
    ap.add_argument("--fps", type=int, default=30, help="FPS used when building mp4s from frames")
    ap.add_argument("--keep_temp", action="store_true", help="Keep temp folders for debugging")
    ap.add_argument("--model_path", type=str, default=None, help="Path to vmaf_v0.6.1.pkl (if not auto-found)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()
    out_json = out_csv.with_suffix(".json")

    if not root.is_dir():
        raise RuntimeError(f"Root directory does not exist: {root}")

    check_ffmpeg_and_libvmaf()

    model_path = find_vmaf_model_file(args.model_path)
    print(f"[INFO] Using VMAF model: {model_path}")

    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    results = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        vis = find_visualize_dir(run_dir)
        if vis is None:
            print(f"[SKIP] {run_name}: visualize/ not found", file=sys.stderr)
            continue

        gt_map, pr_map = index_files(vis)
        if not gt_map or not pr_map:
            print(f"[SKIP] {run_name}: missing gt_* or pred_* images in {vis}", file=sys.stderr)
            continue

        print(f"[INFO] Processing {run_name} (visualize: {vis})")

        if args.keep_temp:
            temp_root = root / "_vmaf_temp"
            temp_root.mkdir(parents=True, exist_ok=True)
            temp_dir = temp_root / run_name
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"vmaf_{run_name}_"))

        try:
            n, pad, ext = make_padded_sequence(gt_map, pr_map, temp_dir)

            gt_mp4 = temp_dir / "gt.mp4"
            pr_mp4 = temp_dir / "pred.mp4"
            encode_mp4_from_sequence(temp_dir, "gt", pad, ext, args.fps, gt_mp4)
            encode_mp4_from_sequence(temp_dir, "pred", pad, ext, args.fps, pr_mp4)

            score = compute_vmaf(temp_dir, pr_mp4, gt_mp4, model_path)

            # Copy VMAF log into run folder
            try:
                shutil.copy2(temp_dir / "vmaf.json", run_dir / "vmaf.json")
            except Exception:
                pass

            results.append({
                "run": run_name,
                "vmaf": score,
                "num_frames": n,
                "fps_used": args.fps,
                "run_dir": str(run_dir),
                "visualize_dir": str(vis),
            })
            print(f"[OK] {run_name}: VMAF={score:.4f} (frames={n})")

        except Exception as e:
            print(f"[FAIL] {run_name}: {e}", file=sys.stderr)

        finally:
            if not args.keep_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "vmaf", "num_frames", "fps_used", "run_dir", "visualize_dir"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")

    if not results:
        print("\nNo VMAF results were produced.", file=sys.stderr)


if __name__ == "__main__":
    main()
