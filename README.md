# TinyNeRV: Efficient Neural Video Representations with Knowledge Distillation and Quantization

This is the official implementation of **TinyNeRV**, a comprehensive framework for compressing Neural Representations for Videos (NeRV) using knowledge distillation, pruning, and quantization techniques to enable deployment on resource-constrained edge devices.

Based on [NeRV: Neural Representations for Videos](https://arxiv.org/abs/2110.13903) (NeurIPS 2021)

### [Project Page](https://haochen-rye.github.io/NeRV) | [Original NeRV Paper](https://arxiv.org/abs/2110.13903) | [UVG Data](http://ultravideo.fi/#testsequences)

---

## Overview

NeRV represents videos as neural networks, enabling efficient compression. TinyNeRV extends this work with:

- **Multiple Knowledge Distillation Strategies**: Feature-space, final-layer, temporal, and frequency-focal distillation
- **Quantization-Aware Training**: INT4 quantization for extreme compression
- **Model Pruning**: Structured pruning with fine-tuning recovery
- **Temporal Quality Metrics**: Specialized metrics for motion stability and flicker
- **Comprehensive Evaluation**: Edge metrics and distortion-compression trade-off analysis

## Method Overview

<img src="https://i.imgur.com/OTdHe6r.png" width="560"  />

---

## Installation

Python 3.8+ required. Set up environment:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── train_nerv.py                    # Base training script (pruning/quantization)
├── train_nerv_kd_feature.py        # Feature KD with 1×1 adapters
├── train_nerv_kd_final.py          # Final-layer KD (simplest)
├── train_nerv_kd_temporal.py       # Temporal KD (motion preservation)
├── train_nerv_kd_freqfocal.py      # Frequency-focal KD
├── train_nerv_qat_int4.py          # INT4 quantization-aware training
├── train_nerv_kd_qat_int4.py       # KD + INT4 combined
├── model_nerv.py                    # NeRV architecture & dataloader
├── utils.py                         # Utility functions
├── edge_metrics.py                  # Edge device performance metrics
├── flicker_measure_tPSNR.py        # Temporal quality metrics
├── data/                            # Video datasets
│   ├── bunny/
│   ├── honeybee/
│   ├── readysetgo/
│   └── yachtride/
├── checkpoints/                     # Pre-trained models
│   ├── nerv_S.pth
│   ├── nerv_S_pruned.pth
│   └── psnr_bpp_results.csv
└── output/                          # Training logs & checkpoints
```

---

## Quick Start

### Training NeRV-S Baseline (Big Buck Bunny)

```bash
python train_nerv.py \
  -e 300 \
  --lower-width 96 \
  --num-blocks 1 \
  --dataset bunny \
  --frame_gap 1 \
  --outf bunny_baseline \
  --embed 1.25_40 \
  --stem_dim_num 512_1 \
  --reduction 2 \
  --fc_hw_dim 9_16_26 \
  --expansion 1 \
  --single_res \
  --loss Fusion6 \
  --warmup 0.2 \
  --lr_type cosine \
  --strides 5 2 2 2 2 \
  --conv_type conv \
  -b 1 \
  --lr 0.0005 \
  --norm none \
  --act swish
```

---

## Compression Techniques

### 1. Knowledge Distillation Strategies

#### **Feature KD** (Recommended for balanced compression)
Matches intermediate feature maps with learnable 1×1 adapters:

```bash
python train_nerv_kd_feature.py \
  --dataset bunny \
  --frame_gap 1 \
  --student_weight /path/to/nerv_t.pth \
  --teacher_weight /path/to/nerv_m.pth \
  --epochs 300 \
  --lr 0.0005 \
  --lambda_kd_feat 0.2 \
  --kd_feat_ramp_epochs 30 \
  --outf bunny_kd_feature
```

#### **Final-Layer KD** (Fastest to train)
Simple output-space distillation:

```bash
python train_nerv_kd_final.py \
  --dataset bunny \
  --student_weight /path/to/nerv_t.pth \
  --teacher_weight /path/to/nerv_m.pth \
  --epochs 300 \
  --lambda_kd 0.1 \
  --outf bunny_kd_final
```

#### **Temporal KD** (Preserves motion quality)
Distills frame-to-frame changes to reduce flicker:

```bash
python train_nerv_kd_temporal.py \
  --dataset bunny \
  --teacher_weight /path/to/nerv_l.pth \
  --student_weight /path/to/nerv_t.pth \
  --epochs 300 \
  --lambda_kd_temp 0.15 \
  --kd_temp_ramp_epochs 50 \
  --outf bunny_kd_temporal
```

#### **Frequency-Focal KD** (Perceptually optimized)
Focuses on frequency components:

```bash
python train_nerv_kd_freqfocal.py \
  --dataset bunny \
  --student_weight /path/to/nerv_t.pth \
  --teacher_weight /path/to/nerv_m.pth \
  --lambda_kd_freq 0.2 \
  --outf bunny_kd_freqfocal
```

### 2. Quantization

#### **INT4 Quantization-Aware Training**
Reduces model precision to 4-bit integers:

```bash
python train_nerv_qat_int4.py \
  --dataset bunny \
  --student_weight /path/to/nerv_t.pth \
  --epochs 200 \
  --lr 0.0001 \
  --outf bunny_qat_int4
```

#### **Combined KD + INT4** (Maximum compression)
Combines knowledge distillation with quantization:

```bash
python train_nerv_kd_qat_int4.py \
  --dataset bunny \
  --student_weight /path/to/nerv_t.pth \
  --teacher_weight /path/to/nerv_m.pth \
  --lambda_kd 0.1 \
  --epochs 300 \
  --outf bunny_kd_qat_int4
```

### 3. Model Pruning

#### **Pruning with Fine-tuning**
Prune and recover performance:

```bash
python train_nerv.py \
  -e 100 \
  --lower-width 96 \
  --num-blocks 1 \
  --dataset bunny \
  --frame_gap 1 \
  --outf bunny_pruned \
  --embed 1.25_40 \
  --stem_dim_num 512_1 \
  --reduction 2 \
  --fc_hw_dim 9_16_26 \
  --expansion 1 \
  --single_res \
  --loss Fusion6 \
  --warmup 0.0 \
  --lr_type cosine \
  --strides 5 2 2 2 2 \
  --conv_type conv \
  -b 1 \
  --lr 0.0005 \
  --norm none \
  --act swish \
  --weight checkpoints/nerv_S.pth \
  --not_resume_epoch \
  --prune_ratio 0.4
```

#### **Evaluate Pruned Model**
With optional quantization:

```bash
python train_nerv.py \
  -e 100 \
  --dataset bunny \
  --outf dbg \
  --weight checkpoints/nerv_S_pruned.pth \
  --prune_ratio 0.4 \
  --eval_only \
  --quant_bit 8 \
  --quant_axis 0
```

---

## Evaluation

### Standard Evaluation

```bash
python train_nerv.py \
  -e 300 \
  --dataset bunny \
  --weight checkpoints/nerv_S.pth \
  --eval_only
```

### With Speed Profiling

```bash
python train_nerv.py \
  -e 300 \
  --dataset bunny \
  --weight checkpoints/nerv_S.pth \
  --eval_only \
  --eval_fps
```

### Dump Reconstructed Frames

```bash
python train_nerv.py \
  -e 300 \
  --dataset bunny \
  --weight checkpoints/nerv_S.pth \
  --eval_only \
  --dump_images
```

---

## Quality Metrics

### Edge Performance Metrics

Measure inference efficiency on edge devices:

```bash
python edge_metrics.py \
  --ckpt /path/to/model.pth \
  --dataset bunny \
  --frame_gap 1
```

### Temporal Quality Metrics

Measure flickering and temporal coherence:

```bash
python flicker_measure_tPSNR.py \
  --gt /path/to/gt_frames \
  --rec /path/to/reconstructed_frames \
  --lpips \
  --lpips_net alex
```

**Metrics computed:**
- **T-PSNR**: Temporal PSNR from frame-difference MSE
- **T-SSIM**: SSIM of temporal changes
- **LPIPS-Temporal**: Perceptual temporal stability

---

## Results

### Compression-Distortion Trade-off

See [checkpoints/psnr_bpp_results.csv](checkpoints/psnr_bpp_results.csv) for detailed results across:
- **Datasets**: UVG, MCL-JCV
- **Models**: NeRV-T, NeRV-S, NeRV-M, NeRV-L
- **Techniques**: Baseline, Pruned, Quantized, Distilled

**Final bits-per-pixel (bpp) calculation:**
$$\text{bpp} = \frac{\text{ModelParams} \times (1 - \text{ModelSparsity}) \times \text{QuantBit}}{\text{PixelNum}}$$

---

## Pre-trained Checkpoints

- `checkpoints/nerv_S.pth` – NeRV-S baseline
- `checkpoints/nerv_S_pruned.pth` – NeRV-S with 40% pruning

---

## Hyperparameter Tuning

### Distillation Weight
- **Feature KD**: `--lambda_kd_feat 0.05-0.3` (default 0.2)
- **Final KD**: `--lambda_kd 0.05-0.2` (default 0.1)
- **Temporal KD**: `--lambda_kd_temp 0.1-0.3` (default 0.15)

### Ramp-up Schedule
- `--kd_feat_ramp_epochs 30` – Ease in feature KD
- `--kd_temp_ramp_epochs 50` – Ease in temporal KD

### Model Architecture
- `--num-blocks` – KD distills "end-of-stage" outputs (1+ blocks)
- `--strides` – Multi-scale representation
- `--fc_hw_dim` – Final MLP dimensions (affects compression)

---

## Citation

If you use TinyNeRV in your research, please cite:

```bibtex
@article{tinynev2025,
  title={TinyNeRV: Efficient Neural Video Compression with Knowledge Distillation and Quantization},
  author={...},
  year={2025},
  journal={...}
}
```

Also cite the original NeRV paper:

```bibtex
@InProceedings{chen2021nerv,
  title={NeRV: Neural Representations for Videos},
  author={Chen, Hao and He, Bo and Wang, Hanyu and Ren, Yixuan and Lim, Ser-Nam and Shrivastava, Abhinav},
  booktitle={NeurIPS},
  year={2021}
}
```

---

## License

[See LICENSE file](LICENSE)

---

## Acknowledgments

- Original NeRV implementation: [https://github.com/haochen-rye/NeRV](https://github.com/haochen-rye/NeRV)
- UVG dataset: [http://ultravideo.fi/](http://ultravideo.fi/)
