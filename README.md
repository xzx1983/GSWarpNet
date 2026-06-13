# GSWarpNet: Decoupled Warp-and-Refine Precipitation Nowcasting

Official implementation of **GSWarpNet**, a radar echo nowcasting network with:
- **FlowEstimator** (CBAM-UNet): predicts per-lead-time optical flow fields
- **WarpModule**: parameter-free differentiable advection via `grid_sample`
- **RefinementNet** (DSC-UNet): residual correction with gradient stop decoupling
- **CompositeLoss**: task loss + warp supervision + physics-informed divergence regularisation
- **GMS / SRI**: two novel structural metrics for storm boundary evaluation

Evaluated on the public **SRadar (SHMU Malý Javorník)** dataset (Pavlík et al., 2025).

---

## Quick Results (SRadar test set, mean over 12 lead times)

| Model            | CSI@20 | CSI@35 | CSI@45 | MAE   | SSIM  | GMS   | SRI   |
|------------------|--------|--------|--------|-------|-------|-------|-------|
| Persistence      | 0.4235 | 0.0750 | 0.0190 | 4.258 | 0.516 | 0.674 | 0.252 |
| OpticalFlow      | 0.3432 | 0.0430 | 0.0094 | 5.189 | 0.448 | 0.616 | 0.223 |
| SmaAt-UNet       | 0.5363 | 0.1500 | 0.0390 | 3.588 | 0.494 | 0.729 | 0.225 |
| GSWarpNet (base) | 0.5638 | 0.1379 | 0.0336 | 2.953 | 0.630 | 0.739 | 0.234 |
| **GSWarpNet+DBZ**| **0.5835** | **0.1557** | 0.0388 | **2.943** | **0.632** | **0.739** | **0.237** |

---

## Installation

Requires Python ≥ 3.10 and CUDA-capable GPU (recommended).

```bash
git clone https://github.com/<your-org>/gswarpnet-nowcasting.git
cd gswarpnet-nowcasting

# Create virtual environment with uv (recommended)
uv venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
uv pip install -r requirements.txt

# Optional: optical flow baseline
uv pip install opencv-python-headless
```

---

## Dataset

The SRadar (SHMU Malý Javorník) dataset is publicly available from:

> Pavlík, P., et al. (2025). *LUPIN: Learning Updraft Patterns in Nowcasting*.
> Dataset: [https://doi.org/10.xxxx/sradar](https://doi.org/10.xxxx/sradar)

Download the dataset and note:
- `mj_uint8/` — directory of individual HDF5 frames (≈63,000 files)
- `metadata.h5` — dataset-level timestamps

---

## Preprocessing (run once)

Convert raw HDF5 frames to a shared float16 memmap and build train/val/test splits:

```bash
python scripts/prepare_data.py \
    --hdf5-dir  /path/to/SRadar/mj_uint8 \
    --metadata  /path/to/SRadar/metadata.h5 \
    --cache-dir ./data/sradar_cache
```

This creates `./data/sradar_cache/` containing:
- `frames.bin` — float16 memmap `[N, 336, 336]`
- `frame_idx.json` — filename → row index mapping
- `train.json`, `val.json`, `test.json` — sequence split indexes

The chronological split boundary is `2018-09-02`, matching the LUPIN paper.

---

## Training

### Step 1: Train base model (~80 epochs, ~12 h on single GPU)

```bash
python scripts/train.py \
    --cache-dir ./data/sradar_cache \
    --ckpt-dir  ./checkpoints/gswarpnet_base

# Resume after interruption
python scripts/train.py --cache-dir ./data/sradar_cache --ckpt-dir ./checkpoints/gswarpnet_base --resume
```

### Step 2: DBZ fine-tuning (~20 epochs, ~3 h)

Adds a quadratic intensity boost `(1 + 4·t²)` to up-weight heavy-rain cells.

```bash
python scripts/finetune_dbz.py \
    --cache-dir ./data/sradar_cache \
    --base-ckpt ./checkpoints/gswarpnet_base/best_model.pt \
    --ckpt-dir  ./checkpoints/gswarpnet_dbz
```

Default hyperparameters reproduce the paper results:
- Base: lr=1e-4, batch=8, epochs=80, patience=15
- DBZ FT: lr=1e-5, intensity_power=4.0, epochs=20, patience=8

---

## Evaluation

```bash
python scripts/evaluate.py \
    --cache-dir  ./data/sradar_cache \
    --base-ckpt  ./checkpoints/gswarpnet_base/best_model.pt \
    --dbz-ckpt   ./checkpoints/gswarpnet_dbz/best_model.pt \
    --output-dir ./outputs/results
```

Results are saved to `./outputs/results/sradar_results.json`.

---

## Qualitative Figures

```bash
python scripts/visualize.py \
    --cache-dir  ./data/sradar_cache \
    --base-ckpt  ./checkpoints/gswarpnet_base/best_model.pt \
    --dbz-ckpt   ./checkpoints/gswarpnet_dbz/best_model.pt \
    --output-dir ./outputs/figures \
    --cases 200 7845
```

Generates for each case:
- `qualitative_case{idx}.{png,pdf}` — input + prediction grid at +10/+30/+45/+60 min
- `error_maps_case{idx}.{png,pdf}` — absolute error maps `|pred - target|` in dBZ

---

## Using GMS and SRI as Standalone Metrics

GMS and SRI are novel structure metrics proposed in this paper. They are exposed
as standalone functions for easy adoption:

```python
import torch
from gswarpnet import compute_gms, compute_sri

# pred and target: [B, T, H, W] normalised reflectivity in [0, 1]
pred   = torch.rand(4, 12, 336, 336)
target = torch.rand(4, 12, 336, 336)

gms = compute_gms(pred, target)   # float, higher = better storm boundary similarity
sri = compute_sri(pred, target)   # float, IoU of Sobel edge masks

print(f"GMS = {gms:.4f}")
print(f"SRI = {sri:.4f}")
```

**GMS (Gradient Map Similarity)**:
```
GMS = mean( 2|∇ŷ||∇y| / (|∇ŷ|² + |∇y|² + ε) )
```
Captures storm boundary sharpness. Values in (0, 1]; 1 = perfect.

**SRI (Structural Reflectivity Index)**:
IoU of binary Sobel-edge masks thresholded at 10 dBZ gradient magnitude.
Values in [0, 1]; 1 = perfect overlap.

---

## Model Architecture

```
[B, T_in, H, W]
    ├─────────────────────────────────────────────┐
    ▼                                             │
FlowEstimator (CBAM-UNet, 4-level)               │
    ▼                                             │
flow [B, T_out, 2, H, W]                          │
    ▼                                             │
WarpModule (grid_sample, zero-param)             │
    ▼                                             │
warp_seq.detach()  ← GRADIENT STOP               │
    ▼                                             │
cat(warp_seq, x) ←────────────────────────────────┘
    ▼
RefinementNet (3-level DSC-UNet)
    ▼
output = (warp_seq + 0.5·tanh(residual)).clamp(0, 1)
```

Key design choices:
- Zero-initialised flow head → persistence at training start
- `warp_seq.detach()` decouples FlowEstimator and RefinementNet gradients
- FlowEstimator supervised by `L_warp + L_pi`; RefinementNet by task loss only

---

## Repository Structure

```
gswarpnet-nowcasting/
├── gswarpnet/           # Installable Python package
│   ├── model.py         # GSWarpNet architecture
│   ├── loss.py          # CompositeLoss (warp + divergence regularisation)
│   ├── metrics.py       # GMS, SRI, compute_all_metrics
│   ├── dataset.py       # SRadarDataset (memmap-based)
│   ├── augmentation.py  # RadarSequenceAugment
│   └── baselines.py     # Persistence, OpticalFlow
├── scripts/
│   ├── prepare_data.py  # HDF5 → memmap + split indexes
│   ├── train.py         # Base training
│   ├── finetune_dbz.py  # DBZ fine-tuning
│   ├── evaluate.py      # Full test-set evaluation
│   └── visualize.py     # Qualitative figures + error maps
├── data/                # Populated by prepare_data.py (gitignored)
├── checkpoints/         # Trained model weights (gitignored)
├── outputs/             # Evaluation results and figures (gitignored)
├── requirements.txt
└── README.md
```

---

## License

MIT License. See LICENSE file.

---

## Citation

Citation information will be added upon publication.
