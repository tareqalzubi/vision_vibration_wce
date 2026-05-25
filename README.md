# Vision-Vibration WCE: Self-Supervised Range and Ego-Motion Estimation

This repository is for the paper:

**Single-Camera 3D Perception and Ego-Motion Estimation in Capsule Endoscopy via Vision-Vibration Fusion with Fourier-Domain Denoising**

## What is implemented

- Manifest-based loaders for VR-Caps, EndoSLAM, C3VDv2-style data.
- Three-frame visual tuples: `{frame_prev, frame_ref, frame_next}`.
- Six-channel vibration snippets of length 240.
- Visual encoder with hierarchical fusion points.
- Two-stream MLSTM-style vibration encoder with squeeze-excitation recalibration.
- Frequency-domain reliability fusion using `torch.fft.rfft2` / `torch.fft.irfft2`.
- Ablation fusion modules: no fusion, concat-conv, FiLM, spatial gating, fixed/global/adaptive rho.
- Inverse-range decoder.
- 6-DoF pose regression head.
- Differentiable view synthesis using projective warping and `grid_sample`.
- Self-supervised losses:
  - brightness-aware intensity fidelity,
  - edge-aware smoothness regularization,
  - cross-frame inverse-range consistency.
- Evaluation metrics:
  - AbsRel, SqRel, RMSE, RMSE-log, delta accuracies,
  - ATE/RPE utilities,
  - photometric error, warp SSIM, temporal consistency,
  - vibration-severity grouping.
- Smoke-test synthetic dataset generator.

## Repository layout

```text
vision_vibration_wce_repo/
├── configs/
│   ├── default.yaml
│   ├── smoke.yaml
│   └── ablations/
├── data/
│   ├── README.md
│   └── sample_manifest.csv
├── docs/
│   ├── paper_to_code_mapping.md
│   └── required_files.md
├── scripts/
│   ├── make_synthetic_smoke_data.py
│   ├── prepare_vrcaps_manifest.py
│   ├── prepare_endoslam_manifest.py
│   ├── prepare_c3vdv2_manifest.py
│   └── run_smoke_test.sh
├── src/vv_wce/
│   ├── data/
│   ├── engine/
│   ├── geometry/
│   ├── losses/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── tests/
├── train.py
├── eval.py
├── infer.py
└── run_ablation.py
```

## Installation

```bash
cd vision_vibration_wce_repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run a smoke test

This creates tiny synthetic RGB/depth/vibration data and runs one training epoch on CPU.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/make_synthetic_smoke_data.py --out data/smoke --num-sequences 2 --frames-per-sequence 8 --image-size 64
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python train.py --config configs/smoke.yaml
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python eval.py --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoints/best.pt
```

## Dataset manifest format

The loader expects a CSV with one row per training tuple:

```csv
sequence_id,index,frame_prev,frame_ref,frame_next,vibration,depth_ref,depth_next,pose_ref,pose_next,fx,fy,cx,cy,split
seq001,1,seq001/rgb/000000.png,seq001/rgb/000001.png,seq001/rgb/000002.png,seq001/vib/000001.npy,seq001/depth/000001.npy,seq001/depth/000002.npy,seq001/pose/000001.txt,seq001/pose/000002.txt,250,250,160,160,train
```

Required for training:

- `frame_prev`, `frame_ref`, `frame_next`
- `vibration` with shape `(6, 240)` or `(240, 6)`
- camera intrinsics columns `fx, fy, cx, cy`

Optional for evaluation:

- `depth_ref`, `depth_next`
- `pose_ref`, `pose_next`

Depth can be `.npy`, `.npz`, `.png`, `.tif`, or `.tiff`. Pose can be a 4x4 matrix text file, flattened 16-number text file, or `.npy`.

## Training

```bash
python train.py --config configs/default.yaml
```

Important config fields:

```yaml
model:
  fusion_type: frequency     # none | frequency | concat | film | spatial_gate
  rho_mode: adaptive         # adaptive | fixed | global | direct
  vibration_encoder: two_stream_mlstm_se
training:
  loss_weights:
    intensity: 1.0
    smoothness: 0.001
    geometry: 0.1
```

## Ablations

```bash
python run_ablation.py --base-config configs/smoke.yaml --ablation-dir configs/ablations
```

The ablation configs correspond to manuscript studies:

- vision-only baseline,
- naive multimodal concat,
- FiLM,
- spatial gating,
- fixed rho,
- global rho,
- direct vibration injection,
- adaptive rho proposed model,
- vibration encoder variants.

## Inference

```bash
python infer.py \
  --checkpoint outputs/smoke/checkpoints/best.pt \
  --frame-prev path/to/prev.png \
  --frame-ref path/to/ref.png \
  --frame-next path/to/next.png \
  --vibration path/to/vibration.npy \
  --config configs/smoke.yaml \
  --out outputs/inference
```

## Citation placeholder

```bibtex
@article{vision_vibration_wce,
  title={Single-Camera 3D Perception and Ego-Motion Estimation in Capsule Endoscopy via Vision-Vibration Fusion with Fourier-Domain Denoising},
  author={Tareq Mahmod AlZubi},
  journal={Manuscript},
  year={2026}
}
```

## License

MIT. Dataset licenses remain with their respective dataset owners.
# vision_vibration_wce
