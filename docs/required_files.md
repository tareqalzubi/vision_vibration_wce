# Required GitHub repository files

The manuscript is not reproducible from one monolithic script. A defensible GitHub repository must include the following file groups.

## Core model files

- `src/vv_wce/models/vision_encoder.py`: hierarchical visual encoder.
- `src/vv_wce/models/vibration_encoder.py`: two-stream MLSTM-style vibration encoder and variants.
- `src/vv_wce/models/fusion.py`: Fourier/Wiener reliability fusion and ablation fusions.
- `src/vv_wce/models/decoders.py`: inverse-range decoder and 6-DoF motion head.
- `src/vv_wce/models/model.py`: full end-to-end network.

## Geometry and self-supervision files

- `src/vv_wce/geometry/pose.py`: SE(3) conversion from 6-DoF vectors.
- `src/vv_wce/geometry/warping.py`: differentiable view synthesis.
- `src/vv_wce/losses/self_supervised.py`: intensity, smoothness, and geometric losses.

## Dataset files

- `src/vv_wce/data/dataset.py`: manifest-based dataset loader.
- `scripts/prepare_vrcaps_manifest.py`: VR-Caps conversion helper.
- `scripts/prepare_endoslam_manifest.py`: EndoSLAM conversion helper.
- `scripts/prepare_c3vdv2_manifest.py`: C3VDv2 conversion helper.
- `scripts/make_synthetic_smoke_data.py`: sanity-check synthetic data generator.

## Experiment files

- `train.py`: self-supervised training.
- `eval.py`: metric evaluation.
- `infer.py`: single-triplet inference.
- `run_ablation.py`: ablation launcher.
- `configs/default.yaml`: main config.
- `configs/ablations/*.yaml`: paper-ablation variants.

## Documentation files

- `README.md`: installation, data format, and commands.
- `docs/paper_to_code_mapping.md`: maps manuscript claims to code modules.
- `data/README.md`: dataset conversion requirements.
