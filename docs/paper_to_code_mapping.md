# Paper-to-code mapping

| Manuscript component | Code implementation |
|---|---|
| Three-frame visual tuple `{J_{t-1}, J_t, J_{t+1}}` | `EndoscopyTripletDataset`, `train.py` |
| Six-channel vibration snippet, length 240 | `load_vibration`, `VibrationEncoder` |
| Visual encoder with hierarchical fusion | `VisionEncoder`, `VisionVibrationModel.encode_depth_features` |
| Two-stream MLSTM + channel recalibration | `TwoStreamMLSTMSEEncoder` |
| Reliability factor `rho_tau` | `RhoEstimator` in `fusion.py` |
| Wiener-style Fourier purification | `FrequencyReliabilityFusion` |
| Naive concat / FiLM / spatial gating ablations | `ConcatFusion`, `FiLMFusion`, `SpatialGateFusion` |
| Dense inverse-range decoder | `InverseRangeDecoder` |
| 6-DoF capsule motion decoder | `PoseDecoder` |
| Differentiable reprojection | `geometry/warping.py` |
| Brightness-aware intensity fidelity | `BrightnessAwarePhotometricLoss` |
| Boundary-guided smoothness | `edge_aware_smoothness_loss` |
| Cross-frame geometric consistency | `geometric_consistency_loss` |
| AbsRel, SqRel, RMSE, delta metrics | `metrics/depth.py` |
| ATE/RPE | `metrics/pose.py` |
| Vibration severity tertiles | `metrics/robustness.py` |
| Ablation studies | `configs/ablations`, `run_ablation.py` |

## Important limitation

The code mirrors the methodology, but exact published numbers require the original train/validation/test splits, preprocessing, vibration simulation or sensing protocol, and calibration files. Those are not contained in the manuscript text.
