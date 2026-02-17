# PointNetFull 90%+ Reproduction and Baseline Delta

## Reproduced Result (CUDA)

Reproduction file:
- `figures/full_pointnet_runs/reproduce_90plus_r10_r14_r15.json`

Observed metrics (test split):
- Single model (`r14`): `89.4652%` accuracy, `0.442078` NLL
- Evaluation-time ensemble (`r10 + r14 + r15`): `90.1135%` accuracy, `0.406828` NLL

So the `>=90%` target is reproduced.

## Full Architecture (kept fixed)

Implementation reference:
- `Code/pointnet.py` (`Tnet`, `PointNetFull`)

`PointNetFull` forward path:
1. Input point cloud: `[B, 3, 1024]`
2. Input T-Net (`k=3`) predicts `3x3` matrix `m3x3`
3. Apply alignment: `x = bmm(m3x3, input)`
4. Shared point MLP (Conv1d 1x1 + BN + ReLU):
   - `3 -> 64`
   - `64 -> 64`
5. Feature T-Net (`k=64`) predicts `64x64` matrix `m64x64`
6. Apply feature alignment: `x = bmm(m64x64, x)`
7. Shared point MLP continues:
   - `64 -> 64`
   - `64 -> 128`
   - `128 -> 1024`
8. Symmetric aggregation: global max over points
9. Classifier MLP:
   - `1024 -> 512` (BN + ReLU)
   - `512 -> 256` (BN + ReLU)
   - `Dropout(p=0.3)`
   - `256 -> 40`
10. `LogSoftmax(dim=1)` output

Loss during training setup (outside architecture):
- NLL classification term + T-Net orthogonality regularization
- In best recipe family: regularization applied to `64x64` transform (`reg_feature=True`, `reg_input=False`)

## Exact Changes From Baseline

Baseline run reference:
- `figures/full_pointnet_runs/r00_baseline120_summary.json`

Best single-model recipe reference:
- `figures/full_pointnet_runs/r14_no_rot_ls_no_dropout_seed14_summary.json`

### Baseline (`r00_baseline120`)
- Augmentation config:
  - `normalize=False`
  - `random_sample=False`
  - `scale_translate=False`
  - `jitter=False` (uses `RandomNoise` path)
  - `point_dropout=False`
  - `rotate_z` defaults to enabled in training transform
- Optimizer/scheduler config:
  - `Adam(lr=1e-3)`
  - `StepLR(step_size=20, gamma=0.5)`
  - `label_smoothing=0.0`
  - `grad_clip=None`
  - `treg_alpha=0.001`
  - regularization defaults: `reg_input=True`, `reg_feature=True`
- Result:
  - Strict test accuracy: `86.1021%`
  - Strict test NLL: `0.545091`

### Best single-model recipe (`r14_no_rot_ls_no_dropout_seed14`)
- Augmentation/config changes vs baseline:
  - `rotate_z: True -> False`
  - `scale_translate: False -> True`
  - `jitter: False(RandomNoise) -> True(RandomJitter)`
  - `normalize` unchanged (`False`)
  - `random_sample` unchanged (`False`)
  - `point_dropout` unchanged (`False`)
- Optimization/loss changes vs baseline:
  - `Adam -> AdamW`
  - add `weight_decay=1e-4`
  - `StepLR -> CosineAnnealingLR(eta_min=1e-5)`
  - `label_smoothing: 0.0 -> 0.1`
  - `grad_clip: None -> 1.0`
  - `reg_input: True -> False`
  - `reg_feature` remains `True`
  - `treg_alpha` unchanged (`0.001`)
- Result:
  - Strict test accuracy: `89.5057%`
  - Strict test NLL: `0.442141`

### Ensemble step (evaluation-time, no retraining)
- Checkpoints averaged at inference:
  - `r10_no_rot_aug_opt_featreg_seed10`
  - `r14_no_rot_ls_no_dropout_seed14`
  - `r15_no_rot_ls_no_dropout_seed15`
- Combination rule:
  - average probabilities (`exp(log_probs)`) then take `argmax`
- Result:
  - Test accuracy: `90.1135%`
  - Test NLL: `0.406828`

## Metric Delta Summary

- Baseline single (`r00`) -> best single (`r14`):
  - Accuracy: `+3.4036` points (`86.1021% -> 89.5057%`)
  - NLL: `-0.102950` (`0.545091 -> 0.442141`)

- Baseline single (`r00`) -> 3-model ensemble (`r10+r14+r15`):
  - Accuracy: `+4.0113` points (`86.1021% -> 90.1135%`)
  - NLL: `-0.138263` (`0.545091 -> 0.406828`)
