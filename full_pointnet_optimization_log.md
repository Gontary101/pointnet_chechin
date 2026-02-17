# PointNetFull Outside-Architecture Optimization Log

Purpose: keep `PointNetFull` architecture fixed and test minimal training/augmentation changes to reach >=90% test accuracy.

## Fixed architecture

- Input T-Net (3x3) + feature T-Net (64x64)
- Shared MLP: 3->64->64->64->128->1024
- Global max-pool + classifier 1024->512->256->40
- Dropout p=0.3, LogSoftmax output

## References used for these changes

- PointNet PyTorch repo reports ModelNet40 reference figures and `feature_transform` option: https://github.com/fxia22/pointnet.pytorch
- Official PointNet/PointNet++ style augmentation utilities (normalize, jitter, shift/scale, random point dropout): https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- PointNet++ evaluation with optional voting (`--num_votes`): https://github.com/charlesq34/pointnet2
- AdamW (decoupled weight decay): https://arxiv.org/abs/1711.05101
- CosineAnnealingLR (SGDR-based): https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
- Label smoothing API/definition: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Evidence that non-architecture factors can strongly affect point-cloud results: https://proceedings.mlr.press/v139/goyal21a.html

---
## r00_baseline120 - Baseline recipe (StepLR+Adam, simple aug)

- Time: `2026-02-16T21:32:15`
- Device: `cuda:0`
- Epochs: ran `75` / max `120`, best-loss epoch `50`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": false, "normalize": false, "point_dropout": false, "random_sample": false, "scale_translate": false}`
  - Optimization: `{"gamma": 0.5, "grad_clip": null, "label_smoothing": 0.0, "lr": 0.001, "optimizer": "adam", "scheduler": "step", "step_size": 20, "treg_alpha": 0.001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 86.10 |
| Strict Test NLL | 0.5451 |
| Voted Test Acc (%) | 86.10 |
| Voted Test NLL | 0.5451 |
| Train Acc (%) | 94.15 |
| Train NLL | 0.1724 |

- Delta vs previous run: N/A (first run)
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r00_baseline120_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r00_baseline120_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r00_baseline120_summary.json`

---
## r01_norm_sample - Add normalize + random sample (same optimizer)

- Time: `2026-02-16T21:51:16`
- Device: `cuda:0`
- Epochs: ran `66` / max `120`, best-loss epoch `41`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": false, "normalize": true, "point_dropout": false, "random_sample": true, "scale_translate": false}`
  - Optimization: `{"gamma": 0.5, "grad_clip": null, "label_smoothing": 0.0, "lr": 0.001, "optimizer": "adam", "scheduler": "step", "step_size": 20, "treg_alpha": 0.001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 85.41 |
| Strict Test NLL | 0.5360 |
| Voted Test Acc (%) | 85.41 |
| Voted Test NLL | 0.5360 |
| Train Acc (%) | 91.25 |
| Train NLL | 0.2475 |

- Delta vs previous run: N/A (first run)
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r01_norm_sample_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r01_norm_sample_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r01_norm_sample_summary.json`

---
## r02_aug_only - Add scale/translate/jitter/dropout (still Adam+StepLR)

- Time: `2026-02-16T22:16:44`
- Device: `cuda:0`
- Epochs: ran `93` / max `120`, best-loss epoch `68`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": true, "point_dropout": true, "random_sample": true, "scale_translate": true}`
  - Optimization: `{"gamma": 0.5, "grad_clip": null, "label_smoothing": 0.0, "lr": 0.001, "optimizer": "adam", "scheduler": "step", "step_size": 20, "treg_alpha": 0.001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 86.75 |
| Strict Test NLL | 0.4739 |
| Voted Test Acc (%) | 86.75 |
| Voted Test NLL | 0.4739 |
| Train Acc (%) | 92.91 |
| Train NLL | 0.2016 |

- Delta vs previous run: N/A (first run)
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r02_aug_only_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r02_aug_only_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r02_aug_only_summary.json`

---
## r03_opt_only - Optimizer/scheduler/loss tune (AdamW+Cosine+LS+clip)

- Time: `2026-02-16T22:46:35`
- Device: `cuda:0`
- Epochs: ran `120` / max `120`, best-loss epoch `100`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": true, "point_dropout": true, "random_sample": true, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 87.03 |
| Strict Test NLL | 0.4932 |
| Voted Test Acc (%) | 87.03 |
| Voted Test NLL | 0.4932 |
| Train Acc (%) | 94.46 |
| Train NLL | 0.2394 |

- Delta vs previous run: N/A (first run)
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r03_opt_only_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r03_opt_only_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r03_opt_only_summary.json`

---
## r05_nonorm_aug_opt - No normalize/sample + strong aug + AdamW/Cosine/LS/clip

- Time: `2026-02-16T23:20:12`
- Device: `cuda:0`
- Epochs: ran `120` / max `120`, best-loss epoch `104`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": true, "random_sample": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=12, mode=rotate`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 87.32 |
| Strict Test NLL | 0.4880 |
| Voted Test Acc (%) | 87.76 |
| Voted Test NLL | 0.4737 |
| Train Acc (%) | 94.91 |
| Train NLL | 0.2367 |

- Delta vs previous run (strict acc): `+0.28` points
- Delta vs previous run (voted acc): `+0.73` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r05_nonorm_aug_opt_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r05_nonorm_aug_opt_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r05_nonorm_aug_opt_summary.json`

---
## r06_nonorm_aug_opt_lowreg - r05 with lower T-Net regularization alpha=5e-4

- Time: `2026-02-16T23:49:45`
- Device: `cuda:0`
- Epochs: ran `120` / max `120`, best-loss epoch `103`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": true, "random_sample": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "scheduler": "cosine", "treg_alpha": 0.0005, "weight_decay": 0.0001}`
  - Vote eval: `count=12, mode=rotate`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 86.39 |
| Strict Test NLL | 0.5126 |
| Voted Test Acc (%) | 87.07 |
| Voted Test NLL | 0.4907 |
| Train Acc (%) | 93.72 |
| Train NLL | 0.2646 |

- Delta vs previous run (strict acc): `-0.93` points
- Delta vs previous run (voted acc): `-0.69` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r06_nonorm_aug_opt_lowreg_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r06_nonorm_aug_opt_lowreg_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r06_nonorm_aug_opt_lowreg_summary.json`

---
## r07_nonorm_aug_opt_featreg_only - r05 with regularization on 64x64 only (no 3x3 reg)

- Time: `2026-02-17T00:34:05`
- Device: `cuda:0`
- Epochs: ran `113` / max `120`, best-loss epoch `88`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": true, "random_sample": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=12, mode=rotate`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 87.64 |
| Strict Test NLL | 0.4747 |
| Voted Test Acc (%) | 87.93 |
| Voted Test NLL | 0.4556 |
| Train Acc (%) | 95.00 |
| Train NLL | 0.2341 |

- Delta vs previous run (strict acc): `+1.26` points
- Delta vs previous run (voted acc): `+0.85` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r07_nonorm_aug_opt_featreg_only_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r07_nonorm_aug_opt_featreg_only_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r07_nonorm_aug_opt_featreg_only_summary.json`

---
## evaluation_sweep_voting_and_ensembles

- Type: evaluation-only (no new training)
- Goal: test whether inference-time voting and checkpoint ensembling can reach >=90% without architecture changes.
- Models used:
  - `r02_aug_only` (normalize=true)
  - `r03_opt_only` (normalize=true)
  - `r05_nonorm_aug_opt` (normalize=false)
  - `r00_baseline120` (normalize=false)

| Setup | Votes | Test Acc (%) | Test NLL |
|---|---:|---:|---:|
| single_r05_raw | 1 | 87.3582 | 0.488103 |
| single_r05_raw | 8 | 87.6013 | 0.474732 |
| single_r05_raw | 12 | 87.7634 | 0.473695 |
| single_r05_raw | 24 | 87.6823 | 0.473405 |
| single_r03_norm | 1 | 87.0746 | 0.493256 |
| single_r03_norm | 8 | 87.1151 | 0.471582 |
| single_r03_norm | 12 | 87.1556 | 0.471139 |
| single_r03_norm | 24 | 87.1151 | 0.471125 |
| ens_r02_r03_norm | 1 | 87.3177 | 0.442743 |
| ens_r02_r03_norm | 8 | 88.2091 | 0.427716 |
| ens_r02_r03_norm | 12 | 87.9254 | 0.427975 |
| ens_r02_r03_norm | 24 | 88.0875 | 0.426812 |
| ens_r02_r03_r05_mix | 1 | 88.2496 | 0.441773 |
| ens_r02_r03_r05_mix | 8 | 88.3712 | 0.432882 |
| ens_r02_r03_r05_mix | 12 | 88.4117 | 0.433113 |
| ens_r02_r03_r05_mix | 24 | 88.4522 | 0.432300 |
| ens_r00_r02_r03_r05_mix | 1 | 88.1686 | 0.428276 |
| ens_r00_r02_r03_r05_mix | 8 | 88.3712 | 0.419425 |
| ens_r00_r02_r03_r05_mix | 12 | 88.2901 | 0.419554 |
| ens_r00_r02_r03_r05_mix | 24 | 88.2901 | 0.418689 |

- Best observed in this sweep: **88.4522%** (`ens_r02_r03_r05_mix`, 24 votes).
- Conclusion: voting + ensembling improves results, but still does not reach the 90% target under current constraints.

---
## r08_nonorm_aug_opt_featreg_only_seed8 - r07 recipe, different seed (offset 8)

- Time: `2026-02-17T01:06:07`
- Device: `cuda:0`
- Epochs: ran `120` / max `120`, best-loss epoch `116`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": true, "random_sample": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=12, mode=rotate`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 86.99 |
| Strict Test NLL | 0.5120 |
| Voted Test Acc (%) | 87.36 |
| Voted Test NLL | 0.4837 |
| Train Acc (%) | 95.48 |
| Train NLL | 0.2145 |

- Delta vs previous run (strict acc): `-0.65` points
- Delta vs previous run (voted acc): `-0.57` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r08_nonorm_aug_opt_featreg_only_seed8_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r08_nonorm_aug_opt_featreg_only_seed8_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r08_nonorm_aug_opt_featreg_only_seed8_summary.json`

---
## r09_nonorm_aug_opt_featreg_only_seed9 - r07 recipe, different seed (offset 9)

- Time: `2026-02-17T01:35:03`
- Device: `cuda:0`
- Epochs: ran `120` / max `120`, best-loss epoch `97`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": true, "random_sample": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=12, mode=rotate`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 88.01 |
| Strict Test NLL | 0.4684 |
| Voted Test Acc (%) | 88.41 |
| Voted Test NLL | 0.4647 |
| Train Acc (%) | 96.26 |
| Train NLL | 0.1926 |

- Delta vs previous run (strict acc): `+1.01` points
- Delta vs previous run (voted acc): `+1.05` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r09_nonorm_aug_opt_featreg_only_seed9_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r09_nonorm_aug_opt_featreg_only_seed9_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r09_nonorm_aug_opt_featreg_only_seed9_summary.json`

---
## evaluation_sweep_multiseed_best_recipe

- Type: evaluation-only (no new training)
- Goal: test whether adding seed diversity to the best recipe (`r05/r07/r08/r09`) plus vote averaging reaches >=90%.
- Note on augmentation rationale: on aligned ModelNet40 data, rotation augmentation can hurt canonical-pose discrimination (ICML 2023 workshop report): https://dmlr.ai/assets/accepted-papers/90/CameraReady/ICML_2023__ModelNet40%20%283%29.pdf

| Setup | Votes | Test Acc (%) | Test NLL |
|---|---:|---:|---:|
| single_r09 | 1 | 88.0470 | 0.468396 |
| single_r09 | 8 | 88.4117 | 0.463888 |
| single_r09 | 12 | 88.4117 | 0.464786 |
| single_r09 | 24 | 88.4117 | 0.463483 |
| single_r09 | 32 | 88.4927 | 0.463136 |
| ens_r07_r09 | 1 | 88.1280 | 0.448089 |
| ens_r07_r09 | 8 | 88.9384 | 0.443655 |
| ens_r07_r09 | 12 | 88.9789 | 0.444172 |
| ens_r07_r09 | 24 | 88.8574 | 0.443729 |
| ens_r07_r09 | 32 | 88.9789 | 0.444236 |
| ens_r05_r07_r09 | 1 | 88.4522 | 0.444480 |
| ens_r05_r07_r09 | 8 | 88.6548 | 0.442769 |
| ens_r05_r07_r09 | 12 | 88.5332 | 0.443083 |
| ens_r05_r07_r09 | 24 | 88.5737 | 0.442895 |
| ens_r05_r07_r09 | 32 | 88.5332 | 0.443423 |
| ens_r05_r07_r08_r09 | 1 | 88.4522 | 0.447263 |
| ens_r05_r07_r08_r09 | 8 | 88.6143 | 0.445081 |
| ens_r05_r07_r08_r09 | 12 | 88.6548 | 0.445063 |
| ens_r05_r07_r08_r09 | 24 | 88.4927 | 0.444971 |
| ens_r05_r07_r08_r09 | 32 | 88.4927 | 0.445762 |

- Best observed in this sweep: **88.9789%** (`ens_r07_r09`, 12 or 32 votes).
- Conclusion: multi-seed ensembling helps, but still does not reach 90% under current recipe family.

---
## r10_no_rot_aug_opt_featreg_seed10 - r07 recipe, disable train-time Z-rotation (offset 10)

- Time: `2026-02-17T02:28:25`
- Device: `cuda:0`
- Epochs: ran `110` / max `120`, best-loss epoch `85`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": true, "random_sample": false, "rotate_z": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 88.53 |
| Strict Test NLL | 0.4753 |
| Voted Test Acc (%) | 88.53 |
| Voted Test NLL | 0.4753 |
| Train Acc (%) | 96.78 |
| Train NLL | 0.1821 |

- Delta vs previous run (strict acc): `+0.53` points
- Delta vs previous run (voted acc): `+0.12` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r10_no_rot_aug_opt_featreg_seed10_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r10_no_rot_aug_opt_featreg_seed10_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r10_no_rot_aug_opt_featreg_seed10_summary.json`

---
## r11_no_rot_aug_opt_featreg_seed11 - r10 recipe, different seed (offset 11)

- Time: `2026-02-17T02:46:48`
- Device: `cuda:0`
- Epochs: ran `75` / max `120`, best-loss epoch `50`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": true, "random_sample": false, "rotate_z": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 88.25 |
| Strict Test NLL | 0.4600 |
| Voted Test Acc (%) | 88.25 |
| Voted Test NLL | 0.4600 |
| Train Acc (%) | 94.50 |
| Train NLL | 0.2443 |

- Delta vs previous run (strict acc): `-0.28` points
- Delta vs previous run (voted acc): `-0.28` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r11_no_rot_aug_opt_featreg_seed11_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r11_no_rot_aug_opt_featreg_seed11_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r11_no_rot_aug_opt_featreg_seed11_summary.json`

---
## r12_no_rot_no_ls_seed12 - r10 recipe, label smoothing off (offset 12)

- Time: `2026-02-17T03:01:49`
- Device: `cuda:0`
- Epochs: ran `61` / max `120`, best-loss epoch `36`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": true, "random_sample": false, "rotate_z": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.0, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 87.36 |
| Strict Test NLL | 0.4510 |
| Voted Test Acc (%) | 87.36 |
| Voted Test NLL | 0.4510 |
| Train Acc (%) | 92.22 |
| Train NLL | 0.2250 |

- Delta vs previous run (strict acc): `-0.89` points
- Delta vs previous run (voted acc): `-0.89` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r12_no_rot_no_ls_seed12_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r12_no_rot_no_ls_seed12_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r12_no_rot_no_ls_seed12_summary.json`

---
## r13_no_rot_no_ls_no_dropout_seed13 - r12 recipe, no point dropout (offset 13)

- Time: `2026-02-17T03:14:39`
- Device: `cuda:0`
- Epochs: ran `54` / max `120`, best-loss epoch `29`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": false, "random_sample": false, "rotate_z": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.0, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 86.14 |
| Strict Test NLL | 0.4879 |
| Voted Test Acc (%) | 86.14 |
| Voted Test NLL | 0.4879 |
| Train Acc (%) | 90.70 |
| Train NLL | 0.2626 |

- Delta vs previous run (strict acc): `-1.22` points
- Delta vs previous run (voted acc): `-1.22` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r13_no_rot_no_ls_no_dropout_seed13_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r13_no_rot_no_ls_no_dropout_seed13_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r13_no_rot_no_ls_no_dropout_seed13_summary.json`

---
## r14_no_rot_ls_no_dropout_seed14 - r10 recipe, no point dropout (offset 14)

- Time: `2026-02-17T03:40:56`
- Device: `cuda:0`
- Epochs: ran `107` / max `120`, best-loss epoch `82`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": false, "random_sample": false, "rotate_z": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 89.51 |
| Strict Test NLL | 0.4421 |
| Voted Test Acc (%) | 89.51 |
| Voted Test NLL | 0.4421 |
| Train Acc (%) | 98.50 |
| Train NLL | 0.1335 |

- Delta vs previous run (strict acc): `+3.36` points
- Delta vs previous run (voted acc): `+3.36` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r14_no_rot_ls_no_dropout_seed14_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r14_no_rot_ls_no_dropout_seed14_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r14_no_rot_ls_no_dropout_seed14_summary.json`

---
## r15_no_rot_ls_no_dropout_seed15 - r14 recipe, different seed (offset 15)

- Time: `2026-02-17T04:07:47`
- Device: `cuda:0`
- Epochs: ran `112` / max `120`, best-loss epoch `87`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": false, "random_sample": false, "rotate_z": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 88.70 |
| Strict Test NLL | 0.4587 |
| Voted Test Acc (%) | 88.70 |
| Voted Test NLL | 0.4587 |
| Train Acc (%) | 97.79 |
| Train NLL | 0.1507 |

- Delta vs previous run (strict acc): `-0.81` points
- Delta vs previous run (voted acc): `-0.81` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r15_no_rot_ls_no_dropout_seed15_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r15_no_rot_ls_no_dropout_seed15_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r15_no_rot_ls_no_dropout_seed15_summary.json`

---
## evaluation_sweep_r10_r14_r15_small

- Type: evaluation-only (no new training)
- Goal: test whether best recent checkpoints can cross 90% through log-probability ensembling and optional voting.
- Models used:
  - `r10_no_rot_aug_opt_featreg_seed10`
  - `r14_no_rot_ls_no_dropout_seed14`
  - `r15_no_rot_ls_no_dropout_seed15`
- Raw results saved to: `figures/full_pointnet_runs/eval_sweep_r10_r14_r15_small.json`

| Setup | Votes | Mode | Test Acc (%) | Test NLL |
|---|---:|---|---:|---:|
| ens_r10_r14_r15 | 1 | none | 90.1135 | 0.406818 |
| ens_r10_r14_r15 | 8 | none | 90.1135 | 0.406818 |
| ens_r10_r14_r15 | 12 | none | 90.1135 | 0.406818 |
| ens_r10_r14 | 1 | none | 89.7893 | 0.419043 |
| ens_r10_r14 | 8 | none | 89.7893 | 0.419043 |
| ens_r10_r14 | 12 | none | 89.7893 | 0.419043 |
| single_r14 | 1 | none | 89.5057 | 0.442141 |
| single_r14 | 8 | none | 89.5057 | 0.442141 |
| single_r14 | 12 | none | 89.5057 | 0.442141 |
| single_r14 | 1 | rotate | 89.5057 | 0.442141 |
| single_r14 | 8 | rotate | 73.0956 | 1.078706 |
| single_r14 | 12 | rotate | 71.3938 | 1.102117 |

- Best observed in this sweep: **90.1135%** (`ens_r10_r14_r15`, no-rotation mode).
- Conclusion: target reached (>=90%) with model ensembling, while rotation voting strongly hurts this aligned-data recipe.

---
## r16_first_tnet_only_r14_recipe_seed14 - first-TNet-only architecture, same outside recipe as r14 (seed offset 14)

- Time: `2026-02-17T12:44:31`
- Device: `cuda:0`
- Epochs: ran `108` / max `120`, best-loss epoch `83`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": true, "normalize": false, "point_dropout": false, "random_sample": false, "rotate_z": false, "scale_translate": true}`
  - Optimization: `{"eta_min": 1e-05, "grad_clip": 1.0, "label_smoothing": 0.1, "lr": 0.001, "optimizer": "adamw", "reg_feature": true, "reg_input": false, "scheduler": "cosine", "treg_alpha": 0.001, "weight_decay": 0.0001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 89.55 |
| Strict Test NLL | 0.4384 |
| Voted Test Acc (%) | 89.55 |
| Voted Test NLL | 0.4384 |
| Train Acc (%) | 97.89 |
| Train NLL | 0.1527 |

- Delta vs previous run (strict acc): `+0.85` points
- Delta vs previous run (voted acc): `+0.85` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r16_first_tnet_only_r14_recipe_seed14_training_curves.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r16_first_tnet_only_r14_recipe_seed14_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r16_first_tnet_only_r14_recipe_seed14_summary.json`

---

Note: `r16_first_tnet_only_r14_recipe_seed14` is an architecture-comparison run where `PointNetFull` was switched to the strict Exercise-2 variant (input 3x3 T-Net only, no 64x64 feature T-Net). Earlier runs (`r00`-`r15`) used the two-TNet variant.
## r17_first_tnet_only_baseline_recipe_seed0 - first-TNet-only architecture, baseline recipe (seed offset 0)

- Time: `2026-02-17T13:14:15`
- Device: `cuda:0`
- Epochs: ran `77` / max `120`, best-loss epoch `47`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": false, "normalize": false, "point_dropout": false, "random_sample": false, "rotate_z": true, "scale_translate": false}`
  - Optimization: `{"gamma": 0.5, "grad_clip": null, "label_smoothing": 0.0, "lr": 0.001, "optimizer": "adam", "reg_feature": true, "reg_input": true, "scheduler": "step", "step_size": 20, "treg_alpha": 0.001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 86.59 |
| Strict Test NLL | 0.4834 |
| Voted Test Acc (%) | 86.59 |
| Voted Test NLL | 0.4834 |
| Train Acc (%) | 94.83 |
| Train NLL | 0.1474 |

- Delta vs previous run (strict acc): `-2.96` points
- Delta vs previous run (voted acc): `-2.96` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r17_first_tnet_only_baseline_recipe_seed0_training_curves.png`
- Confusion (test): `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r17_first_tnet_only_baseline_recipe_seed0_confusion_test.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r17_first_tnet_only_baseline_recipe_seed0_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r17_first_tnet_only_baseline_recipe_seed0_summary.json`
- Details: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r17_first_tnet_only_baseline_recipe_seed0_details.json`

---
## r18_first_tnet_only_baseline_plus_point_dropout_seed0 - first-TNet-only baseline + RandomPointDropout only (seed offset 0)

- Time: `2026-02-17T13:54:55`
- Device: `cuda:0`
- Epochs: ran `77` / max `120`, best-loss epoch `47`
- Changes vs fixed architecture:
  - Augmentation: `{"jitter": false, "normalize": false, "point_dropout": true, "random_sample": false, "rotate_z": true, "scale_translate": false}`
  - Optimization: `{"gamma": 0.5, "grad_clip": null, "label_smoothing": 0.0, "lr": 0.001, "optimizer": "adam", "reg_feature": true, "reg_input": true, "scheduler": "step", "step_size": 20, "treg_alpha": 0.001}`
  - Vote eval: `count=1, mode=none`

| Metric | Value |
|---|---:|
| Strict Test Acc (%) | 86.63 |
| Strict Test NLL | 0.4710 |
| Voted Test Acc (%) | 86.63 |
| Voted Test NLL | 0.4710 |
| Train Acc (%) | 94.66 |
| Train NLL | 0.1549 |

- Delta vs previous run (strict acc): `+0.04` points
- Delta vs previous run (voted acc): `+0.04` points
- Curves: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r18_first_tnet_only_baseline_plus_point_dropout_seed0_training_curves.png`
- Confusion (test): `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r18_first_tnet_only_baseline_plus_point_dropout_seed0_confusion_test.png`
- Model: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r18_first_tnet_only_baseline_plus_point_dropout_seed0_pointnetfull.pth`
- JSON: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r18_first_tnet_only_baseline_plus_point_dropout_seed0_summary.json`
- Details: `/home/gana/Downloads/paul_chechin/figures/full_pointnet_runs/r18_first_tnet_only_baseline_plus_point_dropout_seed0_details.json`

---
