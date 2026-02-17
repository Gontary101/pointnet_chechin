# Comparison: Full PointNet vs First-TNet-Only (Same Recipe)

## Setup

- Full architecture run (3x3 + 64x64 transforms):
  - `figures/full_pointnet_runs/r14_no_rot_ls_no_dropout_seed14_summary.json`
- Strict first-TNet-only architecture run (3x3 transform only):
  - `figures/full_pointnet_runs/r16_first_tnet_only_r14_recipe_seed14_summary.json`
- Outside-architecture recipe was kept the same (`r14` recipe):
  - augment: `rotate_z=False, scale_translate=True, jitter=True, point_dropout=False, normalize=False, random_sample=False`
  - optimization: `AdamW(lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR(eta_min=1e-5)`
  - regularization/other: `label_smoothing=0.1, grad_clip=1.0, treg_alpha=0.001`

## Results

| Model | Strict Test Acc (%) | Strict Test NLL | Best Epoch (by loss) | Epochs Ran |
|---|---:|---:|---:|---:|
| Full (3x3 + 64x64 T-Net) | 89.5057 | 0.442141 | 82 | 107 |
| First-TNet-only (3x3 only) | 89.5462 | 0.438412 | 83 | 108 |

Delta (first-TNet-only minus full):
- Accuracy: `+0.0405` points
- NLL: `-0.003729`

## Conclusion

With the same training recipe, the strict architecture requested in the PDF (first 3x3 T-Net only) is **at least as good** as the previous full two-TNet variant on this run, and slightly better on both strict accuracy and NLL.
