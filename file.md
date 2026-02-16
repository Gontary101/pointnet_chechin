You’re *below* what vanilla PointNet usually hits on ModelNet40 (≈89% OA), and—importantly—**“PointNet + standard protocol” won’t realistically reach 95%**. Recent surveys/benchmarks show ModelNet40 is largely *saturated* around **94–95%** for point-based methods, with current strong results clustering there. 
So the playbook is:

1. **Fix pipeline/protocol issues to get you from ~87 → ~89–91** (classic PointNet territory)
2. **Move to a stronger local-geometry backbone + stronger aug/voting** to get **~92–94.7**
3. If you truly want **≥95**, you likely need **pretraining / 2D knowledge transfer / ensembling**, not just “PointNet tweaks”. ([ICML][1])

Below are the highest-leverage changes, in increasing “invasiveness”.

---

## 1) Quick wins that often add **+2–4 points** on your exact code

### A. Normalize every point cloud (train *and* test)

Even if ModelNet40 is often provided normalized, the particular PLYs you read may not be. Many ModelNet40 pipelines explicitly **center to zero-mean and scale to unit sphere**. ([PMC][2])
Add a transform like:

```python
class NormalizeUnitSphere:
    def __call__(self, pc):
        pc = pc - pc.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(pc, axis=1))
        return pc / (scale + 1e-9)
```

Put it **before** rotation/scale/jitter.

### B. Randomly resample a fixed number of points per shape every epoch

Your pipeline shuffles but doesn’t **resample**. Random resampling (e.g., 1024 out of 2048) is a big “free” augmentation on ModelNet40 and is part of many strong baselines. Some public PointNet/PointNet++ codebases attribute big gains to “uniform sampling” / sampling tricks. ([GitHub][3])

```python
class RandomSample:
    def __init__(self, n=1024):
        self.n = n
    def __call__(self, pc):
        N = pc.shape[0]
        idx = np.random.choice(N, self.n, replace=(N < self.n))
        return pc[idx]
```

Use `RandomSample(1024)` in **train transforms**, and for test-time voting (below).

### C. Don’t “free rotate” beyond yaw unless you’re sure

ModelNet40 is in a canonical pose per class. A careful analysis shows that **rotation augmentation can hurt** because it breaks that alignment (they report a notable drop when adding rotation aug). ([dmlr.ai][4])
Your `RandomRotation_z()` (yaw only) is usually safer than SO(3).

### D. Add test-time voting (this is *huge* for squeezing the last 1–2%)

A lot of top ModelNet40 numbers rely on a **voting** strategy (multiple resamplings/augs at test time, average logits). Papers explicitly call out that protocol differences like “rotation vote” / “repeated scaling vote” materially change accuracy. ([Proceedings of Machine Learning Research][5])

Minimal version:

* For each test shape, do **V=10–30** passes:

  * resample 1024 points
  * apply 1–3 random yaw rotations (and maybe a *tiny* jitter)
  * average predicted probabilities/logits

This alone can be the difference between “good” and “leaderboard-ish”.

---

## 2) Stronger augmentations than jitter/scale/dropout (often +0.5–2, sometimes more)

Classic aug (jitter/scale/translate/dropout) is table-stakes. Modern point-cloud aug mixes *two shapes* or *rigid parts* to improve generalization:

* **PointMixup (ECCV 2020)**: mixup for point clouds via optimal assignment / shortest-path interpolation. ([ECVA][6])
* **PointAugment (CVPR 2020)**: learns/chooses augmentation policies for point clouds. ([CVF Open Access][7])
* **RSMix (CVPR 2021)**: rigidly mix *subsets* to preserve local shape parts. ([CVF Open Access][8])
* **PointCutMix (2021)**: CutMix-style replacement based on correspondences. ([arXiv][9])




[1]: https://icml.cc/virtual/2025/poster/46672 "ICML Poster Exploring Vision Semantic Prompt for Efficient Point Cloud Understanding"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8402300/?utm_source=chatgpt.com "Effective Point Cloud Analysis Using Multi-Scale Features - PMC"
[3]: https://github.com/yanx27/Pointnet_Pointnet2_pytorch?utm_source=chatgpt.com "PointNet and PointNet++ implemented by pytorch (pure ..."
[4]: https://dmlr.ai/assets/accepted-papers/90/CameraReady/ICML_2023__ModelNet40%20%283%29.pdf "Point Cloud Classification with ModelNet40: What is left?"
[5]: https://proceedings.mlr.press/v139/goyal21a/goyal21a.pdf "Revisiting Point Cloud Shape Classification  with a Simple and Effective Baseline"
[6]: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480341.pdf?utm_source=chatgpt.com "PointMixup: Augmentation for Point Clouds"
[7]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_PointAugment_An_Auto-Augmentation_Framework_for_Point_Cloud_Classification_CVPR_2020_paper.pdf?utm_source=chatgpt.com "PointAugment: An Auto-Augmentation Framework for Point ..."
[8]: https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Regularization_Strategy_for_Point_Cloud_via_Rigidly_Mixed_Sample_CVPR_2021_paper.pdf?utm_source=chatgpt.com "Regularization Strategy for Point Cloud via Rigidly Mixed ..."
[9]: https://arxiv.org/pdf/2101.01461?utm_source=chatgpt.com "Regularization Strategy for Point Cloud Classification"
[10]: https://cgg.mff.cuni.cz/wp-content/uploads/2021/08/Survey-and-Evaluation-of-Neural-3D-Shape-Classification-Approaches-accepted-version.pdf?utm_source=chatgpt.com "Survey-and-Evaluation-of-Neural-3D-Shape-Classification- ..."
[11]: https://openaccess.thecvf.com/content/CVPR2022/papers/Ran_Surface_Representation_for_Point_Clouds_CVPR_2022_paper.pdf "Surface Representation for Point Clouds"
