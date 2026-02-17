#!/usr/bin/env python
import argparse
import json
import math
import os
import random
import glob
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from pointnet import PointCloudData, PointNetFull, ToTensor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


class NormalizeUnitSphere(object):
    def __call__(self, pointcloud):
        pc = pointcloud.astype(np.float32, copy=False)
        pc = pc - np.mean(pc, axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(pc, axis=1))
        return pc / (scale + 1e-9)


class RandomRotationZ(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2.0 * math.pi
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return pointcloud @ rot.T


class RandomNoise(object):
    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def __call__(self, pointcloud):
        noise = np.random.normal(0.0, self.sigma, pointcloud.shape).astype(np.float32)
        return pointcloud + noise


class RandomScale(object):
    def __init__(self, low=0.8, high=1.25):
        self.low = low
        self.high = high

    def __call__(self, pointcloud):
        scale = np.random.uniform(self.low, self.high)
        return pointcloud * scale


class RandomTranslate(object):
    def __init__(self, shift=0.1):
        self.shift = shift

    def __call__(self, pointcloud):
        t = np.random.uniform(-self.shift, self.shift, (1, 3)).astype(np.float32)
        return pointcloud + t


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.03):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud):
        noise = np.clip(np.random.normal(0.0, self.sigma, pointcloud.shape), -self.clip, self.clip).astype(np.float32)
        return pointcloud + noise


class RandomPointDropout(object):
    def __init__(self, max_dropout_ratio=0.3):
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pointcloud):
        dropout_ratio = np.random.random() * self.max_dropout_ratio
        drop_idx = np.where(np.random.random(pointcloud.shape[0]) <= dropout_ratio)[0]
        if drop_idx.size > 0:
            pointcloud = pointcloud.copy()
            pointcloud[drop_idx, :] = pointcloud[0, :]
        return pointcloud


class RandomSample(object):
    def __init__(self, n=1024):
        self.n = n

    def __call__(self, pointcloud):
        n_points = pointcloud.shape[0]
        idx = np.random.choice(n_points, self.n, replace=(n_points < self.n))
        return pointcloud[idx]


class ShufflePoints(object):
    def __call__(self, pointcloud):
        pc = pointcloud.copy()
        np.random.shuffle(pc)
        return pc


def build_train_transform(cfg):
    ops = []
    if cfg.get('normalize', False):
        ops.append(NormalizeUnitSphere())
    if cfg.get('random_sample', False):
        ops.append(RandomSample(1024))
    if cfg.get('rotate_z', True):
        ops.append(RandomRotationZ())
    if cfg.get('scale_translate', False):
        ops.extend([RandomScale(), RandomTranslate()])
    if cfg.get('jitter', False):
        ops.append(RandomJitter())
    else:
        ops.append(RandomNoise())
    if cfg.get('point_dropout', False):
        ops.append(RandomPointDropout())
    ops.append(ShufflePoints())
    ops.append(ToTensor())
    return transforms.Compose(ops)


def build_test_transform(cfg):
    ops = []
    if cfg.get('normalize', False):
        ops.append(NormalizeUnitSphere())
    ops.append(ToTensor())
    return transforms.Compose(ops)


def rotate_z_tensor(points_bxn3, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    rot = points_bxn3.new_tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return torch.matmul(points_bxn3, rot.t())


def smoothed_nll_loss(log_probs, labels, smoothing=0.0):
    if smoothing <= 0.0:
        return F.nll_loss(log_probs, labels)
    nll = F.nll_loss(log_probs, labels, reduction='none')
    smooth = -log_probs.mean(dim=1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).mean()


def pointnet_full_loss(log_probs, labels, m3x3, m64x64=None, alpha=0.001, smoothing=0.0,
                       reg_input=True, reg_feature=True):
    bsize = log_probs.size(0)
    cls_loss = smoothed_nll_loss(log_probs, labels, smoothing=smoothing)

    reg = log_probs.new_tensor(0.0)

    if reg_input and m3x3 is not None:
        id3x3 = torch.eye(3, device=log_probs.device).unsqueeze(0).repeat(bsize, 1, 1)
        diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
        reg = reg + torch.norm(diff3x3) / float(bsize)

    if reg_feature and m64x64 is not None:
        id64x64 = torch.eye(64, device=log_probs.device).unsqueeze(0).repeat(bsize, 1, 1)
        diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
        reg = reg + torch.norm(diff64x64) / float(bsize)

    return cls_loss + alpha * reg


def make_optimizer_and_scheduler(model, cfg, epochs):
    optimizer_name = cfg.get('optimizer', 'adam').lower()
    lr = float(cfg.get('lr', 1e-3))

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(cfg.get('weight_decay', 1e-4)),
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler_name = cfg.get('scheduler', 'step').lower()
    if scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get('step_size', 20)),
            gamma=float(cfg.get('gamma', 0.5)),
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(cfg.get('eta_min', lr * 0.01)),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizer, scheduler


def evaluate(model, loader, device, vote_count=1, vote_mode='none', use_amp=True):
    model.eval()
    nll_sum = 0.0
    total = 0
    correct = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            x = batch['pointcloud'].to(device, non_blocking=True).float()  # [B,N,3]
            y = batch['category'].to(device, non_blocking=True)

            if vote_count <= 1:
                with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                    out = model(x.transpose(1, 2))
                    logits = out[0] if isinstance(out, tuple) else out
                loss = F.nll_loss(logits, y, reduction='sum')
                pred = logits.argmax(dim=1)
            else:
                probs_accum = None
                for v in range(vote_count):
                    xv = x
                    if vote_mode in ('rotate', 'rotate_resample'):
                        angle = (2.0 * math.pi * v) / float(vote_count)
                        xv = rotate_z_tensor(xv, angle)
                    if vote_mode == 'rotate_resample':
                        # Resample point order to vary max-pool winner paths.
                        idx = torch.randperm(xv.size(1), device=xv.device)
                        xv = xv[:, idx, :]

                    with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                        out = model(xv.transpose(1, 2))
                        logits = out[0] if isinstance(out, tuple) else out
                        probs = torch.exp(logits)
                    probs_accum = probs if probs_accum is None else (probs_accum + probs)

                probs_mean = probs_accum / float(vote_count)
                logits = torch.log(torch.clamp(probs_mean, min=1e-9))
                loss = F.nll_loss(logits, y, reduction='sum')
                pred = logits.argmax(dim=1)

            nll_sum += float(loss.item())
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())

    result = {
        'loss': nll_sum / max(total, 1),
        'acc': 100.0 * correct / max(total, 1),
        'n': int(total),
    }
    if all_true:
        result['true'] = np.concatenate(all_true).astype(np.int64)
        result['pred'] = np.concatenate(all_pred).astype(np.int64)
    else:
        result['true'] = np.array([], dtype=np.int64)
        result['pred'] = np.array([], dtype=np.int64)
    return result


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion(cm_norm, class_names, save_path, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row-normalized')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks[::2])
    ax.set_yticks(ticks[::2])
    ax.set_xticklabels([class_names[i] for i in ticks[::2]], rotation=90, fontsize=7)
    ax.set_yticklabels([class_names[i] for i in ticks[::2]], fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def train_one_run(run_cfg, common_cfg, data_root, out_dir):
    seed = int(common_cfg.get('seed', 42)) + int(run_cfg.get('seed_offset', 0))
    set_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(common_cfg.get('use_amp', True))

    train_tf = build_train_transform(run_cfg['augment'])
    test_tf = build_test_transform(run_cfg['augment'])

    train_ds = PointCloudData(data_root, folder='train', transform=train_tf)
    test_ds = PointCloudData(data_root, folder='test', transform=test_tf)
    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    class_names = [inv_classes[i] for i in range(len(inv_classes))]

    loader_kwargs = {
        'num_workers': int(common_cfg.get('num_workers', 12)),
        'pin_memory': bool(common_cfg.get('pin_memory', True)),
    }
    if loader_kwargs['num_workers'] > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=int(common_cfg.get('batch_size', 64)),
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(common_cfg.get('batch_size', 64)),
        shuffle=False,
        **loader_kwargs,
    )

    model = PointNetFull(classes=len(train_ds.classes)).to(device)

    epochs = int(common_cfg.get('epochs', 120))
    patience = int(common_cfg.get('patience', 25))
    min_delta = float(common_cfg.get('min_delta', 0.0))
    grad_clip = run_cfg['optim'].get('grad_clip', None)
    label_smoothing = float(run_cfg['optim'].get('label_smoothing', 0.0))

    optimizer, scheduler = make_optimizer_and_scheduler(model, run_cfg['optim'], epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == 'cuda'))

    best_test_loss = float('inf')
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': [],
    }

    t0 = datetime.now()
    print(f"\\n=== RUN {run_cfg['id']} :: {run_cfg['name']} ===")
    print(f"Device={device}, epochs={epochs}, patience={patience}, amp={use_amp}")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            x = batch['pointcloud'].to(device, non_blocking=True).float()
            y = batch['category'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                out = model(x.transpose(1, 2))
                if isinstance(out, tuple):
                    if len(out) == 3:
                        logits, m3x3, m64x64 = out
                    elif len(out) == 2:
                        logits, m3x3 = out
                        m64x64 = None
                    else:
                        raise ValueError(f"Unexpected model tuple output length: {len(out)}")
                    loss = pointnet_full_loss(
                        logits,
                        y,
                        m3x3,
                        m64x64,
                        alpha=float(run_cfg['optim'].get('treg_alpha', 0.001)),
                        smoothing=label_smoothing,
                        reg_input=bool(run_cfg['optim'].get('reg_input', True)),
                        reg_feature=bool(run_cfg['optim'].get('reg_feature', True)),
                    )
                else:
                    logits = out
                    loss = smoothed_nll_loss(logits, y, smoothing=label_smoothing)

            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            pred = logits.argmax(dim=1)
            train_total += int(y.numel())
            train_correct += int((pred == y).sum().item())

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)

        strict_test = evaluate(
            model,
            test_loader,
            device,
            vote_count=1,
            vote_mode='none',
            use_amp=use_amp,
        )

        avg_test_loss = strict_test['loss']
        test_acc = strict_test['acc']

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(float(optimizer.param_groups[0]['lr']))

        improved = avg_test_loss < (best_test_loss - min_delta)
        if improved:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"test_loss={avg_test_loss:.4f} test_acc={test_acc:.2f}% | "
            f"best_loss={best_test_loss:.4f}@{best_epoch} | "
            f"patience={epochs_no_improve}/{patience}"
        )

        scheduler.step()

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    strict_train = evaluate(model, train_loader, device, vote_count=1, vote_mode='none', use_amp=use_amp)
    strict_test = evaluate(model, test_loader, device, vote_count=1, vote_mode='none', use_amp=use_amp)

    vote_cfg = run_cfg.get('vote_eval', {'vote_count': 1, 'vote_mode': 'none'})
    vote_test = evaluate(
        model,
        test_loader,
        device,
        vote_count=int(vote_cfg.get('vote_count', 1)),
        vote_mode=str(vote_cfg.get('vote_mode', 'none')),
        use_amp=use_amp,
    )

    os.makedirs(out_dir, exist_ok=True)
    model_path = get_unique_path(os.path.join(out_dir, f"{run_cfg['id']}_pointnetfull.pth"))
    curves_path = get_unique_path(os.path.join(out_dir, f"{run_cfg['id']}_training_curves.png"))
    json_path = get_unique_path(os.path.join(out_dir, f"{run_cfg['id']}_summary.json"))
    details_path = get_unique_path(os.path.join(out_dir, f"{run_cfg['id']}_details.json"))
    confusion_path = get_unique_path(os.path.join(out_dir, f"{run_cfg['id']}_confusion_test.png"))

    torch.save(model.state_dict(), model_path)
    plot_training_history(history, curves_path, title_suffix=f" ({run_cfg['id']})")

    cm = confusion_matrix(strict_test['true'], strict_test['pred'], len(class_names))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plot_confusion(cm_norm, class_names, confusion_path, f"PointNetFull Test Confusion ({run_cfg['id']})")

    details = {
        'run_id': run_cfg['id'],
        'run_name': run_cfg['name'],
        'class_names': class_names,
        'history': history,
        'strict_train': {
            'loss': float(strict_train['loss']),
            'acc': float(strict_train['acc']),
            'n': int(strict_train['n']),
        },
        'strict_test': {
            'loss': float(strict_test['loss']),
            'acc': float(strict_test['acc']),
            'n': int(strict_test['n']),
        },
        'vote_test': {
            'loss': float(vote_test['loss']),
            'acc': float(vote_test['acc']),
            'n': int(vote_test['n']),
            'vote_count': int(vote_cfg.get('vote_count', 1)),
            'vote_mode': str(vote_cfg.get('vote_mode', 'none')),
        },
        'confusion': {
            'raw': cm.tolist(),
            'normalized': cm_norm.tolist(),
            'per_class_acc_percent': (np.diag(cm_norm) * 100.0).tolist(),
        },
    }
    with open(details_path, 'w') as f:
        json.dump(details, f, indent=2)

    elapsed = (datetime.now() - t0).total_seconds()
    result = {
        'run_id': run_cfg['id'],
        'run_name': run_cfg['name'],
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'device': str(device),
        'epochs_max': epochs,
        'epochs_ran': len(history['train_loss']),
        'best_epoch_by_loss': int(best_epoch),
        'best_test_loss': float(best_test_loss),
        'strict_train_loss': float(strict_train['loss']),
        'strict_train_acc': float(strict_train['acc']),
        'strict_test_loss': float(strict_test['loss']),
        'strict_test_acc': float(strict_test['acc']),
        'vote_test_loss': float(vote_test['loss']),
        'vote_test_acc': float(vote_test['acc']),
        'vote_count': int(vote_cfg.get('vote_count', 1)),
        'vote_mode': str(vote_cfg.get('vote_mode', 'none')),
        'config': run_cfg,
        'paths': {
            'model': model_path,
            'curves': curves_path,
            'summary': json_path,
            'details': details_path,
            'confusion_test': confusion_path,
        },
        'elapsed_seconds': elapsed,
    }

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def plot_training_history(history, save_path, title_suffix=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss over Training{title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Accuracy over Training{title_suffix}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def ensure_log_header(log_path):
    if os.path.exists(log_path):
        return

    header = """# PointNetFull Outside-Architecture Optimization Log

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
"""
    with open(log_path, 'w') as f:
        f.write(header)


def append_run_to_log(log_path, result, prev_result=None):
    cfg = result['config']
    aug = cfg['augment']
    opt = cfg['optim']

    lines = []
    lines.append(f"## {result['run_id']} - {result['run_name']}")
    lines.append('')
    lines.append(f"- Time: `{result['timestamp']}`")
    lines.append(f"- Device: `{result['device']}`")
    lines.append(f"- Epochs: ran `{result['epochs_ran']}` / max `{result['epochs_max']}`, best-loss epoch `{result['best_epoch_by_loss']}`")
    lines.append("- Changes vs fixed architecture:")
    lines.append(f"  - Augmentation: `{json.dumps(aug, sort_keys=True)}`")
    lines.append(f"  - Optimization: `{json.dumps(opt, sort_keys=True)}`")
    lines.append(f"  - Vote eval: `count={result['vote_count']}, mode={result['vote_mode']}`")
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---:|')
    lines.append(f"| Strict Test Acc (%) | {result['strict_test_acc']:.2f} |")
    lines.append(f"| Strict Test NLL | {result['strict_test_loss']:.4f} |")
    lines.append(f"| Voted Test Acc (%) | {result['vote_test_acc']:.2f} |")
    lines.append(f"| Voted Test NLL | {result['vote_test_loss']:.4f} |")
    lines.append(f"| Train Acc (%) | {result['strict_train_acc']:.2f} |")
    lines.append(f"| Train NLL | {result['strict_train_loss']:.4f} |")
    lines.append('')

    if prev_result is not None:
        d_strict = result['strict_test_acc'] - prev_result['strict_test_acc']
        d_vote = result['vote_test_acc'] - prev_result['vote_test_acc']
        lines.append(f"- Delta vs previous run (strict acc): `{d_strict:+.2f}` points")
        lines.append(f"- Delta vs previous run (voted acc): `{d_vote:+.2f}` points")
    else:
        lines.append('- Delta vs previous run: N/A (first run)')

    lines.append(f"- Curves: `{result['paths']['curves']}`")
    lines.append(f"- Confusion (test): `{result['paths']['confusion_test']}`")
    lines.append(f"- Model: `{result['paths']['model']}`")
    lines.append(f"- JSON: `{result['paths']['summary']}`")
    lines.append(f"- Details: `{result['paths']['details']}`")
    lines.append('')
    lines.append('---')
    lines.append('')

    with open(log_path, 'a') as f:
        f.write('\n'.join(lines))


def get_previous_result_from_summaries(out_dir, exclude_summary_path):
    summary_files = sorted(
        glob.glob(os.path.join(out_dir, '*_summary.json')),
        key=os.path.getmtime
    )
    summary_files = [p for p in summary_files if os.path.abspath(p) != os.path.abspath(exclude_summary_path)]
    if not summary_files:
        return None
    with open(summary_files[-1], 'r') as f:
        return json.load(f)


def load_default_runs():
    # Minimal-delta ablation chain.
    return [
        {
            'id': 'r00_baseline120',
            'name': 'Baseline recipe (StepLR+Adam, simple aug)',
            'seed_offset': 0,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'scale_translate': False,
                'jitter': False,
                'point_dropout': False,
            },
            'optim': {
                'optimizer': 'adam',
                'lr': 1e-3,
                'scheduler': 'step',
                'step_size': 20,
                'gamma': 0.5,
                'label_smoothing': 0.0,
                'grad_clip': None,
                'treg_alpha': 0.001,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r01_norm_sample',
            'name': 'Add normalize + random sample (same optimizer)',
            'seed_offset': 1,
            'augment': {
                'normalize': True,
                'random_sample': True,
                'scale_translate': False,
                'jitter': False,
                'point_dropout': False,
            },
            'optim': {
                'optimizer': 'adam',
                'lr': 1e-3,
                'scheduler': 'step',
                'step_size': 20,
                'gamma': 0.5,
                'label_smoothing': 0.0,
                'grad_clip': None,
                'treg_alpha': 0.001,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r02_aug_only',
            'name': 'Add scale/translate/jitter/dropout (still Adam+StepLR)',
            'seed_offset': 2,
            'augment': {
                'normalize': True,
                'random_sample': True,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adam',
                'lr': 1e-3,
                'scheduler': 'step',
                'step_size': 20,
                'gamma': 0.5,
                'label_smoothing': 0.0,
                'grad_clip': None,
                'treg_alpha': 0.001,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r03_opt_only',
            'name': 'Optimizer/scheduler/loss tune (AdamW+Cosine+LS+clip)',
            'seed_offset': 3,
            'augment': {
                'normalize': True,
                'random_sample': True,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r04_opt_plus_vote12',
            'name': 'Same as r03 + test-time voting (12 rotations)',
            'seed_offset': 4,
            'augment': {
                'normalize': True,
                'random_sample': True,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
            },
            'vote_eval': {'vote_count': 12, 'vote_mode': 'rotate'},
        },
        {
            'id': 'r05_nonorm_aug_opt',
            'name': 'No normalize/sample + strong aug + AdamW/Cosine/LS/clip',
            'seed_offset': 5,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
            },
            'vote_eval': {'vote_count': 12, 'vote_mode': 'rotate'},
        },
        {
            'id': 'r06_nonorm_aug_opt_lowreg',
            'name': 'r05 with lower T-Net regularization alpha=5e-4',
            'seed_offset': 6,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 5e-4,
            },
            'vote_eval': {'vote_count': 12, 'vote_mode': 'rotate'},
        },
        {
            'id': 'r07_nonorm_aug_opt_featreg_only',
            'name': 'r05 with regularization on 64x64 only (no 3x3 reg)',
            'seed_offset': 7,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 12, 'vote_mode': 'rotate'},
        },
        {
            'id': 'r08_nonorm_aug_opt_featreg_only_seed8',
            'name': 'r07 recipe, different seed (offset 8)',
            'seed_offset': 8,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 12, 'vote_mode': 'rotate'},
        },
        {
            'id': 'r09_nonorm_aug_opt_featreg_only_seed9',
            'name': 'r07 recipe, different seed (offset 9)',
            'seed_offset': 9,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': True,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 12, 'vote_mode': 'rotate'},
        },
        {
            'id': 'r10_no_rot_aug_opt_featreg_seed10',
            'name': 'r07 recipe, disable train-time Z-rotation (offset 10)',
            'seed_offset': 10,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r11_no_rot_aug_opt_featreg_seed11',
            'name': 'r10 recipe, different seed (offset 11)',
            'seed_offset': 11,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r12_no_rot_no_ls_seed12',
            'name': 'r10 recipe, label smoothing off (offset 12)',
            'seed_offset': 12,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.0,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r13_no_rot_no_ls_no_dropout_seed13',
            'name': 'r12 recipe, no point dropout (offset 13)',
            'seed_offset': 13,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': False,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.0,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r14_no_rot_ls_no_dropout_seed14',
            'name': 'r10 recipe, no point dropout (offset 14)',
            'seed_offset': 14,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': False,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r15_no_rot_ls_no_dropout_seed15',
            'name': 'r14 recipe, different seed (offset 15)',
            'seed_offset': 15,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': False,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r16_first_tnet_only_r14_recipe_seed14',
            'name': 'first-TNet-only architecture, same outside recipe as r14 (seed offset 14)',
            'seed_offset': 14,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': False,
                'scale_translate': True,
                'jitter': True,
                'point_dropout': False,
            },
            'optim': {
                'optimizer': 'adamw',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'eta_min': 1e-5,
                'label_smoothing': 0.1,
                'grad_clip': 1.0,
                'treg_alpha': 0.001,
                'reg_input': False,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r17_first_tnet_only_baseline_recipe_seed0',
            'name': 'first-TNet-only architecture, baseline recipe (seed offset 0)',
            'seed_offset': 0,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': True,
                'scale_translate': False,
                'jitter': False,
                'point_dropout': False,
            },
            'optim': {
                'optimizer': 'adam',
                'lr': 1e-3,
                'scheduler': 'step',
                'step_size': 20,
                'gamma': 0.5,
                'label_smoothing': 0.0,
                'grad_clip': None,
                'treg_alpha': 0.001,
                'reg_input': True,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
        {
            'id': 'r18_first_tnet_only_baseline_plus_point_dropout_seed0',
            'name': 'first-TNet-only baseline + RandomPointDropout only (seed offset 0)',
            'seed_offset': 0,
            'augment': {
                'normalize': False,
                'random_sample': False,
                'rotate_z': True,
                'scale_translate': False,
                'jitter': False,
                'point_dropout': True,
            },
            'optim': {
                'optimizer': 'adam',
                'lr': 1e-3,
                'scheduler': 'step',
                'step_size': 20,
                'gamma': 0.5,
                'label_smoothing': 0.0,
                'grad_clip': None,
                'treg_alpha': 0.001,
                'reg_input': True,
                'reg_feature': True,
            },
            'vote_eval': {'vote_count': 1, 'vote_mode': 'none'},
        },
    ]


def main():
    parser = argparse.ArgumentParser(description='PointNetFull outside-architecture optimization runs')
    parser.add_argument('--run-ids', type=str, default='all',
                        help='Comma-separated run ids to execute, or "all"')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = os.path.join(root, 'data', 'ModelNet40_PLY')
    out_dir = os.path.join(root, 'figures', 'full_pointnet_runs')
    log_path = os.path.join(root, 'full_pointnet_optimization_log.md')

    ensure_log_header(log_path)

    common_cfg = {
        'epochs': int(args.epochs),
        'patience': int(args.patience),
        'min_delta': 0.0,
        'batch_size': int(args.batch_size),
        'num_workers': int(args.num_workers),
        'pin_memory': torch.cuda.is_available(),
        'seed': int(args.seed),
        'use_amp': (not args.no_amp),
    }

    runs = load_default_runs()
    if args.run_ids.strip().lower() != 'all':
        wanted = {x.strip() for x in args.run_ids.split(',') if x.strip()}
        runs = [r for r in runs if r['id'] in wanted]
        missing = wanted - {r['id'] for r in runs}
        if missing:
            raise ValueError(f"Unknown run ids: {sorted(missing)}")

    prev = None
    all_results = []

    for r in runs:
        result = train_one_run(r, common_cfg, data_root, out_dir)
        prev_for_log = prev
        if prev_for_log is None:
            prev_for_log = get_previous_result_from_summaries(out_dir, result['paths']['summary'])
        append_run_to_log(log_path, result, prev_result=prev_for_log)
        all_results.append(result)
        prev = result

        print(
            f"[RESULT] {result['run_id']} strict_test_acc={result['strict_test_acc']:.2f}% "
            f"vote_test_acc={result['vote_test_acc']:.2f}%"
        )
        if result['strict_test_acc'] >= 90.0 or result['vote_test_acc'] >= 90.0:
            print(f"Target reached (>=90%) at run {result['run_id']}")
            break

    summary_path = os.path.join(root, 'full_pointnet_optimization_runs_latest.json')
    with open(summary_path, 'w') as f:
        json.dump({'common_cfg': common_cfg, 'results': all_results}, f, indent=2)

    print(f"Saved cumulative run summary: {summary_path}")
    print(f"Updated markdown log: {log_path}")


if __name__ == '__main__':
    main()
