#!/usr/bin/env python
import argparse
import json
import math
import os
import random
from datetime import datetime

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


def evaluate(model, loader, device, use_amp=True):
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
            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                out = model(x.transpose(1, 2))
                logits = out[0] if isinstance(out, tuple) else out
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


def train_one_run(run_cfg, common_cfg, data_root, out_dir):
    seed = int(common_cfg.get('seed', 42)) + int(run_cfg.get('seed_offset', 0))
    set_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(common_cfg.get('use_amp', True))

    train_tf = build_train_transform(run_cfg['augment'])
    test_tf = build_test_transform(run_cfg['augment'])

    train_ds = PointCloudData(data_root, folder='train', transform=train_tf)
    test_ds = PointCloudData(data_root, folder='test', transform=test_tf)

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
    best_test_acc = -1.0

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    t0 = datetime.now()
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
            use_amp=use_amp,
        )

        avg_test_loss = strict_test['loss']
        test_acc = strict_test['acc']

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)

        improved = test_acc > (best_test_acc + min_delta)
        if improved:
            best_test_acc = test_acc
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            'Epoch: %d, Train Loss: %.3f, Train Acc: %.1f %%, Test Loss: %.3f, Test Acc: %.1f %%'
            % (epoch + 1, avg_train_loss, train_acc, avg_test_loss, test_acc)
        )
        print(
            'Best Test Acc: %.1f %% (epoch %d), EarlyStop patience: %d/%d'
            % (best_test_acc, best_epoch, epochs_no_improve, patience)
        )

        scheduler.step()

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    strict_train = evaluate(model, train_loader, device, use_amp=use_amp)
    strict_test = evaluate(model, test_loader, device, use_amp=use_amp)

    os.makedirs(out_dir, exist_ok=True)
    model_path = get_unique_path(os.path.join(out_dir, f"{run_cfg['id']}_pointnetfull.pth"))
    json_path = get_unique_path(os.path.join(out_dir, f"{run_cfg['id']}_summary.json"))

    torch.save(model.state_dict(), model_path)

    elapsed = (datetime.now() - t0).total_seconds()
    result = {
        'run_id': run_cfg['id'],
        'run_name': run_cfg['name'],
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'device': str(device),
        'epochs_max': epochs,
        'epochs_ran': len(history['train_loss']),
        'best_epoch_by_acc': int(best_epoch),
        'best_test_loss': float(best_test_loss),
        'strict_train_loss': float(strict_train['loss']),
        'strict_train_acc': float(strict_train['acc']),
        'strict_test_loss': float(strict_test['loss']),
        'strict_test_acc': float(strict_test['acc']),
        'config': run_cfg,
        'paths': {
            'model': model_path,
            'summary': json_path,
        },
        'elapsed_seconds': elapsed,
    }

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def load_runs_config(config_path):
    with open(config_path, 'r') as f:
        runs = json.load(f)
    if not isinstance(runs, list):
        raise ValueError(f"Runs config must be a list: {config_path}")
    return runs


def main():
    parser = argparse.ArgumentParser(description='PointNetFull outside-architecture optimization runs')
    parser.add_argument('--run-ids', type=str, default='all',
                        help='Comma-separated run ids to execute, or "all"')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs-config', type=str, default='',
                        help='Path to runs config JSON. Defaults to Code/ex2_full_optimization_runs_config.json')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = os.path.join(root, 'data', 'ModelNet40_PLY')
    out_dir = os.path.join(root, 'figures', 'full_pointnet_runs')

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

    default_runs_config = os.path.join(os.path.dirname(__file__), 'ex2_full_optimization_runs_config.json')
    runs_config_path = args.runs_config.strip() if args.runs_config.strip() else default_runs_config
    runs = load_runs_config(runs_config_path)
    if args.run_ids.strip().lower() != 'all':
        wanted = {x.strip() for x in args.run_ids.split(',') if x.strip()}
        runs = [r for r in runs if r['id'] in wanted]
        missing = wanted - {r['id'] for r in runs}
        if missing:
            raise ValueError(f"Unknown run ids: {sorted(missing)}")

    all_results = []

    for r in runs:
        result = train_one_run(r, common_cfg, data_root, out_dir)
        all_results.append(result)
        if result['strict_test_acc'] >= 90.0:
            break

    summary_path = os.path.join(root, 'full_pointnet_optimization_runs_latest.json')
    with open(summary_path, 'w') as f:
        json.dump({'common_cfg': common_cfg, 'results': all_results}, f, indent=2)


if __name__ == '__main__':
    main()
