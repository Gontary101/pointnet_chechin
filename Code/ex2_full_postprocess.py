#!/usr/bin/env python
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from torchvision import transforms
from torch.utils.data import DataLoader

from pointnet import PointCloudData, PointMLP, PointNetBasic, PointNetFull, ToTensor


def evaluate_model(model, loader, device, use_full=False):
    model.eval()
    nll = torch.nn.NLLLoss(reduction='sum')
    total, correct = 0, 0
    loss_sum = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            x = batch['pointcloud'].to(device).float()
            y = batch['category'].to(device)

            out = model(x.transpose(1, 2))
            if isinstance(out, tuple):
                logits, m3x3 = out
                # keep same loss style as report comparison (NLL only)
                loss_sum += float(nll(logits, y))
            else:
                logits = out
                loss_sum += float(nll(logits, y))

            pred = logits.argmax(1)
            correct += int((pred == y).sum())
            total += y.numel()
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    return {
        'loss': loss_sum / max(total, 1),
        'acc': 100.0 * correct / max(total, 1),
        'n': int(total),
        'true': np.concatenate(all_true),
        'pred': np.concatenate(all_pred),
    }


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion(cm, labels, save_path, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row-normalized')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks[::2])
    ax.set_yticks(ticks[::2])
    ax.set_xticklabels([labels[i] for i in ticks[::2]], rotation=90, fontsize=7)
    ax.set_yticklabels([labels[i] for i in ticks[::2]], fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_architecture_full(save_path):
    fig, ax = plt.subplots(figsize=(23, 6))
    ax.axis('off')

    blocks = [
        ('Input\\n[B,3,1024]', '#cfe8ff'),
        ('T-Net (3x3)\\nalignment', '#f6cf71'),
        ('Transform\\nBMM', '#d9d9d9'),
        ('Conv1d 3->64\\nBN+ReLU', '#f7b267'),
        ('Conv1d 64->64\\nBN+ReLU', '#f7b267'),
        ('Conv1d 64->64\\nBN+ReLU', '#f7b267'),
        ('Conv1d 64->128\\nBN+ReLU', '#f7b267'),
        ('Conv1d 128->1024\\nBN+ReLU', '#f7b267'),
        ('MaxPool over\\npoints', '#d9d9d9'),
        ('FC 1024->512\\nBN+ReLU', '#96e6b3'),
        ('FC 512->256\\nBN+ReLU', '#96e6b3'),
        ('Dropout p=0.3', '#d9d9d9'),
        ('FC 256->40', '#f7b267'),
        ('LogSoftmax', '#cfe8ff'),
    ]

    x, y, w, h, gap = 0.3, 2.0, 1.35, 1.25, 0.22
    for i, (txt, color) in enumerate(blocks):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.08',
                              facecolor=color, edgecolor='#2a2a2a', linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=9.8)
        if i < len(blocks) - 1:
            ax.annotate('', xy=(x + w + gap - 0.03, y + h / 2), xytext=(x + w + 0.03, y + h / 2),
                        arrowprops=dict(arrowstyle='->', lw=1.3, color='#2a2a2a'))
        x += w + gap

    ax.text(0.3, 3.65, 'PointNetFull Architecture (Exercise 2 - with input T-Net 3x3)',
            fontsize=15, weight='bold')
    ax.set_xlim(0, x + 0.3)
    ax.set_ylim(1.6, 4.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    root = '/home/gana/Downloads/paul_chechin'
    data_root = os.path.join(root, 'data', 'ModelNet40_PLY')

    tf = transforms.Compose([ToTensor()])
    train_ds = PointCloudData(data_root, folder='train', transform=tf)
    test_ds = PointCloudData(data_root, folder='test', transform=tf)

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    class_names = [inv_classes[i] for i in range(len(inv_classes))]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    mlp = PointMLP(classes=len(class_names)).to(device)
    mlp.load_state_dict(torch.load(os.path.join(root, 'pointmlp_modelnet40.pth'), map_location=device))

    pnb = PointNetBasic(classes=len(class_names)).to(device)
    pnb.load_state_dict(torch.load(os.path.join(root, 'pointnetbasic_modelnet40.pth'), map_location=device))

    pnf = PointNetFull(classes=len(class_names)).to(device)
    pnf.load_state_dict(torch.load(os.path.join(root, 'pointnetfull_modelnet40.pth'), map_location=device))

    print('Evaluating MLP/Basic/Full...')
    m_tr = evaluate_model(mlp, train_loader, device)
    m_te = evaluate_model(mlp, test_loader, device)
    b_tr = evaluate_model(pnb, train_loader, device)
    b_te = evaluate_model(pnb, test_loader, device)
    f_tr = evaluate_model(pnf, train_loader, device)
    f_te = evaluate_model(pnf, test_loader, device)

    plot_architecture_full(os.path.join(root, 'architecture_pointnetfull.png'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = ['PointMLP', 'PointNetBasic', 'PointNetFull']
    x = np.arange(3)
    width = 0.35

    axes[0].bar(x - width/2, [m_tr['acc'], b_tr['acc'], f_tr['acc']], width, label='Train Acc', color='#4e79a7')
    axes[0].bar(x + width/2, [m_te['acc'], b_te['acc'], f_te['acc']], width, label='Test Acc', color='#e15759')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].bar(x - width/2, [m_tr['loss'], b_tr['loss'], f_tr['loss']], width, label='Train Loss', color='#59a14f')
    axes[1].bar(x + width/2, [m_te['loss'], b_te['loss'], f_te['loss']], width, label='Test Loss', color='#f28e2b')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel('NLL Loss')
    axes[1].set_title('Loss Comparison')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(root, 'comparison_mlp_basic_full.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)

    cm_full = confusion_matrix(f_te['true'], f_te['pred'], len(class_names))
    plot_confusion(cm_full, class_names, os.path.join(root, 'confusion_pointnetfull_test.png'),
                   'PointNetFull Test Confusion (normalized)')

    cm_basic = confusion_matrix(b_te['true'], b_te['pred'], len(class_names))
    per_cls_basic = np.diag(cm_basic) / np.maximum(cm_basic.sum(axis=1), 1)
    per_cls_full = np.diag(cm_full) / np.maximum(cm_full.sum(axis=1), 1)
    delta = per_cls_full - per_cls_basic
    top = np.argsort(np.abs(delta))[::-1][:15]

    fig, ax = plt.subplots(figsize=(12, 5))
    vals = delta[top] * 100.0
    ax.bar(np.arange(len(top)), vals, color=np.where(vals >= 0, '#59a14f', '#e15759'))
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels([class_names[i] for i in top], rotation=45, ha='right')
    ax.set_ylabel('Accuracy delta (percentage points)')
    ax.set_title('Top |Per-class Delta|: PointNetFull - PointNetBasic (test)')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(root, 'per_class_delta_pointnetfull_minus_basic.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'device': str(device),
        'pointmlp': {'train_acc': m_tr['acc'], 'test_acc': m_te['acc'], 'train_loss': m_tr['loss'], 'test_loss': m_te['loss']},
        'pointnetbasic': {'train_acc': b_tr['acc'], 'test_acc': b_te['acc'], 'train_loss': b_tr['loss'], 'test_loss': b_te['loss']},
        'pointnetfull': {'train_acc': f_tr['acc'], 'test_acc': f_te['acc'], 'train_loss': f_tr['loss'], 'test_loss': f_te['loss']},
    }
    with open(os.path.join(root, 'ex2_pointnetfull_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('PointMLP   train_acc=%.2f test_acc=%.2f train_loss=%.4f test_loss=%.4f' % (m_tr['acc'], m_te['acc'], m_tr['loss'], m_te['loss']))
    print('PointNetB  train_acc=%.2f test_acc=%.2f train_loss=%.4f test_loss=%.4f' % (b_tr['acc'], b_te['acc'], b_tr['loss'], b_te['loss']))
    print('PointNetF  train_acc=%.2f test_acc=%.2f train_loss=%.4f test_loss=%.4f' % (f_tr['acc'], f_te['acc'], f_tr['loss'], f_te['loss']))


if __name__ == '__main__':
    main()
