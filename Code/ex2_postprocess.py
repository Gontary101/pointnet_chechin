#!/usr/bin/env python
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from torchvision import transforms
from torch.utils.data import DataLoader

from pointnet import PointCloudData, PointMLP, PointNetBasic, ToTensor


def evaluate_model(model, loader, device):
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
            loss_sum += float(nll(out, y))
            pred = out.argmax(1)
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


def plot_architecture(save_path):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.axis('off')

    blocks = [
        ('Input\\n[B, 3, 1024]', '#cfe8ff'),
        ('Conv1d 3->64\\nBN + ReLU', '#f7b267'),
        ('Conv1d 64->64\\nBN + ReLU', '#f7b267'),
        ('Conv1d 64->64\\nBN + ReLU', '#f7b267'),
        ('Conv1d 64->128\\nBN + ReLU', '#f7b267'),
        ('Conv1d 128->1024\\nBN + ReLU', '#f7b267'),
        ('MaxPool\\nover points', '#d9d9d9'),
        ('Flatten\\n[B, 1024]', '#d9d9d9'),
        ('FC 1024->512\\nBN + ReLU', '#96e6b3'),
        ('FC 512->256\\nBN + ReLU', '#96e6b3'),
        ('Dropout\\np=0.3', '#d9d9d9'),
        ('FC 256->40', '#f7b267'),
        ('LogSoftmax', '#cfe8ff'),
    ]

    x = 0.3
    y = 2.0
    w = 1.25
    h = 1.3
    gap = 0.25

    for i, (txt, color) in enumerate(blocks):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.08',
                              facecolor=color, edgecolor='#2a2a2a', linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=10)
        if i < len(blocks) - 1:
            ax.annotate('', xy=(x + w + gap - 0.03, y + h / 2), xytext=(x + w + 0.03, y + h / 2),
                        arrowprops=dict(arrowstyle='->', lw=1.4, color='#2a2a2a'))
        x += w + gap

    ax.text(0.3, 3.65, 'PointNetBasic Architecture (Exercise 2 - Basic PointNet, no T-Net)',
            fontsize=15, weight='bold')
    ax.set_xlim(0, x + 0.3)
    ax.set_ylim(1.6, 4.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = os.path.join(root, 'data', 'ModelNet40_PLY')
    fig_dir = os.path.join(root, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    test_tf = transforms.Compose([ToTensor()])
    train_ds = PointCloudData(data_root, folder='train', transform=test_tf)
    test_ds = PointCloudData(data_root, folder='test', transform=test_tf)

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

    print('Evaluating PointMLP...')
    mlp_train = evaluate_model(mlp, train_loader, device)
    mlp_test = evaluate_model(mlp, test_loader, device)

    print('Evaluating PointNetBasic...')
    pnb_train = evaluate_model(pnb, train_loader, device)
    pnb_test = evaluate_model(pnb, test_loader, device)

    plot_architecture(os.path.join(fig_dir, 'architecture_pointnetbasic.png'))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = ['PointMLP', 'PointNetBasic']
    x = np.arange(2)
    width = 0.35

    axes[0].bar(x - width / 2, [mlp_train['acc'], pnb_train['acc']], width, label='Train Acc', color='#4e79a7')
    axes[0].bar(x + width / 2, [mlp_test['acc'], pnb_test['acc']], width, label='Test Acc', color='#e15759')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].bar(x - width / 2, [mlp_train['loss'], pnb_train['loss']], width, label='Train Loss', color='#59a14f')
    axes[1].bar(x + width / 2, [mlp_test['loss'], pnb_test['loss']], width, label='Test Loss', color='#f28e2b')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel('NLL Loss')
    axes[1].set_title('Loss Comparison')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'comparison_mlp_vs_pointnetbasic.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)

    cm_mlp = confusion_matrix(mlp_test['true'], mlp_test['pred'], len(class_names))
    cm_pnb = confusion_matrix(pnb_test['true'], pnb_test['pred'], len(class_names))

    plot_confusion(cm_mlp, class_names, os.path.join(fig_dir, 'confusion_pointmlp_test.png'),
                   'PointMLP Test Confusion (normalized)')
    plot_confusion(cm_pnb, class_names, os.path.join(fig_dir, 'confusion_pointnetbasic_test.png'),
                   'PointNetBasic Test Confusion (normalized)')

    per_cls_mlp = np.diag(cm_mlp) / np.maximum(cm_mlp.sum(axis=1), 1)
    per_cls_pnb = np.diag(cm_pnb) / np.maximum(cm_pnb.sum(axis=1), 1)
    delta = per_cls_pnb - per_cls_mlp
    top = np.argsort(np.abs(delta))[::-1][:15]

    fig, ax = plt.subplots(figsize=(12, 5))
    vals = delta[top] * 100.0
    ax.bar(np.arange(len(top)), vals, color=np.where(vals >= 0, '#59a14f', '#e15759'))
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels([class_names[i] for i in top], rotation=45, ha='right')
    ax.set_ylabel('Accuracy delta (percentage points)')
    ax.set_title('Top |Per-class Delta|: PointNetBasic - PointMLP (test)')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'per_class_delta_pointnetbasic_minus_mlp.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'device': str(device),
        'pointmlp': {
            'train_acc': mlp_train['acc'],
            'test_acc': mlp_test['acc'],
            'train_loss': mlp_train['loss'],
            'test_loss': mlp_test['loss'],
        },
        'pointnetbasic': {
            'train_acc': pnb_train['acc'],
            'test_acc': pnb_test['acc'],
            'train_loss': pnb_train['loss'],
            'test_loss': pnb_test['loss'],
        }
    }

    with open(os.path.join(root, 'ex2_pointnetbasic_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('PointMLP  : train_acc=%.2f test_acc=%.2f train_loss=%.4f test_loss=%.4f' %
          (mlp_train['acc'], mlp_test['acc'], mlp_train['loss'], mlp_test['loss']))
    print('PointNetB : train_acc=%.2f test_acc=%.2f train_loss=%.4f test_loss=%.4f' %
          (pnb_train['acc'], pnb_test['acc'], pnb_train['loss'], pnb_test['loss']))


if __name__ == '__main__':
    main()
