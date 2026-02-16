#!/usr/bin/env python
# PointNet for point cloud classification
#
# -- Paul CHECCHIN - 5/11/2021
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display needed)
import matplotlib.pyplot as plt

# Import functions to read and write ply files
from ply import write_ply, read_ply


class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta),      0],
                               [math.sin(theta),  math.cos(theta),      0],
                               [0,                              0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


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
        translation = np.random.uniform(-self.shift, self.shift, (1, 3))
        return pointcloud + translation


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.03):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud):
        noise = np.clip(np.random.normal(0.0, self.sigma, pointcloud.shape),
                        -self.clip, self.clip)
        return pointcloud + noise


class RandomPointDropout(object):
    def __init__(self, max_dropout_ratio=0.3):
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pointcloud):
        dropout_ratio = np.random.random() * self.max_dropout_ratio
        drop_idx = np.where(np.random.random(pointcloud.shape[0]) <= dropout_ratio)[0]
        if drop_idx.size > 0:
            pointcloud[drop_idx, :] = pointcloud[0, :]
        return pointcloud


class ShufflePoints(object):
    def __call__(self, pointcloud):
        np.random.shuffle(pointcloud)
        return pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),
                               RandomScale(),
                               RandomTranslate(),
                               RandomJitter(),
                               RandomPointDropout(),
                               ShufflePoints(),
                               ToTensor()])


class PointCloudData(Dataset):
    def __init__(self,
                 root_dir,
                 folder="train",
                 transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir))
                   if os.path.isdir(root_dir + "/" + dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []
        self.cache = {}
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    sample = {}
                    sample['ply_path'] = new_dir+"/"+file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.cache:
            pointcloud_np = self.cache[idx].copy()
        else:
            ply_path = self.files[idx]['ply_path']
            data = read_ply(ply_path)
            pointcloud_np = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
            self.cache[idx] = pointcloud_np
            pointcloud_np = pointcloud_np.copy()
        category = self.files[idx]['category']
        pointcloud = self.transforms(pointcloud_np)
        return {'pointcloud': pointcloud, 'category': self.classes[category]}


class PointMLP(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        x = input.reshape(input.size(0), -1)  # we flatten the input: [batch, 3, 1024] to [batch, 3072]
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.log_softmax(x)  
        return x
class PointNetBasic(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        # Shared MLP over points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)

        # Symmetric max pooling to get global feature
        self.maxpool = nn.MaxPool1d(1024)
        self.flatten = nn.Flatten(1)

        # Global MLP classifier
        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)
        # Start from an identity transform at initialization.
        nn.init.constant_(self.fc3.weight, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, input):
        bsize = input.size(0)

        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        id_matrix = torch.eye(self.k, device=input.device).reshape(1, self.k * self.k).repeat(bsize, 1)
        x = x + id_matrix
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFull(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        m3x3 = self.input_transform(input)
        x = torch.bmm(m3x3, input)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        m64x64 = self.feature_transform(x)
        x = torch.bmm(m64x64, x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x, m3x3, m64x64


def smoothed_nll_loss(outputs, labels, label_smoothing=0.0):
    nll = F.nll_loss(outputs, labels, reduction='mean')
    if label_smoothing <= 0.0:
        return nll
    smooth = -outputs.mean(dim=1).mean()
    return (1.0 - label_smoothing) * nll + label_smoothing * smooth


def basic_loss(outputs, labels, label_smoothing=0.0):
    return smoothed_nll_loss(outputs, labels, label_smoothing=label_smoothing)


def pointnet_full_loss(outputs, labels, m3x3, m64x64=None, alpha=0.001, label_smoothing=0.0):
    bsize = outputs.size(0)
    reg = outputs.new_tensor(0.0)

    # In full PointNet, orthogonality regularization is primarily applied to
    # the feature transform (64x64). Keep a 3x3 fallback for compatibility.
    if m64x64 is not None:
        id64x64 = torch.eye(64, device=outputs.device).unsqueeze(0).repeat(bsize, 1, 1)
        diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
        reg = reg + torch.norm(diff64x64) / float(bsize)
    elif m3x3 is not None:
        id3x3 = torch.eye(3, device=outputs.device).unsqueeze(0).repeat(bsize, 1, 1)
        diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
        reg = reg + torch.norm(diff3x3) / float(bsize)

    return smoothed_nll_loss(outputs, labels, label_smoothing=label_smoothing) + alpha * reg


def train(model, device, train_loader, test_loader=None, epochs=250, patience=30, min_delta=0.0,
          label_smoothing=0.1, lr=0.001, weight_decay=1e-4, grad_clip=1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=epochs,
                                                           eta_min=lr * 0.01)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_test_loss = float('inf')
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    
    # History tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device, non_blocking=True).float(), data['category'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                model_out = model(inputs.transpose(1, 2))
                if isinstance(model_out, tuple):
                    if len(model_out) == 3:
                        outputs, m3x3, m64x64 = model_out
                    else:
                        outputs, m3x3 = model_out
                        m64x64 = None
                    loss = pointnet_full_loss(outputs, labels, m3x3, m64x64,
                                              label_smoothing=label_smoothing)
                else:
                    outputs = model_out
                    loss = basic_loss(outputs, labels,
                                      label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device, non_blocking=True).float(), data['category'].to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        model_out = model(inputs.transpose(1, 2))
                        if isinstance(model_out, tuple):
                            if len(model_out) == 3:
                                outputs, m3x3, m64x64 = model_out
                            else:
                                outputs, m3x3 = model_out
                                m64x64 = None
                            loss = pointnet_full_loss(outputs, labels, m3x3, m64x64,
                                                      label_smoothing=0.0)
                        else:
                            outputs = model_out
                            loss = basic_loss(outputs, labels,
                                              label_smoothing=0.0)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_acc = 100. * test_correct / test_total
            history['test_loss'].append(avg_test_loss)
            history['test_acc'].append(test_acc)

            if avg_test_loss < (best_test_loss - min_delta):
                best_test_loss = avg_test_loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            print('Epoch: %d, Train Loss: %.3f, Train Acc: %.1f %%, Test Loss: %.3f, Test Acc: %.1f %%' 
                  % (epoch+1, avg_train_loss, train_acc, avg_test_loss, test_acc))
            print('Best Test Loss: %.3f (epoch %d), EarlyStop patience: %d/%d'
                  % (best_test_loss, best_epoch, epochs_no_improve, patience))

            if epochs_no_improve >= patience:
                print('Early stopping triggered at epoch %d' % (epoch + 1))
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_epoch, best_test_loss


def plot_training_history(history, save_path=None):
    """Plot training and test loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if history['test_loss']:
        ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss over Training', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    if history['test_acc']:
        ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy over Training', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()  


if __name__ == '__main__':
    t0 = time.time()
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = os.path.join(project_root, "data", "ModelNet40_PLY")
    train_ds = PointCloudData(data_root)
    test_ds = PointCloudData(data_root,
                             folder='test',
                             transform=transforms.Compose([ToTensor()]))

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    num_workers = min(16, os.cpu_count() or 1)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    train_loader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(dataset=test_ds, batch_size=128, shuffle=False, **loader_kwargs)

    #model = PointMLP()
    # model = PointNetBasic()
    model = PointNetFull()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ",
          sum([np.prod(p.size()) for p in model_parameters]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model.to(device)

    history, best_epoch, best_test_loss = train(model, device, train_loader, test_loader,
                                                epochs=120, patience=30)
    print("Total time for training : ", time.time() - t0)
    print("Best validation test loss: %.3f at epoch %d" % (best_test_loss, best_epoch))

    model_name = model.__class__.__name__.lower()
    model_path = f"{model_name}_modelnet40.pth"
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    curves_path = os.path.join(figures_dir, f"{model_name}_training_curves.png")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot and save training curves
    plot_training_history(history, save_path=curves_path)
