import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ===================== 全局配置（CPU运行）=====================
device = torch.device("cpu")
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001
NUM_CLASSES = 100
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===================== 1. 数据加载 =====================
def get_dataloader(batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    return train_loader, test_loader


# ===================== 2. ResNet 基础块 =====================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


# ===================== 3. 改进模块 =====================
class CoordAttention(nn.Module):
    """Residual Coordinate Attention. The residual gate avoids suppressing early weak features."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.att_h = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)
        self.att_w = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        _, _, h, w = x.shape
        x_h = self.avg_pool_h(x)
        x_w = self.avg_pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.shared(y)
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.att_h(y_h))
        a_w = torch.sigmoid(self.att_w(y_w.permute(0, 1, 3, 2)))
        return x * (1.0 + a_h * a_w)


def _build_dct_filter(pos, freq, size):
    value = math.cos(math.pi * freq * (pos + 0.5) / size) / math.sqrt(size)
    return value if freq == 0 else value * math.sqrt(2.0)


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, channels, dct_h=7, dct_w=7):
        super().__init__()
        freq_sel = [
            (0, 0), (0, 1), (1, 0), (1, 1),
            (0, 2), (2, 0), (1, 2), (2, 1),
            (2, 2), (0, 3), (3, 0), (1, 3),
            (3, 1), (2, 3), (3, 2), (3, 3)
        ]
        c_part = channels // len(freq_sel)
        weight = torch.zeros(channels, dct_h, dct_w)
        for i, (u, v) in enumerate(freq_sel):
            start = i * c_part
            end = channels if i == len(freq_sel) - 1 else (i + 1) * c_part
            for x in range(dct_h):
                for y in range(dct_w):
                    weight[start:end, x, y] = _build_dct_filter(x, u, dct_h) * _build_dct_filter(y, v, dct_w)
        self.register_buffer("weight", weight)
        self.dct_h = dct_h
        self.dct_w = dct_w

    def forward(self, x):
        if x.shape[-2:] != (self.dct_h, self.dct_w):
            x = nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        return torch.sum(x * self.weight.unsqueeze(0), dim=(2, 3))


class FcaLayer(nn.Module):
    """FcaNet-style frequency channel attention with a residual gate."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.dct = MultiSpectralDCTLayer(channels)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.dct(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * (1.0 + y)


class Res2Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, scale=4):
        super().__init__()
        self.scale = scale
        self.stride = stride
        width = planes // scale
        self.width = width
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True)
            )
            for _ in range(scale - 1)
        ])
        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if stride != 1 else nn.Identity()
        self.conv2 = nn.Conv2d(width * scale, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        splits = torch.split(out, self.width, dim=1)
        outputs = [self.pool(splits[0])]
        for i in range(1, self.scale):
            branch = splits[i]
            if self.stride == 1:
                branch = branch + outputs[i - 1]
            outputs.append(self.convs[i - 1](branch))
        out = torch.cat(outputs, dim=1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Conv2formerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        hidden = dim * mlp_ratio
        self.norm1 = nn.BatchNorm2d(dim)
        self.token_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1)
        )

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ===================== 4. 模型 =====================
class ResNet_Slim(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def classify(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def forward(self, x):
        return self.classify(self.forward_features(x))


class ResNet_CA(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = ResNet_Slim(BasicBlock, [2, 2, 2, 2], num_classes)
        self.ca = CoordAttention(512)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.ca(x)
        return self.backbone.classify(x)


class ResNet_FCA(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = ResNet_Slim(BasicBlock, [2, 2, 2, 2], num_classes)
        self.fca = FcaLayer(512)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.fca(x)
        return self.backbone.classify(x)


def Res2Net_Slim():
    return ResNet_Slim(Res2Block, [2, 2, 2, 2])


class ResNet_Conv2former(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = ResNet_Slim(BasicBlock, [2, 2, 2, 2], num_classes)
        self.conv2f3 = Conv2formerBlock(256)
        self.conv2f4 = Conv2formerBlock(512)

    def forward(self, x):
        x = torch.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.conv2f3(x)
        x = self.backbone.layer4(x)
        x = self.conv2f4(x)
        return self.backbone.classify(x)


# ===================== 5. 训练测试 =====================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def test(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def append_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "best_acc"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def build_models():
    return {
        "Base_ResNet": ResNet_Slim(BasicBlock, [2, 2, 2, 2]),
        "ResNet_CA": ResNet_CA(),
        "ResNet_FCA": ResNet_FCA(),
        "Res2Net": Res2Net_Slim(),
        "ResNet_Conv2former": ResNet_Conv2former()
    }


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-100 CPU training for improved ResNet variants.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Res2Net", "ResNet_Conv2former"],
        help="Model names to train, or 'all'."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="yangleez/results/0_5_metrics.csv")
    return parser.parse_args()


# ===================== 6. 主程序 =====================
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    train_loader, test_loader = get_dataloader(args.batch_size, args.num_workers)
    all_models = build_models()
    selected = list(all_models.keys()) if args.models == ["all"] else args.models

    for name in selected:
        if name not in all_models:
            raise ValueError(f"Unknown model: {name}. Available models: {list(all_models.keys())}")

        print(f"\n========== 训练模型: {name} ==========")
        model = all_models[name].to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_acc = 0.0
        csv_rows = []
        start_time = time.time()
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            test_loss, test_acc = test(model, test_loader, criterion)
            scheduler.step()
            best_acc = max(best_acc, test_acc)
            csv_rows.append({
                "model": name,
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.2f}",
                "test_loss": f"{test_loss:.4f}",
                "test_acc": f"{test_acc:.2f}",
                "best_acc": f"{best_acc:.2f}"
            })
            print(
                f"Epoch {epoch + 1:2d} | Train Loss:{train_loss:.3f} | "
                f"Train Acc:{train_acc:.2f}% | Test Loss:{test_loss:.3f} | Test Acc:{test_acc:.2f}%"
            )

        append_csv(args.output, csv_rows)
        elapsed = time.time() - start_time
        print(f"\n【{name}】最优准确率 = {best_acc:.2f}% | 用时 = {elapsed / 60:.1f} min\n")
