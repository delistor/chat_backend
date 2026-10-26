#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DRAEM 单文件复刻（对齐 anomalib v2.3.0 实现）—— 训练 / 测试 / 推理一体。

DRAEM（Zavrtanik et al., ICCV 2021）是【从零训练】的重建+判别式方法:
不依赖任何 ImageNet 预训练骨干, 用 Perlin 噪声合成假缺陷自监督训练, 在
pixel 级指标上通常强于 FastFlow(官方论文 MVTec 平均 pixel-AUROC 97.3 vs FastFlow 96.5)。

用法（与 fastflow.py 完全同构）:
    # 训练 + 测试
    python draem.py --category bottle

    # 更高质量: 提供 DTD 纹理库作合成缺陷源(官方协议), 如 ./datasets/dtd
    python draem.py --category bottle --dtd-dir datasets/dtd

    # 用已训权重推理
    python draem.py --predict D:/some.png --weights results/bottle/seed0/best.pt

依赖: torch, torchvision, numpy, scikit-learn, pillow, kornia(anomalib 同款损失), matplotlib(可选)

对齐要点（anomalib v2.3.0 src/anomalib/models/image/draem/, 拷贝件见
reference/anomalib_fastflow_src/ 或 pip show anomalib 的源码包）:
  1. 重建子网 U-Net base_width=128 + 判别子网 U-Net(6ch输入->2ch输出, base_width=64),
     保留 anomalib/原版代码的全部细节(判别器 mp2 复用 mp3、重建解码器无跳连拼接、
     Upsample align_corners=True)。
  2. 损失 = MSE + 2*SSIM(window=11) + Focal(alpha=1)（loss.py DraemLoss, kornia 实现）。
  3. 训练输入【不做 ImageNet 归一化】, 仅 Resize((S,S)) 后保持 [0,1]
     （lightning_model.configure_pre_processor 明确禁止 Normalize）。
  4. 合成异常: PerlinAnomalyGenerator —— Perlin 噪声(2^0..2^5 随机尺度, [-90,90]°旋转,
     >0.5 二值化得 mask) 生成/或 DTD 源图, 经 3 个随机光度增广后按
     beta~U(0.1,1.0) 与原图混合; 每张图 50% 概率被增广。
  5. 优化: Adam(lr=1e-4) 全参数 + MultiStepLR(milestones=[400,600], gamma=0.1),
     梯度裁剪 0。
  6. 推理: anomaly_map = softmax(logits)[:, 1], 图像分数 = map.max()。
默认协议: 256x256, batch=32, 700 epochs, 早停 patience=20(监控 val pixel_AUROC, 验证集=测试集,
与 anomalib MVTec 协议一致)。DTD 未提供时退化为纯 Perlin 噪声纹理(指标会略降, 属预期)。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import io as tv_io
from torchvision import transforms as v2
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2 as T

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_SHA256 = "e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 数据（MVTec 格式; 注意 DRAEM 不做 ImageNet 归一化, 保持 [0,1]）
# --------------------------------------------------------------------------- #
class MVTecLike(Dataset):
    def __init__(self, root: Path, category: str, split: str, image_size: int):
        self.category_dir = root / category
        self.split = split
        self.size = image_size
        self.samples: list[tuple[Path, int, Path | None, str]] = []
        if split == "train":
            for p in sorted((self.category_dir / "train" / "good").iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, 0, None, "good"))
        else:
            for sub in sorted((self.category_dir / "test").iterdir()):
                if not sub.is_dir():
                    continue
                for p in sorted(sub.iterdir()):
                    if p.suffix.lower() not in IMG_EXTS:
                        continue
                    label = 0 if sub.name == "good" else 1
                    mask = self.category_dir / "ground_truth" / sub.name / f"{p.stem}_mask.png"
                    self.samples.append((p, label, mask if label else None, sub.name))

    def __len__(self):
        return len(self.samples)

    def _resize(self, t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t, size=(self.size, self.size), mode="bilinear", antialias=True,
                             align_corners=False)

    def __getitem__(self, idx: int):
        path, label, mask_path, defect_type = self.samples[idx]
        img = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)
        img = self._resize(img[None])[0].float().div_(255.0)  # [0,1], 无 Normalize
        mask = torch.zeros(self.size, self.size, dtype=torch.uint8)
        if mask_path is not None and mask_path.exists():
            m = torch.from_numpy(np.array(Image.open(mask_path).convert("L")))[None, None]
            mask = self._resize(m)[0, 0].to(torch.uint8)
        return img, mask, label, defect_type


# --------------------------------------------------------------------------- #
# 合成异常增广（忠实移植 anomalib PerlinAnomalyGenerator）
# --------------------------------------------------------------------------- #
def generate_perlin_noise(height: int, width: int, device=None) -> torch.Tensor:
    """anomalib data/utils/generators/perlin.py: 2^0..2^5 随机尺度的 Perlin 噪声。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_scale, max_scale = 0, 6
    scalex = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()
    scaley = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()

    def nextpow2(v):
        return int(2 ** torch.ceil(torch.log2(torch.tensor(v))).int().item())

    pad_h, pad_w = nextpow2(height), nextpow2(width)
    delta = (scalex / pad_h, scaley / pad_w)
    d = (pad_h // scalex, pad_w // scaley)
    grid = (torch.stack(torch.meshgrid(
        torch.arange(0, scalex, delta[0], device=device),
        torch.arange(0, scaley, delta[1], device=device), indexing="ij"), dim=-1) % 1)
    angles = 2 * torch.pi * torch.rand(int(scalex) + 1, int(scaley) + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(s1, s2):
        return (gradients[s1[0]:s1[1], s2[0]:s2[1]]
                .repeat_interleave(int(d[0]), 0).repeat_interleave(int(d[1]), 1))

    def dot(grad, shift):
        return (torch.stack((grid[:pad_h, :pad_w, 0] + shift[0],
                             grid[:pad_h, :pad_w, 1] + shift[1]), dim=-1) * grad[:pad_h, :pad_w]).sum(-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    t = fade(grid[:pad_h, :pad_w])
    noise = torch.sqrt(torch.tensor(2.0, device=device)) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
    return noise[:height, :width]


class PerlinAnomalyGenerator:
    """anomalib PerlinAnomalyGenerator 的等价实现(MultiRandomChoice 用 random.sample 替代)。

    beta 混合、50% 触发概率、[-90,90]° 噪声旋转、>0.5 二值化、噪声源强度 *0.5+0.25、
    3 个随机光度增广 —— 均与 anomalib 源码一致。
    """

    def __init__(self, anomaly_source_path: Path | None = None, probability: float = 0.5,
                 blend_factor=(0.1, 1.0), rotation_range=(-90, 90)):
        self.probability = probability
        self.blend_factor = blend_factor
        self.anomaly_source_paths: list[Path] = []
        if anomaly_source_path is not None and Path(anomaly_source_path).is_dir():
            for ext in IMG_EXTENSIONS:
                self.anomaly_source_paths.extend(Path(anomaly_source_path).rglob("*" + ext))
            print(f"[augmenter] 使用 {len(self.anomaly_source_paths)} 张外部纹理作合成缺陷源")
        else:
            print("[augmenter] 未找到 DTD 纹理目录, 退化为纯 Perlin 噪声源(指标略降属预期)")
        self.perlin_rotation = T.RandomAffine(
            degrees=rotation_range, interpolation=T.InterpolationMode.BILINEAR, fill=0)
        self.augmenters = [
            T.ColorJitter(contrast=(0.5, 2.0)),
            T.RandomPhotometricDistort(brightness=(0.8, 1.2), contrast=(1.0, 1.0),
                                       saturation=(1.0, 1.0), hue=(0.0, 0.0), p=1.0),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),
            T.ColorJitter(hue=[-50 / 360, 50 / 360], saturation=[0.5, 1.5]),
            T.RandomSolarize(threshold=torch.empty(1).uniform_(32 / 255, 128 / 255).item(), p=1.0),
            T.RandomPosterize(bits=4, p=1.0),
            T.RandomInvert(p=1.0),
            T.AutoAugment(),
            T.RandomEqualize(p=1.0),
            T.RandomAffine(degrees=(-45, 45), interpolation=T.InterpolationMode.BILINEAR, fill=0),
        ]

    def generate_perturbation(self, h, w, device):
        noise = generate_perlin_noise(h, w, device=device)
        if not (noise > 0.5).any():
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            noise = (noise * 2) - 1
        noise = self.perlin_rotation(noise[None]).squeeze(0)
        mask = (noise > 0.5).float().unsqueeze(-1)  # [H,W,1]
        if self.anomaly_source_paths:
            src = random.choice(self.anomaly_source_paths)
            src_img = tv_io.read_image(str(src), mode=tv_io.ImageReadMode.RGB).float().to(device) / 255.0
            if src_img.shape[-2:] != (h, w):
                src_img = v2.functional.resize(src_img, [h, w], antialias=True)
            src_img = src_img.permute(1, 2, 0)
        else:
            src_img = (noise.unsqueeze(-1).repeat(1, 1, 3) * 0.5) + 0.25
        augmented = random.sample(self.augmenters, 3)
        for aug in augmented:
            src_img = aug(src_img.permute(2, 0, 1)).permute(1, 2, 0)
        return src_img * mask, mask

    def __call__(self, images: torch.Tensor):
        """batch [B,3,H,W] -> (augmented_batch, masks [B,1,H,W])"""
        b, _, h, w = images.shape
        device = images.device
        outs, masks = [], []
        for i in range(b):
            if torch.rand(1) > self.probability:
                outs.append(images[i])
                masks.append(torch.zeros(1, h, w, device=device))
                continue
            pert, mask = self.generate_perturbation(h, w, device)  # [H,W,3], [H,W,1]
            pert = pert.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)
            lo, hi = self.blend_factor
            beta = torch.rand(1, device=device) * (hi - lo) + lo
            outs.append(images[i] * (1 - mask) + beta * pert + (1 - beta) * images[i] * mask)
            masks.append(mask)
        return torch.stack(outs), torch.stack(masks)


# --------------------------------------------------------------------------- #
# 模型（忠实移植 anomalib torch_model.py, 含原版怪癖）
# --------------------------------------------------------------------------- #
class DraemModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstructive_subnetwork = ReconstructiveSubNetwork()
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

    def forward(self, x):
        reconstruction = self.reconstructive_subnetwork(x)
        prediction = self.discriminative_subnetwork(torch.cat([x, reconstruction], dim=1))
        if self.training:
            return reconstruction, prediction
        anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return pred_score, anomaly_map


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels=3, base_width=128):
        super().__init__()
        self.block1 = _ConvBlock(in_channels, base_width)
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = _ConvBlock(base_width, base_width * 2)
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = _ConvBlock(base_width * 2, base_width * 4)
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = _ConvBlock(base_width * 4, base_width * 8)
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = _ConvBlock(base_width * 8, base_width * 8)

    def forward(self, x):
        act1 = self.block1(x); mp1 = self.mp1(act1)
        act2 = self.block2(mp1); mp2 = self.mp2(act2)
        act3 = self.block3(mp2); mp3 = self.mp3(act3)
        act4 = self.block4(mp3); mp4 = self.mp4(act4)
        return self.block5(mp4)


class DecoderReconstructive(nn.Module):
    """注意: anomalib/原版 DRAEM 的重建解码器不做 skip-concat(逐层顺序上采样)。"""

    def __init__(self, base_width=128, out_channels=3):
        super().__init__()
        w = base_width
        self.up1 = self._up(w * 8, w * 8)
        self.db1 = self._db(w * 8, w * 8, w * 4)
        self.up2 = self._up(w * 4, w * 4)
        self.db2 = self._db(w * 4, w * 4, w * 2)
        self.up3 = self._up(w * 2, w * 2)
        self.db3 = self._db(w * 2, w * 2, w * 1)
        self.up4 = self._up(w, w)
        self.db4 = self._db(w, w, w)
        self.fin_out = nn.Conv2d(w, out_channels, 3, padding=1)

    @staticmethod
    def _up(cin, cout):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                             nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    @staticmethod
    def _db(cin, cmid, cout):
        return nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
                             nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, act5):
        up1 = self.up1(act5); db1 = self.db1(up1)
        up2 = self.up2(db1); db2 = self.db2(up2)
        up3 = self.up3(db2); db3 = self.db3(up3)
        up4 = self.up4(db3); db4 = self.db4(up4)
        return self.fin_out(db4)


class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_width=128):
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width=64):
        super().__init__()
        w = base_width
        self.block1 = _ConvBlock(in_channels, w)
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = _ConvBlock(w, w * 2)
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = _ConvBlock(w * 2, w * 4)
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = _ConvBlock(w * 4, w * 8)
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = _ConvBlock(w * 8, w * 8)
        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = _ConvBlock(w * 8, w * 8)

    def forward(self, x):
        act1 = self.block1(x); mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp3(act2)  # anomalib/原版即如此(mp3 复用), 保持一致
        act3 = self.block3(mp2); mp3 = self.mp3(act3)
        act4 = self.block4(mp3); mp4 = self.mp4(act4)
        act5 = self.block5(mp4); mp5 = self.mp5(act5)
        act6 = self.block6(mp5)
        return act1, act2, act3, act4, act5, act6


class DecoderDiscriminative(nn.Module):
    """skip-concat 逐层解码(与 anomalib DecoderDiscriminative 一致)。"""

    def __init__(self, base_width=64, out_channels=2):
        super().__init__()
        w = base_width

        def up(cin, cout):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                 nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        def db(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.up_b, self.db_b = up(w * 8, w * 8), db(w * 16, w * 8)
        self.up1, self.db1 = up(w * 8, w * 4), db(w * 12, w * 4)
        self.up2, self.db2 = up(w * 4, w * 2), db(w * 6, w * 2)
        self.up3, self.db3 = up(w * 2, w), db(w * 3, w)
        self.up4, self.db4 = up(w, w), db(w * 2, w)
        self.fin_out = nn.Conv2d(w, out_channels, 3, padding=1)

    def forward(self, act1, act2, act3, act4, act5, act6):
        up_b = self.up_b(act6); db_b = self.db_b(torch.cat((up_b, act5), dim=1))
        up1 = self.up1(db_b); db1 = self.db1(torch.cat((up1, act4), dim=1))
        up2 = self.up2(db1); db2 = self.db2(torch.cat((up2, act3), dim=1))
        up3 = self.up3(db2); db3 = self.db3(torch.cat((up3, act2), dim=1))
        up4 = self.up4(db3); db4 = self.db4(torch.cat((up4, act1), dim=1))
        return self.fin_out(db4)


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, base_width=64):
        super().__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels)

    def forward(self, x):
        return self.decoder_segment(*self.encoder_segment(x))


# --------------------------------------------------------------------------- #
# 损失（anomalib DraemLoss: MSE + 2*SSIM(11) + Focal(alpha=1), kornia 实现）
# --------------------------------------------------------------------------- #
_SSIM = None   # 惰性单例, 避免每步重建 kornia 的高斯核缓存
_FOCAL = None


def draem_loss(input_image, reconstruction, anomaly_mask, prediction):
    global _SSIM, _FOCAL
    if _SSIM is None:
        from kornia.losses import FocalLoss, SSIMLoss
        _SSIM = SSIMLoss(window_size=11)
        _FOCAL = FocalLoss(alpha=1, reduction="mean")
    l2 = F.mse_loss(reconstruction, input_image)
    ssim = _SSIM(reconstruction, input_image) * 2
    focal = _FOCAL(prediction, anomaly_mask.squeeze(1).long())
    return l2 + ssim + focal


# --------------------------------------------------------------------------- #
# 评测 / 可视化（与 fastflow.py 一致的接口）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    labels, scores, maps, masks, types = [], [], [], [], []
    for img, mask, label, defect_type in loader:
        pred_score, anomaly_map = model(img.to(device))
        scores.append(pred_score.view(-1).cpu())
        maps.append(anomaly_map.cpu())
        masks.append(mask)
        labels.extend(label.tolist())
        types.extend(defect_type)
    return (np.array(labels), torch.cat(scores).numpy(), torch.cat(maps).numpy(),
            torch.cat(masks).numpy(), types)


def compute_metrics(labels, scores, maps, masks, types):
    metrics = {
        "image_auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
        "pixel_auroc": float("nan"), "per_type": {},
    }
    y = masks.reshape(-1).astype(bool)
    if y.any():
        metrics["pixel_auroc"] = float(roc_auc_score(y, maps.reshape(-1)))
    for t in sorted(set(types)):
        sel = np.array([x == t for x in types])
        if len(set(labels[sel])) > 1:
            metrics["per_type"][t] = float(roc_auc_score(labels[sel], scores[sel]))
        else:
            metrics["per_type"][t] = {"count": int(sel.sum()),
                                      "mean_score": float(np.mean(scores[sel]))}
    return metrics


def save_visualization(img, mask, amap, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amap_n = (amap - amap.min()) / max(amap.max() - amap.min(), 1e-12)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    axes[0].imshow(img.permute(1, 2, 0).numpy())
    axes[0].set_title("image")
    axes[1].imshow(mask.numpy(), cmap="gray")
    axes[1].set_title("ground truth")
    im2 = axes[2].imshow(amap_n, cmap="turbo")
    axes[2].set_title("anomaly map")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[3].imshow(img.permute(1, 2, 0).numpy())
    axes[3].imshow(amap_n, cmap="turbo", alpha=0.45)
    axes[3].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 训练 + 测试
# --------------------------------------------------------------------------- #
def train_and_test(args) -> Path:
    device = args.device
    set_seed(args.seed)
    # 注意: DRAEM 在 256 全分辨率上卷积很多, cudnn.benchmark 的穷举搜索会反复申请大
    # workspace 触发 Windows WDDM 显存回退(单步可达分钟级), 必须保持 False
    torch.backends.cudnn.benchmark = False

    out_dir = args.out_dir or Path(__file__).parent / "results" / f"{args.category}_draem" / \
        f"seed{args.seed}_s{args.image_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(exist_ok=True)

    train_set = MVTecLike(args.data_root, args.category, "train", args.image_size)
    test_set = MVTecLike(args.data_root, args.category, "test", args.image_size)
    if args.limit_train:
        train_set.samples = train_set.samples[: args.limit_train]
    if args.limit_test:
        test_set.samples = test_set.samples[: args.limit_test]
    assert len(train_set) > 0, f"{args.data_root}/{args.category} 下没有训练图片"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    print(f"[data] {args.category}: train {len(train_set)} | test {len(test_set)}")

    augmenter = PerlinAnomalyGenerator(
        anomaly_source_path=args.dtd_dir if args.dtd_dir else None,
        probability=args.augment_probability, blend_factor=(0.1, 1.0))
    augmenter.probability = args.augment_probability

    model = DraemModel().to(device)
    n_train = sum(p.numel() for p in model.parameters())
    print(f"[model] DRAEM(无预训练权重) 参数 {n_train/1e6:.2f}M")

    # anomalib lightning_model.configure_optimizers: Adam(1e-4) + MultiStepLR([400,600], 0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 600], gamma=0.1)

    best_auroc, best_epoch, patience_left = -1.0, -1, args.patience
    with open(out_dir / "epoch_log.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "lr", "val_image_auroc", "val_pixel_auroc",
                                "epoch_seconds"])

    micro = args.micro_batch or args.batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for img, _mask, _label, _type in train_loader:
            img = img.to(device)
            n = img.shape[0]
            # 先对整个 batch 合成异常(与 anomalib training_step 一致), 再微批累积
            aug_img, aug_mask = augmenter(img)
            optimizer.zero_grad(set_to_none=True)
            for s in range(0, n, micro):
                e = min(s + micro, n)
                reconstruction, prediction = model(aug_img[s:e])
                loss = draem_loss(img[s:e], reconstruction, aug_mask[s:e], prediction) * ((e - s) / n)
                loss.backward()
                losses.append(loss.item() * n / (e - s))
            optimizer.step()
        scheduler.step()

        labels, scores, maps, masks, _types = run_inference(model, test_loader, device)
        m = compute_metrics(labels, scores, maps, masks, _types)
        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0
        print(f"[epoch {epoch:4d}] loss {np.mean(losses):.4f} lr {lr_now:.1e} | val image_AUROC "
              f"{m['image_auroc']:.4f} | val pixel_AUROC {m['pixel_auroc']:.4f} | {dt:.1f}s")
        with open(out_dir / "epoch_log.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, float(np.mean(losses)), lr_now,
                                    m["image_auroc"], m["pixel_auroc"], dt])

        val_pix = m["pixel_auroc"]
        if val_pix > best_auroc:
            best_auroc, best_epoch, patience_left = val_pix, epoch, args.patience
            # 直接从显存状态流式落盘, 不做整份 CPU 克隆(小内存机器上克隆易触发
            # 提交内存耗尽); 最终测试阶段再从磁盘重载
            torch.save({"state_dict": model.state_dict(), "config": vars(args)},
                       out_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] pixel_AUROC 连续 {args.patience} 轮未提升, 于 epoch {epoch} 停止")
                break

    # best.pt 在训练中刷优时已增量保存; 从磁盘重载最优权重做最终测试
    if (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        del ckpt

    labels, scores, maps, masks, types = run_inference(model, test_loader, device)
    metrics = compute_metrics(labels, scores, maps, masks, types)
    metrics.update({"category": args.category, "image_size": args.image_size,
                    "best_epoch": best_epoch, "best_val_pixel_auroc": best_auroc,
                    "seed": args.seed, "dtd": bool(args.dtd_dir)})
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dir / "predictions.npz", labels=labels, scores=scores, maps=maps,
                        masks=masks, types=np.array(types))

    order = sorted(range(len(labels)), key=lambda i: -scores[i])
    vis_idx = list(dict.fromkeys(order[: args.num_vis] + list(range(min(args.num_vis, len(labels))))))
    for i in vis_idx:
        img, mask, _, _ = test_set[i]
        save_visualization(img, mask, maps[i],
                           out_dir / "vis" / f"{i:03d}_{types[i]}_score{scores[i]:.4f}.png")

    print(f"\n===== 最终测试结果（best epoch {best_epoch}）=====")
    print(f"image_AUROC = {metrics['image_auroc']:.4f}")
    print(f"pixel_AUROC = {metrics['pixel_auroc']:.4f}")
    print("[参考] DRAEM 原论文 MVTec 全类平均: image_AUROC 0.984 / pixel_AUROC 0.973 (使用 DTD)")
    print(f"[产物目录] {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def predict(args) -> None:
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = DraemModel()
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device).eval()

    target = Path(args.predict)
    files = sorted(p for p in (target.iterdir() if target.is_dir() else [target])
                   if p.suffix.lower() in IMG_EXTS)
    out_dir = args.out_dir or Path(args.weights).parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for p in files:
            img = torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1)
            img = F.interpolate(img[None].float(), size=(cfg["image_size"],) * 2, mode="bilinear",
                                antialias=True, align_corners=False)[0] / 255.0
            score, amap = model(img[None].to(args.device))
            amap = amap[0].cpu()
            rows.append([p.name, float(score)])
            save_visualization(img.cpu(), torch.zeros_like(amap).to(torch.uint8), amap,
                               out_dir / f"{p.stem}_pred.png")
            print(f"{p.name}: score {float(score):.4f}")
    with open(out_dir / "scores.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["file", "anomaly_score"], *rows])
    print(f"[产物目录] {out_dir}")


def download_dtd(dest: Path) -> None:
    """下载并解压官方 DTD 纹理库(~600MB, sha256 与 anomalib 一致)。"""
    import hashlib
    import tarfile
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest.parent / "dtd-r1.0.1.tar.gz"
    print(f"[dtd] 下载 {DTD_URL} -> {tar_path} (~600MB)")
    urllib.request.urlretrieve(DTD_URL, tar_path)
    h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    assert h == DTD_SHA256, f"DTD sha256 校验失败: {h}"
    print("[dtd] 解压中...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest.parent)
    print(f"[dtd] 完成: {dest}")


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "original")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--dtd-dir", type=Path, default=None,
                    help="DTD 纹理目录(官方协议的合成缺陷源); 不给则用纯 Perlin 噪声")
    ap.add_argument("--download-dtd", action="store_true", help="下载 DTD(~600MB)到 ./datasets/dtd")
    ap.add_argument("--augment-probability", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=4,
                    help="显存不足时的微批大小, 0=不分微批(DRAEM 双 U-Net 很吃显存)")
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=700)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader 进程数; 每个 worker 多占 ~1.8GB 提交内存, 小内存机器保持 0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--num-vis", type=int, default=8)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--weights", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.download_dtd:
        download_dtd(Path(__file__).parent / "datasets" / "dtd")
        return
    if args.predict:
        assert args.weights, "--predict 模式需要 --weights 指向 best.pt"
        predict(args)
    else:
        train_and_test(args)


if __name__ == "__main__":
    main()
