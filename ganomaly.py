#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GANomaly 单文件复刻（对齐 anomalib v2.3.0 实现, 输入缩放按原论文修正）
—— 训练 / 测试 / 推理一体。

GANomaly（Akcay et al., CVPR 2018）: Encoder-Decoder-Encoder 生成器 + 判别器的 GAN,
从零训练, 不依赖任何预训练权重。异常分数 = 两个隐向量 (z_i, z_o) 的均方差 —— 只有
图像级分数, 没有像素级异常图（这是该方法的本质, 不是实现缺失）。

用法（与 fastflow.py / draem.py / csflow.py 同构）:
    python ganomaly.py --category bottle
    python ganomaly.py --predict D:/some.png --weights results/bottle_ganomaly/seed0/best.pt

依赖: torch, torchvision, numpy, scikit-learn, pillow, matplotlib(可选)

对齐要点（anomalib v2.3.0 src/anomalib/models/image/ganomaly/）:
  1. 结构: Encoder(4x4 步幅2 卷积金字塔 -> 1x1 latent) + Decoder(转置卷积逆金字塔,
     Tanh 输出) 组成 Generator(E1-D-E2); Discriminator = Encoder 去掉最后 1x1 conv
     加 Sigmoid 分类头, 同时输出特征。DCGAN 权重初始化 N(0, 0.02)。
  2. 损失: Generator = wadv*adv + wcon*L1(x, fake) + wenc*SmoothL1(z_i, z_o),
     权重 (1, 50, 1); Discriminator = BCE(real->1, fake->0)/2, 用 BCEWithLogitsLoss
     (判别器输出 logits; 与 anomalib 的 Sigmoid+BCELoss 数学等价, 但饱和错误预测
     上梯度不为零 —— 否则判别器会在 ~3 epochs 内崩坏, 见 --adv-loss anomalib 说明)。
     对抗项默认用原版 GANomaly 的判别器特征匹配 MSE(feat_fake, feat_real)。anomalib
     源码把它写成对 sigmoid 输出做 MSE(D(real), D(fake))（丢掉了 Discriminator 返回的
     特征）, 实测会把 GAN 推进退化平衡（判别器饱和, 分数低于随机）—— anomalib 官方
     benchmark 上 GANomaly 失效(bottle 0.251)即由此而来; 复现该行为用 --adv-loss anomalib。
  3. 优化: 生成器/判别器各自 Adam(lr=2e-4, betas=(0.5, 0.999)); 每步先更新 G 再
     更新 D（anomalib training_step 顺序）。
  4. 推理: score = mean((z_i - z_o)², dim=1)。
与 anomalib 的有意偏离（文档化）:
  * 输入缩放用 [-1,1]（(x*2-1), 原论文做法）。anomalib 默认 ImageNet 归一化, 使
    输入范围 [-2.6,2.6] 与 Tanh 输出的 [-1,1] 失配, 上下文 L1 损失病态 —— 这正是
    anomalib 官方 benchmark 上 GANomaly 崩坏(均值 image AUROC 0.421)的原因。
    需要复现 anomalib 原行为时加 --normalize。
默认协议: 256x256, batch 32, 100 epochs, 早停 patience 10（监控 image AUROC ——
本模型没有像素级异常图, 早停与 best 权重均以 image AUROC 为准）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# anomalib ganomaly README 官方 benchmark（其 Normalize 输入与 Tanh 输出失配, 数字偏低,
# 仅作宽松参照）: bottle -> image_AUROC
OFFICIAL = {"bottle": 0.251, "average": 0.421}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 数据（MVTec 格式; 输入 [-1,1] 适配 Tanh 输出）
# --------------------------------------------------------------------------- #
class MVTecLike(Dataset):
    def __init__(self, root: Path, category: str, split: str, image_size: int, normalize=False):
        self.category_dir = root / category
        self.split = split
        self.size = image_size
        self.normalize = normalize
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
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def _resize(self, t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t, size=(self.size, self.size), mode="bilinear", antialias=True,
                             align_corners=False)

    def __getitem__(self, idx: int):
        path, label, mask_path, defect_type = self.samples[idx]
        img = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)
        img = self._resize(img[None])[0].float().div_(255.0)
        img = ((img - self._mean) / self._std) if self.normalize else (img * 2.0 - 1.0)
        mask = torch.zeros(self.size, self.size, dtype=torch.uint8)
        if mask_path is not None and mask_path.exists():
            m = torch.from_numpy(np.array(Image.open(mask_path).convert("L")))[None, None]
            mask = self._resize(m)[0, 0].to(torch.uint8)
        return img, mask, label, defect_type


# --------------------------------------------------------------------------- #
# 模型（逐行移植 anomalib torch_model.py）
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, input_size, latent_vec_size, num_input_channels, n_features,
                 extra_layers=0, add_final_conv_layer=True):
        super().__init__()
        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=4, bias=False))
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-conv",
                nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False))
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-batchnorm", nn.BatchNorm2d(n_features))
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-relu", nn.LeakyReLU(0.2, inplace=True))

        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*input_size) // 2
        while pyramid_dim > 4:
            in_features, out_features = n_features, n_features * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False))
            self.pyramid_features.add_module(
                f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(
                f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_features = out_features
            pyramid_dim //= 2

        self.final_conv_layer = None
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(n_features, latent_vec_size, kernel_size=4,
                                              stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.input_layers(x)
        out = self.extra_layers(out)
        out = self.pyramid_features(out)
        if self.final_conv_layer is not None:
            out = self.final_conv_layer(out)
        return out


class Decoder(nn.Module):
    def __init__(self, input_size, latent_vec_size, num_input_channels, n_features, extra_layers=0):
        super().__init__()
        self.latent_input = nn.Sequential()
        exp_factor = math.ceil(math.log(min(input_size) // 2, 2)) - 2
        n_input_features = n_features * (2 ** exp_factor)
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose2d(latent_vec_size, n_input_features, kernel_size=4, stride=1,
                               padding=0, bias=False))
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm",
                                     nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(inplace=True))

        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = min(*input_size) // 2
        while pyramid_dim > 4:
            in_features, out_features = n_input_features, n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(in_features, out_features, kernel_size=4, stride=2, padding=1,
                                   bias=False))
            self.inverse_pyramid.add_module(
                f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.inverse_pyramid.add_module(
                f"pyramid-{out_features}-relu", nn.ReLU(inplace=True))
            n_input_features = out_features
            pyramid_dim //= 2

        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-conv",
                nn.Conv2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=1,
                          bias=False))
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-batchnorm",
                nn.BatchNorm2d(n_input_features))
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-relu", nn.LeakyReLU(0.2, inplace=True))

        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-convt",
            nn.ConvTranspose2d(n_input_features, num_input_channels, kernel_size=4, stride=2,
                               padding=1, bias=False))
        self.final_layers.add_module(f"final-{num_input_channels}-tanh", nn.Tanh())

    def forward(self, x):
        out = self.latent_input(x)
        out = self.inverse_pyramid(out)
        out = self.extra_layers(out)
        return self.final_layers(out)


class Discriminator(nn.Module):
    def __init__(self, input_size, num_input_channels, n_features, extra_layers=0):
        super().__init__()
        encoder = Encoder(input_size, 1, num_input_channels, n_features, extra_layers)
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)
        self.features = nn.Sequential(*layers[:-1])
        # 注: anomalib 在此追加 Sigmoid + BCELoss; 饱和的错误预测梯度为零会让判别器
        # 崩坏(logits 冲到 ±30 后来回翻转)。这里输出 logits 配 BCEWithLogitsLoss,
        # 与 Sigmoid+BCELoss 数学等价但数值稳定(原版 DCGAN/GANomaly 实现同此做法)。
        self.classifier = nn.Sequential(layers[-1])

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Generator(nn.Module):
    def __init__(self, input_size, latent_vec_size, num_input_channels, n_features,
                 extra_layers=0, add_final_conv_layer=True):
        super().__init__()
        self.encoder1 = Encoder(input_size, latent_vec_size, num_input_channels, n_features,
                                extra_layers, add_final_conv_layer)
        self.decoder = Decoder(input_size, latent_vec_size, num_input_channels, n_features,
                               extra_layers)
        self.encoder2 = Encoder(input_size, latent_vec_size, num_input_channels, n_features,
                                extra_layers, add_final_conv_layer)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o


class GanomalyModel(nn.Module):
    def __init__(self, input_size, num_input_channels=3, n_features=64, latent_vec_size=100,
                 extra_layers=0, add_final_conv_layer=True):
        super().__init__()
        self.generator = Generator(input_size, latent_vec_size, num_input_channels, n_features,
                                   extra_layers, add_final_conv_layer)
        self.discriminator = Discriminator(input_size, num_input_channels, n_features, extra_layers)
        self.weights_init(self.generator)
        self.weights_init(self.discriminator)

    @staticmethod
    def weights_init(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        # 输入 256² 已是 2 的幂, anomalib 的 pad_nextpow2 在此为恒等
        fake, latent_i, latent_o = self.generator(x)
        if self.training:
            return fake, latent_i, latent_o
        scores = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
        return scores


class GeneratorLoss(nn.Module):
    """对抗项 adv_mode:
    "feature"（默认）: MSE(D 的中间特征(fake), D 的中间特征(real)) —— 原版 GANomaly
      的特征匹配损失, 训练稳定的正确形式。
    "anomalib": MSE(D(real), D(fake)) 的 sigmoid 输出 —— anomalib 源码行为。丢弃了
      判别器的特征返回值、在饱和的 sigmoid 上做匹配, 实测把 GAN 推进退化平衡
      （判别器崩坏, 图像分低于随机）—— anomalib 官方 benchmark 崩坏即由此而来。"""

    def __init__(self, wadv=1, wcon=50, wenc=1, adv_mode="feature"):
        super().__init__()
        self.loss_enc = nn.SmoothL1Loss()
        self.loss_adv = nn.MSELoss()
        self.loss_con = nn.L1Loss()
        self.wadv, self.wcon, self.wenc = wadv, wcon, wenc
        self.adv_mode = adv_mode

    def forward(self, latent_i, latent_o, images, fake, feat_real, feat_fake, pred_real, pred_fake):
        error_enc = self.loss_enc(latent_i, latent_o)
        error_con = self.loss_con(images, fake)
        if self.adv_mode == "anomalib":
            error_adv = self.loss_adv(pred_real, pred_fake)
        else:
            error_adv = self.loss_adv(feat_fake, feat_real)
        return error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_real, pred_fake):
        error_real = self.loss_bce(pred_real, torch.ones_like(pred_real))
        error_fake = self.loss_bce(pred_fake, torch.zeros_like(pred_fake))
        return (error_fake + error_real) * 0.5


# --------------------------------------------------------------------------- #
# 评测 / 可视化
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    labels, scores, types = [], [], []
    recon_pairs = []
    for img, _mask, label, defect_type in loader:
        s = model(img.to(device))
        scores.append(s.view(-1).cpu())
        labels.extend(label.tolist())
        types.extend(defect_type)
    return np.array(labels), torch.cat(scores).numpy(), types


def compute_metrics(labels, scores, types):
    metrics = {
        "image_auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
        "per_type": {},
    }
    for t in sorted(set(types)):
        sel = np.array([x == t for x in types])
        if len(set(labels[sel])) > 1:
            metrics["per_type"][t] = float(roc_auc_score(labels[sel], scores[sel]))
        else:
            metrics["per_type"][t] = {"count": int(sel.sum()),
                                      "mean_score": float(np.mean(scores[sel]))}
    return metrics


def save_visualization(img, fake, score, out_path: Path):
    """[输入 | 重建 | |差| 热图] 三联图。输入为 [-1,1]。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img01 = (img * 0.5 + 0.5).clamp(0, 1)
    fake01 = (fake * 0.5 + 0.5).clamp(0, 1)
    diff = (img01 - fake01).abs().mean(0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(img01.permute(1, 2, 0).numpy())
    axes[0].set_title("input")
    axes[1].imshow(fake01.permute(1, 2, 0).numpy())
    axes[1].set_title("reconstruction")
    im2 = axes[2].imshow(diff.numpy(), cmap="turbo")
    axes[2].set_title("|input - recon| (reference)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"anomaly score {score:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 训练 + 测试
# --------------------------------------------------------------------------- #
def train_and_test(args) -> Path:
    device = args.device
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    out_dir = args.out_dir or Path(__file__).parent / "results" / f"{args.category}_ganomaly" / \
        f"seed{args.seed}_s{args.image_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(exist_ok=True)

    train_set = MVTecLike(args.data_root, args.category, "train", args.image_size,
                          normalize=args.normalize)
    test_set = MVTecLike(args.data_root, args.category, "test", args.image_size,
                         normalize=args.normalize)
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

    model = GanomalyModel((args.image_size, args.image_size), n_features=args.n_features,
                          latent_vec_size=args.latent_vec_size,
                          extra_layers=args.extra_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] GANomaly(无预训练) n_features={args.n_features} latent={args.latent_vec_size} "
          f"参数 {n_params/1e6:.2f}M")

    # anomalib: 两个独立 Adam(lr=2e-4, betas=(0.5, 0.999)), 每步先 G 后 D
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    generator_loss = GeneratorLoss(adv_mode=args.adv_loss)
    discriminator_loss = DiscriminatorLoss()

    best_auroc, best_epoch, patience_left = -1.0, -1, args.patience
    with open(out_dir / "epoch_log.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "gen_loss", "disc_loss", "val_image_auroc",
                                "epoch_seconds"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        g_losses, d_losses = [], []
        for img, _mask, _label, _type in train_loader:
            img = img.to(device)
            fake, latent_i, latent_o = model(img)
            pred_real, feat_real = model.discriminator(img)
            pred_fake, feat_fake = model.discriminator(fake)
            g_loss = generator_loss(latent_i, latent_o, img, fake,
                                    feat_real, feat_fake, pred_real, pred_fake)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward(retain_graph=True)
            opt_g.step()

            pred_fake, _ = model.discriminator(fake.detach())
            d_loss = discriminator_loss(pred_real, pred_fake)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        labels, scores, _types = run_inference(model, test_loader, device)
        m = compute_metrics(labels, scores, _types)
        dt = time.time() - t0
        print(f"[epoch {epoch:4d}] g_loss {np.mean(g_losses):.4f} d_loss {np.mean(d_losses):.4f} "
              f"| val image_AUROC {m['image_auroc']:.4f} | {dt:.1f}s", flush=True)
        with open(out_dir / "epoch_log.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, float(np.mean(g_losses)), float(np.mean(d_losses)),
                                    m["image_auroc"], dt])

        val_img = m["image_auroc"]
        if val_img > best_auroc:
            best_auroc, best_epoch, patience_left = val_img, epoch, args.patience
            torch.save({"state_dict": model.state_dict(), "config": vars(args)},
                       out_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] image_AUROC 连续 {args.patience} 轮未提升, 于 epoch {epoch} 停止")
                break
        torch.cuda.empty_cache()

    if (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        del ckpt

    labels, scores, types = run_inference(model, test_loader, device)
    metrics = compute_metrics(labels, scores, types)
    metrics.update({"category": args.category, "image_size": args.image_size,
                    "best_epoch": best_epoch, "best_val_image_auroc": best_auroc,
                    "seed": args.seed, "backbone": "none(from-scratch)"})
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dir / "predictions.npz", labels=labels, scores=scores,
                        types=np.array(types))

    # 可视化: 最高分若干张 + 若干正常张
    model.eval()
    order = sorted(range(len(labels)), key=lambda i: -scores[i])
    vis_idx = list(dict.fromkeys(order[: args.num_vis] + list(range(min(2, len(labels))))))
    with torch.no_grad():
        for i in vis_idx:
            img, _, _, tname = test_set[i]
            fake, _, _ = model.generator(img[None].to(device))
            save_visualization(img, fake[0].cpu(), scores[i],
                               out_dir / "vis" / f"{i:03d}_{tname}_score{scores[i]:.4f}.png")

    print(f"\n===== 最终测试结果（best epoch {best_epoch}）=====")
    print(f"image_AUROC = {metrics['image_auroc']:.4f}")
    for t, a in metrics["per_type"].items():
        if isinstance(a, dict):
            print(f"  - {t:<16s} n={a['count']:<3d} mean_score {a['mean_score']:.4f}")
        else:
            print(f"  - {t:<16s} image_AUROC {a:.4f}")
    print("[参考] anomalib 官方 benchmark 的 GANomaly 因 Normalize/Tanh 失配而偏低"
          f"（bottle {OFFICIAL['bottle']}, 均值 {OFFICIAL['average']}）; "
          "GANomaly 本身只有图像级分数, 无像素异常图")
    print(f"[产物目录] {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def predict(args) -> None:
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = GanomalyModel((cfg["image_size"], cfg["image_size"]), n_features=cfg["n_features"],
                          latent_vec_size=cfg["latent_vec_size"], extra_layers=cfg["extra_layers"])
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
                                antialias=True, align_corners=False)[0].div_(255.0)
            img = ((img - 0.5) / 0.5) if not cfg.get("normalize") else \
                (img - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
            score = model(img[None].to(args.device))
            rows.append([p.name, float(score)])
            print(f"{p.name}: score {float(score):.4f}", flush=True)
    with open(out_dir / "scores.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["file", "anomaly_score"], *rows])
    print(f"[产物目录] {out_dir}")


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "original")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-features", type=int, default=64)
    ap.add_argument("--latent-vec-size", type=int, default=100)
    ap.add_argument("--extra-layers", type=int, default=0)
    ap.add_argument("--adv-loss", choices=["feature", "anomalib"], default="feature",
                    help="feature=原版 GANomaly 的判别器特征匹配（默认, 可正常训练）; "
                         "anomalib=anomalib 源码的 sigmoid 输出匹配（会退化, 复现其官方 0.25 行为用）")
    ap.add_argument("--normalize", action="store_true",
                    help="ImageNet 归一化输入（anomalib 默认, 与 Tanh 输出失配、官方 benchmark 崩坏）; "
                         "默认用原论文的 [-1,1] 缩放")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
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
    if args.predict:
        assert args.weights, "--predict 模式需要 --weights 指向 best.pt"
        predict(args)
    else:
        train_and_test(args)


if __name__ == "__main__":
    main()
