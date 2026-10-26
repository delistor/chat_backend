#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FastFlow 单文件复刻（对齐 anomalib v2.3.0 实现）—— 训练 / 测试 / 推理一体。

用法:
    # 训练 + 测试（默认 wide_resnet50_2, 256x256, 官方 benchmark 协议）
    python fastflow.py --category bottle

    # 自定义（骨干 / 输入尺寸 / 训练参数均可改）
    python fastflow.py --category cable --backbone resnet18 --image-size 448 --patience 20

    # 只用已训权重推理（单张图或整个文件夹）
    python fastflow.py --predict D:/some.png --weights results/bottle/seed0/best.pt
    python fastflow.py --predict D:/some_folder --weights results/bottle/seed0/best.pt

依赖: torch, timm, FrEIA(>=0.2), torchvision, numpy, scikit-learn, pillow, matplotlib(可选, 仅可视化)

对齐要点（参照 anomalib v2.3.0 src/anomalib/models/image/fastflow/，参考拷贝在
../reference/anomalib_fastflow_src/）:
  1. backbone: timm features_only=True, out_indices=[1,2,3], 完全冻结。
  2. 每层特征接一个【可训练】的 LayerNorm([C, H/s, W/s])（torch_model.py L138-145）。
  3. 每层特征独立走一个 FrEIA SequenceINN：flow_steps x AllInOneBlock,
     affine_clamping=2.0, permute_soft=False；conv3x3_only=False 时卷积核 3/1 交替
     （torch_model.py create_fast_flow_block）。
  4. coupling subnet: Conv2d(in, int(in*hidden_ratio), k, padding='same')-ReLU-
     Conv2d(hid, out, k, padding='same')。
  5. loss = Σ_scale mean(0.5*Σz² − log|det J|)（loss.py FastflowLoss）。
  6. 异常图: prob = exp(−0.5*mean(z²,dim=1))，map = interp(−prob → 输入尺寸, bilinear,
     align_corners=False)，各尺度取均值；图像分数 = map.max()（anomaly_map.py +
     torch_model.py L200-202）。
  7. 优化器 Adam(lr=1e-3, weight_decay=1e-5)，无 LR 调度、无梯度裁剪（lightning_model.py）。
  8. 官方 benchmark 协议（README + examples/configs/model/fastflow.yaml）:
     batch_size=32、seed=0、max_epochs=500、EarlyStopping(patience=3, monitor=pixel_AUROC)、
     验证集=测试集（datamodules/image/mvtecad.py: val_split_mode=SAME_AS_TEST）。
  9. 预处理: Resize((S,S)) 拉伸缩放 + ImageNet Normalize，默认 256x256
     （models/components/base/anomalib_module.py configure_pre_processor）。

注意: 训练含 CUDA 非确定性（GPU/驱动/torch 版本差异），指标在 ±0.3~0.5 个点内波动属正常，
与官方表的对照打印在训练结束后输出。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# anomalib fastflow README 官方 benchmark（seed=0, 早停 patience=3）: 类别 -> (image_AUROC, pixel_AUROC)
OFFICIAL = {
    "wide_resnet50_2": {
        "bottle": (1.000, 0.986), "cable": (0.962, 0.972), "capsule": (0.963, 0.990),
        "carpet": (0.994, 0.991), "grid": (1.000, 0.992), "hazelnut": (0.994, 0.980),
        "leather": (0.999, 0.996), "metal_nut": (0.995, 0.988), "pill": (0.942, 0.976),
        "screw": (0.839, 0.966), "tile": (1.000, 0.966), "toothbrush": (0.836, 0.980),
        "transistor": (0.979, 0.971), "wood": (0.992, 0.941), "zipper": (0.951, 0.985),
        "average": (0.963, 0.979),
    },
    "resnet18": {
        "bottle": (1.000, 0.983), "cable": (0.891, 0.954), "capsule": (0.900, 0.985),
        "carpet": (0.979, 0.983), "grid": (0.988, 0.985), "hazelnut": (0.846, 0.953),
        "leather": (1.000, 0.996), "metal_nut": (0.963, 0.972), "pill": (0.916, 0.972),
        "screw": (0.521, 0.926), "tile": (0.967, 0.944), "toothbrush": (0.844, 0.979),
        "transistor": (0.938, 0.964), "wood": (0.978, 0.956), "zipper": (0.878, 0.965),
        "average": (0.907, 0.968),
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 数据（MVTec 格式: category/{train/good, test/*, ground_truth/*}）
# --------------------------------------------------------------------------- #
class MVTecLike(Dataset):
    """anomalib MVTec 协议: train 只用 train/good；test 用全部子目录并配 ground_truth 掩码。
    预处理对齐 anomalib v2.x: 拉伸缩放到 (S,S)（双线性, antialias）+ ImageNet 归一化。"""

    def __init__(self, root: Path, category: str, split: str, image_size: int):
        self.category_dir = root / category
        self.split = split
        self.size = image_size
        self.samples: list[tuple[Path, int, Path | None, str]] = []  # (img, label, mask, type)
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
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self._mean, self._std = mean, std

    def __len__(self):
        return len(self.samples)

    def _resize(self, t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t, size=(self.size, self.size), mode="bilinear", antialias=True,
                             align_corners=False)

    def __getitem__(self, idx: int):
        path, label, mask_path, defect_type = self.samples[idx]
        img = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)
        img = self._resize(img[None])[0].float().div_(255.0)
        img = (img - self._mean) / self._std
        mask = torch.zeros(self.size, self.size, dtype=torch.uint8)
        if mask_path is not None and mask_path.exists():
            m = torch.from_numpy(np.array(Image.open(mask_path).convert("L")))[None, None]
            mask = self._resize(m)[0, 0].to(torch.uint8)
        return img, mask, label, defect_type


# --------------------------------------------------------------------------- #
# 模型（逐行对齐 anomalib torch_model.py / anomaly_map.py / loss.py）
# --------------------------------------------------------------------------- #
def subnet_conv_func(kernel_size: int, hidden_ratio: float):
    """anomalib torch_model.py L27-50: coupling 子网络构造器。"""

    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def create_fast_flow_block(input_dimensions, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    """anomalib torch_model.py L53-85: 每层特征一个独立 SequenceINN。"""
    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        kernel_size = 1 if i % 2 == 1 and not conv3x3_only else 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class AnomalyMapGenerator(nn.Module):
    """anomalib anomaly_map.py: prob=exp(-0.5*mean(z²)), map=interp(-prob), 尺度间均值。"""

    def __init__(self, input_size: tuple[int, int]):
        super().__init__()
        self.input_size = tuple(input_size)

    def forward(self, hidden_variables: list[torch.Tensor]) -> torch.Tensor:
        flow_maps = []
        for z in hidden_variables:
            log_prob = -torch.mean(z**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            flow_map = F.interpolate(-prob, size=self.input_size, mode="bilinear",
                                     align_corners=False)
            flow_maps.append(flow_map)
        return torch.mean(torch.stack(flow_maps, dim=-1), dim=-1)


class FastflowModel(nn.Module):
    """anomalib torch_model.py FastflowModel（CNN 分支: resnet18 / wide_resnet50_2）。"""

    def __init__(self, input_size, backbone="wide_resnet50_2", pre_trained=True, flow_steps=8,
                 conv3x3_only=False, hidden_ratio=1.0):
        super().__init__()
        self.input_size = tuple(input_size)
        if backbone not in {"resnet18", "wide_resnet50_2"}:
            raise ValueError(f"backbone {backbone!r} 不支持（可选: resnet18, wide_resnet50_2）")
        self.feature_extractor = timm.create_model(
            backbone, pretrained=pre_trained, features_only=True, out_indices=[1, 2, 3])
        channels = list(self.feature_extractor.feature_info.channels())
        scales = list(self.feature_extractor.feature_info.reduction())
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # 可训练 LayerNorm（注意: anomalib 中它是 trainable 的，不属于 backbone）
        self.norms = nn.ModuleList(
            nn.LayerNorm([c, int(self.input_size[0] / s), int(self.input_size[1] / s)],
                         elementwise_affine=True)
            for c, s in zip(channels, scales))
        self.fast_flow_blocks = nn.ModuleList(
            create_fast_flow_block([c, int(self.input_size[0] / s), int(self.input_size[1] / s)],
                                   conv3x3_only, hidden_ratio, flow_steps)
            for c, s in zip(channels, scales))
        self.anomaly_map_generator = AnomalyMapGenerator(self.input_size)

    def forward(self, x: torch.Tensor):
        self.feature_extractor.eval()
        features = [self.norms[i](f) for i, f in enumerate(self.feature_extractor(x))]
        hidden_variables, log_jacobians = [], []
        for block, feat in zip(self.fast_flow_blocks, features):
            z, jac = block(feat)
            hidden_variables.append(z)
            log_jacobians.append(jac)
        if self.training:
            return hidden_variables, log_jacobians
        anomaly_map = self.anomaly_map_generator(hidden_variables)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return pred_score, anomaly_map


def fastflow_loss(hidden_variables, jacobians) -> torch.Tensor:
    """anomalib loss.py FastflowLoss。"""
    loss = torch.tensor(0.0, device=hidden_variables[0].device)
    for z, jac in zip(hidden_variables, jacobians):
        loss = loss + torch.mean(0.5 * torch.sum(z**2, dim=(1, 2, 3)) - jac)
    return loss


# --------------------------------------------------------------------------- #
# 评测 / 可视化
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(model: FastflowModel, loader: DataLoader, device: str):
    """返回 (labels, scores, maps, masks, defect_types)。map/mask 均为 (N, S, S) numpy。"""
    model.eval()
    labels, scores, maps, masks, types = [], [], [], [], []
    for img, mask, label, defect_type in loader:
        pred_score, anomaly_map = model(img.to(device))
        scores.append(pred_score.view(-1).cpu())
        maps.append(anomaly_map[:, 0].cpu())
        masks.append(mask)
        labels.extend(label.tolist())
        types.extend(defect_type)
    return (np.array(labels), torch.cat(scores).numpy(), torch.cat(maps).numpy(),
            torch.cat(masks).numpy(), types)


def compute_metrics(labels, scores, maps, masks, types):
    """image/pixel AUROC + 分缺陷类型明细（与 anomalib Evaluator 的 AUROC 字段一致）。"""
    metrics = {
        "image_auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
        "pixel_auroc": float("nan"),
        "per_type": {},
    }
    y = masks.reshape(-1).astype(bool)
    if y.any():
        metrics["pixel_auroc"] = float(roc_auc_score(y, maps.reshape(-1)))
    for t in sorted(set(types)):
        sel = np.array([x == t for x in types])
        if len(set(labels[sel])) > 1:
            metrics["per_type"][t] = float(roc_auc_score(labels[sel], scores[sel]))
        else:
            # 单一标签的缺陷类型无法算 AUROC, 报告数量与平均分(类别间比较更直观)
            metrics["per_type"][t] = {"count": int(sel.sum()),
                                      "mean_score": float(np.mean(scores[sel]))}
    return metrics


def save_visualization(img, mask, amap, out_path: Path):
    """[原图 | GT掩码 | 异常热力图 | 叠加] 四联图。img/mask/amap: (S,S[,3]) 0-1 范围。"""
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
def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """只保存可训练部分（norms + flow），backbone 由 timm 缓存重建，权重文件小 ~10x。"""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("feature_extractor.")}


def train_and_test(args) -> Path:
    device = args.device
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    out_dir = args.out_dir or Path(__file__).parent / "results" / args.category / \
        f"{args.backbone}_s{args.image_size}_seed{args.seed}"
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
                              num_workers=args.num_workers, drop_last=False,
                              persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    print(f"[data] {args.category}: train {len(train_set)} | test {len(test_set)}")

    model = FastflowModel((args.image_size, args.image_size), args.backbone, pre_trained=True,
                          flow_steps=args.flow_steps, conv3x3_only=args.conv3x3_only,
                          hidden_ratio=args.hidden_ratio).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.backbone} flow_steps={args.flow_steps} "
          f"conv3x3_only={args.conv3x3_only} 可训练参数 {n_train/1e6:.2f}M")

    # anomalib lightning_model.configure_optimizers: Adam(1e-3, wd=1e-5), 无调度无裁剪
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)

    best_auroc, best_epoch, patience_left = -1.0, -1, args.patience
    log_rows = []
    with open(out_dir / "epoch_log.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_image_auroc", "val_pixel_auroc",
                                "epoch_seconds"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        micro = args.micro_batch or args.batch_size
        for img, _mask, _label, _type in train_loader:
            # 梯度累积: 显存不够一整个 batch 时用微批 + 累积, 梯度严格等价整批平均
            img = img.to(device)
            n = img.shape[0]
            starts = range(0, n, micro)
            optimizer.zero_grad(set_to_none=True)
            for s in starts:
                e = min(s + micro, n)
                z, jac = model(img[s:e])
                loss = fastflow_loss(z, jac) * ((e - s) / n)
                loss.backward()
                losses.append(loss.item() * n / (e - s))
            optimizer.step()
        # anomalib 协议: 每轮在（=测试集的）验证集上评估 pixel_AUROC 用于早停
        labels, scores, maps, masks, _types = run_inference(model, test_loader, device)
        m = compute_metrics(labels, scores, maps, masks, _types)
        dt = time.time() - t0
        print(f"[epoch {epoch:4d}] loss {np.mean(losses):.4f} | val image_AUROC "
              f"{m['image_auroc']:.4f} | val pixel_AUROC {m['pixel_auroc']:.4f} | {dt:.1f}s")
        log_rows.append([epoch, float(np.mean(losses)), m["image_auroc"], m["pixel_auroc"], dt])
        with open(out_dir / "epoch_log.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(log_rows[-1])

        val_pix = m["pixel_auroc"]
        if val_pix > best_auroc:
            best_auroc, best_epoch, patience_left = val_pix, epoch, args.patience
            # 直接从显存状态流式落盘, 不做整份 CPU 克隆(小内存机器上克隆易触发
            # 提交内存耗尽); 最终测试阶段再从磁盘重载
            torch.save({"state_dict": trainable_state_dict(model), "config": vars(args)},
                       out_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] pixel_AUROC 连续 {args.patience} 轮未提升，于 epoch {epoch} 停止")
                break

    # best.pt 在训练中刷优时已增量保存; 从磁盘重载最优权重做最终测试
    if (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt

    labels, scores, maps, masks, types = run_inference(model, test_loader, device)
    metrics = compute_metrics(labels, scores, maps, masks, types)
    metrics.update({"category": args.category, "backbone": args.backbone,
                    "image_size": args.image_size, "best_epoch": best_epoch,
                    "best_val_pixel_auroc": best_auroc, "seed": args.seed})
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dir / "predictions.npz", labels=labels, scores=scores, maps=maps,
                        masks=masks, types=np.array(types))

    # 可视化: 每种类型前几张 + 分数最高的几张
    order = sorted(range(len(labels)), key=lambda i: -scores[i])
    vis_idx = list(dict.fromkeys(order[: args.num_vis] + list(range(min(args.num_vis, len(labels))))))
    dmean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    dstd = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for i in vis_idx:
        img, mask, _, _ = test_set[i]
        img01 = (img * dstd + dmean).clamp(0, 1)
        save_visualization(img01, mask, maps[i], out_dir / "vis" / f"{i:03d}_{types[i]}_score{scores[i]:.4f}.png")

    ref = OFFICIAL.get(args.backbone, {}).get(args.category)
    print("\n===== 最终测试结果（best epoch {}）=====".format(best_epoch))
    print(f"image_AUROC = {metrics['image_auroc']:.4f}")
    print(f"pixel_AUROC = {metrics['pixel_auroc']:.4f}")
    for t, a in metrics["per_type"].items():
        if isinstance(a, dict):
            print(f"  - {t:<16s} n={a['count']:<3d} mean_score {a['mean_score']:.4f}")
        else:
            print(f"  - {t:<16s} image_AUROC {a:.4f}")
    if ref:
        print(f"\n[官方 benchmark 对照] anomalib seed=0: image_AUROC {ref[0]:.3f} / "
              f"pixel_AUROC {ref[1]:.3f}")
    print(f"[产物目录] {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def predict(args) -> None:
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = FastflowModel((cfg["image_size"], cfg["image_size"]), cfg["backbone"],
                          pre_trained=True, flow_steps=cfg["flow_steps"],
                          conv3x3_only=cfg["conv3x3_only"], hidden_ratio=cfg["hidden_ratio"])
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(args.device).eval()

    target = Path(args.predict)
    files = sorted(p for p in (target.iterdir() if target.is_dir() else [target])
                   if p.suffix.lower() in IMG_EXTS)
    out_dir = args.out_dir or Path(args.weights).parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    dmean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    dstd = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    rows = []
    with torch.no_grad():
        for p in files:
            img = torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1)
            img = F.interpolate(img[None].float(), size=(cfg["image_size"],) * 2, mode="bilinear",
                                antialias=True, align_corners=False)[0]
            img = (img / 255.0 - dmean) / dstd
            score, amap = model(img[None].to(args.device))
            amap = amap[0, 0].cpu()
            rows.append([p.name, float(score)])
            img01 = (img * dstd + dmean).clamp(0, 1)
            zero = torch.zeros_like(amap).to(torch.uint8)
            save_visualization(img01.cpu(), zero, amap, out_dir / f"{p.stem}_pred.png")
            print(f"{p.name}: score {float(score):.4f}")
    with open(out_dir / "scores.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["file", "anomaly_score"], *rows])
    print(f"[产物目录] {out_dir}")


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="bottle", help="MVTec 类别名（训练模式必填）")
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "original")
    ap.add_argument("--backbone", default="wide_resnet50_2", choices=["resnet18", "wide_resnet50_2"])
    ap.add_argument("--image-size", type=int, default=256, help="输入尺寸（anomalib 默认 256；论文 WRN-50 用 448）")
    ap.add_argument("--flow-steps", type=int, default=8)
    ap.add_argument("--conv3x3-only", type=int, default=0, help="1: 全 3x3（官方 benchmark 用 0，3/1 交替）")
    ap.add_argument("--hidden-ratio", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=12,
                    help="显存不足时每个 optimizer step 内的微批大小, 0=不分微批(=batch-size)")
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=3, help="早停耐心（官方 benchmark 用 3）")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader 进程数; 每个 worker 多占 ~1.8GB 提交内存, 小内存机器保持 0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=None, help="产物目录（默认 results/<类别>/<配置>）")
    ap.add_argument("--num-vis", type=int, default=8, help="保存可视化张数")
    ap.add_argument("--limit-train", type=int, default=0, help=">0 时只取前 N 张训练图（冒烟测试）")
    ap.add_argument("--limit-test", type=int, default=0, help=">0 时只取前 N 张测试图（冒烟测试）")
    ap.add_argument("--predict", default=None, help="推理模式: 单张图或文件夹路径")
    ap.add_argument("--weights", default=None, help="推理模式的权重路径（best.pt）")
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
