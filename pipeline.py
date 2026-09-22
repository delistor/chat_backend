"""adc_pipeline.py — 32x32 三通道 (raw/compare/diff) patch ADC 单文件流水线 (自包含).

一个文件覆盖全部阶段 (模型/数据/损失/评估全部内联, 不依赖外部包):
    synth     生成合成数据 (真实数据可跳过)
    pretrain  MAE 自监督预训练 (所有 layer 混训, 学共性; 可选吃 unlabeled.npy 无标注池)
              结束时跑 linear probe (门控二分) 判断预训练值不值
    train     多任务微调: 共享 backbone + noise/real 门控头 + per-layer bin 头,
              可用 --init-from 加载 MAE 编码器或已有监督 ckpt
    heads     冻结 backbone, 只训门控/bin 头 —— 新 layer 上线/接新数据用
    baseline  手工特征 + GBDT 门控基线 (深度模型的 sanity check, 自动打印对照)
    eval      加载 ckpt 复评 (按 wafer 组划分, 输出 eval_report.json)
    infer     批量推理: 全量 patch -> predictions.csv (三段路由 + bin 预测, review 即人审队列)
    all       synth(数据缺失时) -> pretrain -> train -> eval -> baseline -> infer

内联模型: ViT-tiny(可 MAE 预训练) / ResNet18-CIFAR / TinyCNN + MAE + 多任务头。

两种运行方式:
    1) VSCode: 直接改下方 CONFIG, 按 F5 / Run (无命令行参数时使用 CONFIG 全量值)
    2) 命令行: python adc_pipeline.py --stage all --mae-epochs 3 ...
       命令行只覆盖显式传入的参数, 其余用 CONFIG

数据目录格式 (同 adc/ 包):
    labels.csv  列: patch_id, layer, wafer, is_real, bin, [split]
    patches.npy float16/32, (N,3,32,32), 通道 [raw, compare, diff], 行与 csv 对齐
    unlabeled.npy (可选) (M,3,32,32) MAE 无标注池
    或 raw/{id}.png compare/{id}.png diff/{id}.png 三目录
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

BASE = Path(__file__).resolve().parent

# ============================================================================
# VSCode 直接改这里; 命令行传参会覆盖同名项 (下划线对应 --ke-y 形式)
# ============================================================================
CONFIG = dict(
    stage="all",                 # synth | pretrain | train | heads | eval | all
    data_dir="data/pipe",        # 相对路径基于本文件所在目录
    out_dir="runs/pipe",
    # ---- synth (仅合成阶段用) ----
    per_layer=1500, wafers=30, real_ratio=0.35, synth_seed=7,
    # ---- 模型 ----
    arch="vit_tiny",             # vit_tiny (可接 MAE) | resnet18 | tiny_cnn
    channels="rcd",              # rcd | rd | d
    vit_dim=192, vit_depth=6, vit_heads=3, patch_size=4,
    # ---- pretrain (MAE) ----
    mae_epochs=20, mae_lr=3e-4, mae_wd=0.05, mae_batch=256,
    mask_ratio=0.55, dec_dim=96, dec_depth=2,
    # ---- train / heads ----
    epochs=15, batch_size=256, lr=1e-3, wd=5e-4,
    lam_bin=1.0,                 # bin loss 权重
    aug="geom",                  # geom | none (禁止亮度/对比度类: 会抹掉缺陷强度信号)
    init_from="",                # MAE 编码器或监督 ckpt 路径; 空=从头训
    freeze="none",               # none | backbone (冻结 backbone 只训头)
    layers="",                   # 逗号分隔, 空=全部层 (heads 接新层时指定)
    target_recall=0.98,          # 门控阈值按该 recall 取 (漏检代价最高)
    review_band=0.10,            # 人审不确定带宽度
    # ---- eval ----
    ckpt="",                     # eval 用; 空 = out_dir/ckpt.pt
    # ---- 通用 ----
    seed=0, num_workers=0, force_synth=False,
)

CHANNEL_SETS = {"rcd": (0, 1, 2), "rd": (0, 2), "d": (2,)}
SPLIT_RATIOS = (0.70, 0.15, 0.15)


# ---------------------------------------------------------------- utilities
def log(msg: str):
    print(msg, flush=True)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rpath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else BASE / p


def channel_stats(patches, rows, split="train"):
    sel = [i for i, r in enumerate(rows) if r["split"] == split]
    sub = patches[sel]
    mean = sub.mean(axis=(0, 2, 3))
    std = sub.std(axis=(0, 2, 3)) + 1e-6
    return (torch.tensor(mean, dtype=torch.float32)[:, None, None],
            torch.tensor(std, dtype=torch.float32)[:, None, None])


# ---------------------------------------------------------------- data
def read_labels(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"patch_id", "layer", "wafer", "is_real", "bin"}
        if not need.issubset(reader.fieldnames or []):
            raise ValueError(f"labels.csv 缺列, 需要 {need}, 实际 {reader.fieldnames}")
        for r in reader:
            rows.append(dict(patch_id=r["patch_id"].strip(), layer=r["layer"].strip(),
                             wafer=r["wafer"].strip(), is_real=int(r["is_real"]),
                             bin=(r["bin"] or "").strip(),
                             split=(r.get("split") or "").strip() or None))
    if not rows:
        raise ValueError(f"{csv_path} 无数据行")
    return rows


def load_patches(data_dir: Path, rows: list[dict]) -> np.ndarray:
    npy = data_dir / "patches.npy"
    if npy.exists():
        arr = np.load(npy)
        if arr.ndim != 4 or arr.shape[0] != len(rows) or arr.shape[1] != 3:
            raise ValueError(f"patches.npy {arr.shape} 与 labels.csv ({len(rows)}x3) 不符")
        return arr.astype(np.float32)
    if (data_dir / "raw").exists():
        from PIL import Image
        out = np.zeros((len(rows), 3, 32, 32), np.float32)
        for i, r in enumerate(rows):
            for c, name in enumerate(("raw", "compare", "diff")):
                out[i, c] = np.asarray(Image.open(data_dir / name / f"{r['patch_id']}.png"),
                                       dtype=np.float32) / 255.0
        return out
    raise FileNotFoundError(f"{data_dir} 下无 patches.npy 也无 raw/ 目录")


def assign_splits(rows: list[dict], seed: int = 0) -> None:
    """按 wafer 整组划分, 同一 wafer 只落在一个 split (防相邻 patch 泄漏)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if not r["split"]:
            groups[r["wafer"]].append(i)
    wafers = sorted(groups)
    rng.shuffle(wafers)
    n_tr = int(round(len(wafers) * SPLIT_RATIOS[0]))
    n_va = int(round(len(wafers) * SPLIT_RATIOS[1]))
    for j, w in enumerate(wafers):
        s = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
        for i in groups[w]:
            rows[i]["split"] = s


def build_vocab(rows: list[dict], layers_filter: list[str] | None = None):
    layers = sorted({r["layer"] for r in rows if r["split"] == "train"})
    if layers_filter:
        keep = set(layers_filter)
        layers = [L for L in layers if L in keep]
        if not layers:
            raise ValueError(f"--layers {layers_filter} 在 train split 中不存在; 可用: {sorted({r['layer'] for r in rows})}")
    vocab = {L: sorted({r["bin"] for r in rows
                        if r["split"] == "train" and r["layer"] == L
                        and r["is_real"] == 1 and r["bin"]})
             for L in layers}
    return layers, vocab


def geom_aug(img: torch.Tensor) -> torch.Tensor:
    k = int(torch.randint(4, (1,)))
    if k:
        img = torch.rot90(img, k, dims=(1, 2))
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=(1,))
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=(2,))
    return img.contiguous()


class PatchDataset(Dataset):
    def __init__(self, patches, rows, split, layers, bin_vocab, mean, std,
                 channels=(0, 1, 2), aug=None):
        self.patches, self.rows = patches, rows
        self.layer_idx = {L: i for i, L in enumerate(layers)}
        self.bin_index = {L: {b: i for i, b in enumerate(bs)} for L, bs in bin_vocab.items()}
        self.mean, self.std = mean[list(channels)], std[list(channels)]
        self.channels, self.aug = list(channels), aug
        self.idx = [i for i, r in enumerate(rows)
                    if (split is None or r["split"] == split) and r["layer"] in self.layer_idx]
        if not self.idx:
            raise ValueError(f"split={split} 无样本 (layers={layers})")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        r = self.rows[i]
        img = torch.from_numpy(self.patches[i])[self.channels]
        if self.aug is not None:
            img = self.aug(img)
        img = (img - self.mean) / self.std
        b = -1
        if r["is_real"] == 1:
            b = self.bin_index[r["layer"]].get(r["bin"], -1)   # -1: train 未见的 bin
        return img, r["is_real"], b, self.layer_idx[r["layer"]]


class ArrayDataset(Dataset):
    """MAE 用: 只要图, 不要标签."""

    def __init__(self, patches, mean, std, channels=(0, 1, 2), aug=None):
        self.patches = patches
        self.mean, self.std = mean[list(channels)], std[list(channels)]
        self.channels, self.aug = list(channels), aug

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        img = torch.from_numpy(self.patches[i])[self.channels]
        if self.aug is not None:
            img = self.aug(img)
        return (img - self.mean) / self.std


def make_classification_loaders(data_dir, channels, batch_size, aug_geom, seed,
                                layers_filter=None, num_workers=0, layers=None, bin_vocab=None):
    """layers/bin_vocab 传入时用外部词表 (eval 重放 ckpt 词表); 否则从 train 重建."""
    data_dir = rpath(data_dir)
    rows = read_labels(data_dir / "labels.csv")
    if any(not r["split"] for r in rows):
        assign_splits(rows, seed=seed)
    patches = load_patches(data_dir, rows)
    if layers is None:
        layers, bin_vocab = build_vocab(rows, layers_filter)
    mean, std = channel_stats(patches, rows, "train")
    aug = geom_aug if aug_geom else None
    mk = lambda split, sh: DataLoader(
        PatchDataset(patches, rows, split, layers, bin_vocab, mean, std, channels,
                     aug if sh else None),
        batch_size=batch_size, shuffle=sh, num_workers=num_workers)
    return dict(train=mk("train", True), val=mk("val", False), test=mk("test", False),
                rows=rows, layers=layers, bin_vocab=bin_vocab, mean=mean, std=std)


# ---------------------------------------------------------------- synthetic
LAYER_CFG = {
    "M1": dict(bground=dict(px=8, py=8, ax=0.35, ay=0.15),
               bins={"bridge": ("bar", True), "open": ("hole", True), "particle": ("blob", False)}),
    "M2": dict(bground=dict(px=11, py=5, ax=0.30, ay=0.25),
               bins={"scratch": ("line", False), "residue": ("cluster", False), "particle": ("blob", False)}),
    "V1": dict(bground=dict(px=6, py=13, ax=0.20, ay=0.30),
               bins={"pit": ("hole", False), "particle": ("blob", False), "bridge": ("bar", True)}),
}
NOISE_SPECKLE_AMP = (0.25, 0.45)
DEFECT_AMP = (0.55, 1.10)
BG_SIGMA = 0.10
_YY, _XX = np.mgrid[0:32, 0:32].astype(np.float32)


def _seg_mask(x0, y0, x1, y1, width):
    d = np.stack([_XX - x0, _YY - y0], -1)
    v = np.array([x1 - x0, y1 - y0], np.float32)
    t = np.clip((d @ v) / max(float(v @ v), 1e-6), 0, 1)
    return np.linalg.norm(d - t[..., None] * v, axis=-1) <= width


def _blob(rng, sign=1.0):
    cx, cy = rng.uniform(9, 23, 2)
    r = rng.uniform(1.5, 3.2)
    return (rng.uniform(*DEFECT_AMP) * sign
            * np.exp(-((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2 * r * r))).astype(np.float32)


def _bar(rng):
    if rng.random() < 0.5:
        x0 = rng.uniform(5, 12); x1 = min(x0 + rng.uniform(12, 18), 27)
        y = rng.uniform(10, 22)
        m = _seg_mask(x0, y, x1, y, rng.uniform(1.0, 2.0))
    else:
        y0 = rng.uniform(5, 12); y1 = min(y0 + rng.uniform(12, 18), 27)
        x = rng.uniform(10, 22)
        m = _seg_mask(x, y0, x, y1, rng.uniform(1.0, 2.0))
    return (m * rng.uniform(*DEFECT_AMP)).astype(np.float32)


def _line(rng):
    th = rng.uniform(0, np.pi)
    cx, cy = rng.uniform(13, 19, 2)
    ln = rng.uniform(10, 15)
    m = _seg_mask(cx - ln / 2 * np.cos(th), cy - ln / 2 * np.sin(th),
                  cx + ln / 2 * np.cos(th), cy + ln / 2 * np.sin(th), 1.0)
    return (m * rng.uniform(*DEFECT_AMP)).astype(np.float32)


def _cluster(rng):
    out = np.zeros((32, 32), np.float32)
    cx, cy = rng.uniform(12, 20, 2)
    for _ in range(int(rng.integers(5, 10))):
        dx, dy = rng.normal(0, 2.5, 2)
        r = rng.uniform(0.8, 1.6)
        out += rng.uniform(0.3, 0.7) * np.exp(
            -((_XX - (cx + dx)) ** 2 + (_YY - (cy + dy)) ** 2) / (2 * r * r))
    return np.clip(out, 0, 1.2).astype(np.float32)


_RENDER = dict(blob=lambda r: _blob(r), hole=lambda r: _blob(r, -1.0),
               bar=_bar, line=_line, cluster=_cluster)


def _background(cfg, rng):
    b = cfg["bground"]
    phx, phy = rng.uniform(0, 2 * np.pi, 2)
    g = b["ax"] * np.sin(2 * np.pi * _XX / b["px"] + phx) \
        + b["ay"] * np.sin(2 * np.pi * _YY / b["py"] + phy)
    low = np.sin(2 * np.pi * _XX / 31 + rng.uniform(0, 6)) * np.sin(2 * np.pi * _YY / 29 + rng.uniform(0, 6))
    return (g + 0.08 * low + rng.normal(0, BG_SIGMA, (32, 32))).astype(np.float32)


def _speckles(rng):
    out = np.zeros((32, 32), np.float32)
    for _ in range(int(rng.integers(3, 9))):
        cx, cy = rng.uniform(4, 28, 2)
        r = rng.uniform(0.7, 1.3)
        out += rng.uniform(*NOISE_SPECKLE_AMP) * np.exp(
            -((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2 * r * r))
    return out


def stage_synth(cfg):
    out = rpath(cfg["data_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows, patches = [], []
    for li, (layer, lcfg) in enumerate(LAYER_CFG.items()):
        rng = np.random.default_rng(cfg["synth_seed"] + 101 * li)
        bin_names = list(lcfg["bins"])
        per_wafer = cfg["per_layer"] // cfg["wafers"]
        for w in range(cfg["wafers"]):
            prop = rng.dirichlet(np.full(len(bin_names), 0.7))  # wafer 级 bin 倾向
            for k in range(per_wafer):
                raw = _background(lcfg, rng)
                compare = _background(lcfg, rng)   # 同 pattern 另一 die
                if rng.random() < cfg["real_ratio"]:
                    bname = str(rng.choice(bin_names, p=prop))
                    morph, repeating = lcfg["bins"][bname]
                    defect = _RENDER[morph](rng)
                    raw = raw + defect
                    if repeating and rng.random() < 0.7:       # repeating defect 弱重现
                        compare = compare + defect * rng.uniform(0.35, 0.6)
                    is_real, bcol = 1, bname
                else:
                    raw = raw + _speckles(rng)
                    is_real, bcol = 0, ""
                rows.append(dict(patch_id=f"{layer}_W{w:03d}_{k:04d}", layer=layer,
                                 wafer=f"{layer}_W{w:03d}", is_real=is_real, bin=bcol, split=""))
                patches.append(np.stack([raw, compare, raw - compare])[None])
    np.save(out / "patches.npy", np.concatenate(patches).astype(np.float16))
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patch_id", "layer", "wafer", "is_real", "bin", "split"])
        w.writeheader()
        w.writerows(rows)
    log(f"[synth] {len(rows)} patches -> {out}")


# ---------------------------------------------------------------- backbones
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + (self.short(x) if self.short is not None else x))


class ResNet18CIFAR(nn.Module):
    def __init__(self, in_ch=3, width=64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(in_ch, w, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(BasicBlock(w, w, 1), BasicBlock(w, w, 1))
        self.stage2 = nn.Sequential(BasicBlock(w, 2 * w, 2), BasicBlock(2 * w, 2 * w, 1))
        self.stage3 = nn.Sequential(BasicBlock(2 * w, 4 * w, 2), BasicBlock(4 * w, 4 * w, 1))
        self.stage4 = nn.Sequential(BasicBlock(4 * w, 8 * w, 2), BasicBlock(8 * w, 8 * w, 1))
        self.out_dim = 8 * w

    def forward(self, x):
        x = self.stem(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)


class TinyCNN(nn.Module):
    def __init__(self, in_ch=3, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 1, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, 2, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, 2 * width, 3, 1, 1), nn.BatchNorm2d(2 * width), nn.ReLU(inplace=True),
            nn.Conv2d(2 * width, 2 * width, 3, 2, 1), nn.BatchNorm2d(2 * width), nn.ReLU(inplace=True),
            nn.Conv2d(2 * width, 4 * width, 3, 1, 1), nn.BatchNorm2d(4 * width), nn.ReLU(inplace=True))
        self.out_dim = 4 * width

    def forward(self, x):
        return torch.flatten(F.adaptive_avg_pool2d(self.net(x), 1), 1)


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        h = int(dim * mlp_ratio)
        self.fc1, self.fc2 = nn.Linear(dim, h), nn.Linear(h, dim)

    def forward(self, x):  # x: (B,N,D)
        B, N, D = x.shape
        h = self.n1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                          # 各 (B,H,N,Dh)
        attn = F.scaled_dot_product_attention(q, k, v)   # (B,H,N,Dh)
        attn = attn.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(attn)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class ViT(nn.Module):
    """32x32 用 4x4 patch (64 token); 分类取 token 均值, MAE 复用其子模块."""

    def __init__(self, in_ch=3, dim=192, depth=6, heads=3, patch=4, img=32):
        super().__init__()
        self.patch, self.img, self.in_ch = patch, img, in_ch
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, patch)
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def embed(self, x):  # (B,N,D) 已加 pos, 未过 blocks
        t = self.patch_embed(x).flatten(2).transpose(1, 2)
        return t + self.pos

    def forward(self, x):  # 分类特征
        t = self.embed(x)
        for b in self.blocks:
            t = b(t)
        return self.norm(t).mean(1)


def build_backbone(arch, in_ch, cfg):
    if arch == "vit_tiny":
        return ViT(in_ch, cfg["vit_dim"], cfg["vit_depth"], cfg["vit_heads"], cfg["patch_size"])
    if arch == "resnet18":
        return ResNet18CIFAR(in_ch)
    if arch == "tiny_cnn":
        return TinyCNN(in_ch)
    raise ValueError(f"未知 arch: {arch}")


# ---------------------------------------------------------------- MAE
class MAE(nn.Module):
    """随机遮 4x4 patch, 编码可见 token, 解码器重建被遮 patch; loss 只算被遮位置."""

    def __init__(self, encoder: ViT, mask_ratio=0.55, dec_dim=96, dec_depth=2, dec_heads=3):
        super().__init__()
        self.encoder, self.mask_ratio = encoder, mask_ratio
        n = encoder.pos.shape[1]
        dim = encoder.out_dim
        self.dec_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.dec_pos = nn.Parameter(torch.zeros(1, n, dec_dim))
        nn.init.normal_(self.dec_pos, std=0.02)
        self.dec_blocks = nn.ModuleList([ViTBlock(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, encoder.patch ** 2 * encoder.in_ch)

    def patchify(self, x):  # (B,C,H,W) -> (B,N,C*p*p)
        p = self.encoder.patch
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B,C,H/p,W/p,p,p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(x.size(0), -1, self.encoder.in_ch * p * p)
        return x

    def forward(self, x):
        B, N, _ = self.patchify(x).shape
        k = max(1, int(N * (1 - self.mask_ratio)))
        tokens = self.encoder.embed(x)                   # (B,N,D)
        ids = torch.randperm(N, device=x.device)
        keep, mask_ids = ids[:k], ids[k:]
        t = tokens[:, keep]
        for b in self.encoder.blocks:
            t = b(t)
        t = self.encoder.norm(t)
        full = torch.empty(B, N, self.dec_embed.out_features, device=x.device,
                           dtype=t.dtype)
        full[:, keep] = self.dec_embed(t)
        full[:, mask_ids] = self.mask_token.expand(B, -1, -1)
        full = full + self.dec_pos
        unshuf = torch.empty_like(full)
        unshuf[:, ids] = full                             # 恢复空间顺序
        for b in self.dec_blocks:
            unshuf = b(unshuf)
        pred = self.pred(self.dec_norm(unshuf))
        target = self.patchify(x)
        loss = F.mse_loss(pred[:, mask_ids], target[:, mask_ids])
        return loss, pred, target, mask_ids


def load_pretrain_images(cfg):
    """MAE 数据 = train split 的图 (标签不用) + 可选 unlabeled.npy 无标注池."""
    data_dir = rpath(cfg["data_dir"])
    rows = read_labels(data_dir / "labels.csv")
    if any(not r["split"] for r in rows):
        assign_splits(rows, seed=cfg["seed"])
    patches = load_patches(data_dir, rows)
    sel = [i for i, r in enumerate(rows) if r["split"] == "train"]
    imgs = [patches[sel]]
    unl = data_dir / "unlabeled.npy"
    if unl.exists():
        extra = np.load(unl).astype(np.float32)
        imgs.append(extra)
        log(f"[pretrain] + unlabeled.npy {extra.shape[0]} 张")
    mean, std = channel_stats(patches, rows, "train")
    return np.concatenate(imgs), mean, std, rows


@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    feats, ys = [], []
    for img, y, _, _ in loader:
        feats.append(backbone(img.to(device)).cpu())
        ys.append(y)
    return torch.cat(feats).numpy(), torch.cat(ys).numpy()


def linear_probe(backbone, loaders, device):
    """冻结特征上训逻辑回归, 报告门控 val AUC —— 判断预训练值不值的快速信号."""
    xt, yt = extract_features(backbone, loaders["train"], device)
    xv, yv = extract_features(backbone, loaders["val"], device)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(xt, yt)
    return float(roc_auc_score(yv, clf.predict_proba(xv)[:, 1]))


def stage_pretrain(cfg):
    if cfg["arch"] != "vit_tiny":
        raise ValueError("MAE 预训练目前只支持 arch=vit_tiny (resnet18 走监督直训)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch = CHANNEL_SETS[cfg["channels"]]
    imgs, mean, std, rows = load_pretrain_images(cfg)
    ds = ArrayDataset(imgs, mean, std, ch, aug=geom_aug)   # MAE 对增强不敏感, 几何可用
    loader = DataLoader(ds, batch_size=cfg["mae_batch"], shuffle=True,
                        num_workers=cfg["num_workers"])
    backbone = ViT(len(ch), cfg["vit_dim"], cfg["vit_depth"], cfg["vit_heads"],
                   cfg["patch_size"]).to(device)
    mae = MAE(backbone, cfg["mask_ratio"], cfg["dec_dim"], cfg["dec_depth"]).to(device)
    opt = torch.optim.AdamW(mae.parameters(), lr=cfg["mae_lr"], weight_decay=cfg["mae_wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["mae_epochs"])
    log(f"[pretrain] {len(ds)} imgs, mask={cfg['mask_ratio']}, "
        f"device={device}, {cfg['mae_epochs']} epochs")
    for ep in range(1, cfg["mae_epochs"] + 1):
        mae.train()
        tot = n = 0
        for img in loader:
            img = img.to(device)
            loss, _, _, _ = mae(img)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * img.size(0)
            n += img.size(0)
        log(f"[pretrain] epoch {ep:3d} | masked mse {tot / n:.5f}")
        sched.step()
    out = rpath(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(encoder=backbone.state_dict(), config=dict(cfg)), out / "mae_encoder.pt")
    # linear probe: 冻结编码器上门控二分的可分性
    layers_filter = [s.strip() for s in cfg["layers"].split(",") if s.strip()] or None
    loaders = make_classification_loaders(cfg["data_dir"], ch, cfg["batch_size"], False,
                                          cfg["seed"], layers_filter, cfg["num_workers"])
    backbone.eval()
    auc = linear_probe(backbone, loaders, device)
    log(f"[pretrain] linear-probe gate val AUC = {auc:.4f} "
        f"(对比从头训的 probe, 高则预训练值得用)")
    (out / "pretrain_report.json").write_text(
        json.dumps(dict(last_mse=tot / n, probe_gate_val_auc=auc), indent=2), encoding="utf-8")
    return out / "mae_encoder.pt"


# ---------------------------------------------------------------- classifier
class ADCModel(nn.Module):
    def __init__(self, backbone: nn.Module, bin_vocab: dict[str, list[str]]):
        super().__init__()
        self.backbone = backbone
        self.layer_names = sorted(bin_vocab)
        self.gate = nn.Linear(backbone.out_dim, 2)
        self.bin_heads = nn.ModuleDict({
            L: nn.Linear(backbone.out_dim, len(bs)) for L, bs in bin_vocab.items() if bs})
        self._n_max = max((len(b) for b in bin_vocab.values()), default=1)

    def forward(self, x, layer_idx):
        feat = self.backbone(x)
        gate_logits = self.gate(feat)
        # 不足 n_max 的头用 -inf 填充, 防止 argmax/softmax 误选 padding 列
        bin_logits = torch.full((x.size(0), self._n_max), float("-inf"),
                                device=x.device, dtype=feat.dtype)
        for li in layer_idx.unique():
            name = self.layer_names[int(li)]
            if name not in self.bin_heads:
                continue
            m = layer_idx == li
            out = self.bin_heads[name](feat[m])
            bin_logits[m, :out.shape[1]] = out    # 只写该头有效列, 其余保持 -inf
        return gate_logits, bin_logits


def load_backbone_weights(model: ADCModel, path: str, log_fn=log):
    """兼容三种 ckpt: MAE {encoder: sd} / 监督 {model: sd(backbone.前缀)} / 裸 sd."""
    ckpt = torch.load(rpath(path), map_location="cpu", weights_only=False)
    sd = ckpt.get("encoder") if isinstance(ckpt, dict) and "encoder" in ckpt else None
    if sd is None and isinstance(ckpt, dict) and "model" in ckpt:
        sd = {k[len("backbone."):]: v for k, v in ckpt["model"].items()
              if k.startswith("backbone.")}
    if sd is None and isinstance(ckpt, dict):
        sd = {k: v for k, v in ckpt.items()
              if isinstance(v, torch.Tensor) and not k.startswith(("gate", "bin_heads"))}
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    log_fn(f"[init] backbone <- {path} (missing={len(missing)}, unexpected={len(unexpected)})")


def bin_class_weights(rows, bin_vocab):
    counts = {L: {b: 0 for b in bs} for L, bs in bin_vocab.items()}
    for r in rows:
        if r["split"] == "train" and r["is_real"] == 1:
            d = counts.get(r["layer"])
            if d is not None and r["bin"] in d:
                d[r["bin"]] += 1
    weights = {}
    for L, d in counts.items():
        if d:
            total, n = sum(d.values()), len(d)
            weights[L] = torch.tensor([total / (n * max(c, 1)) for c in d.values()])
    return weights


def run_epoch(model, loader, device, opt=None, lam_bin=1.0, weights=None,
              scheduler=None, freeze_backbone=False):
    training = opt is not None
    model.train(training)
    if training and freeze_backbone:
        model.backbone.eval()   # 冻结时 BN 统计也不再更新
    tot = n = 0
    for img, gate_y, bin_y, layer_i in loader:
        img, gate_y = img.to(device), gate_y.to(device)
        bin_y, layer_i = bin_y.to(device), layer_i.to(device)
        gate_logits, bin_logits = model(img, layer_i)
        loss = F.cross_entropy(gate_logits, gate_y)
        for li in layer_i.unique():
            name = model.layer_names[int(li)]
            m_bin = (layer_i == li) & (gate_y == 1) & (bin_y >= 0)
            if m_bin.any() and name in weights:
                loss = loss + lam_bin * F.cross_entropy(
                    bin_logits[m_bin], bin_y[m_bin], weight=weights[name].to(device))
        if training:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        tot += loss.item() * img.size(0)
        n += img.size(0)
    if training and scheduler is not None:
        scheduler.step()
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, G, BY, BP, LI = [], [], [], [], []
    for img, gate_y, bin_y, layer_i in loader:
        gate_logits, bin_logits = model(img.to(device), layer_i.to(device))
        P.append(torch.softmax(gate_logits.float(), 1)[:, 1].cpu())
        G.append(gate_y)
        BY.append(bin_y)
        has_head = torch.tensor([1 if model.layer_names[int(l)] in model.bin_heads else 0
                                 for l in layer_i])
        BP.append(torch.where(has_head.bool(), bin_logits.argmax(1), torch.tensor(-2)))
        LI.append(layer_i)
    return dict(p_real=torch.cat(P).numpy(), y_gate=torch.cat(G).numpy(),
                y_bin=torch.cat(BY).numpy(), p_bin=torch.cat(BP).numpy(),
                layer=torch.cat(LI).numpy())


def pick_threshold(p, y, target_recall):
    pr = p[y == 1]
    return float(np.quantile(pr, 1 - target_recall)) if len(pr) else 0.5


def gate_report(p, y, tau, band):
    auto_real = p >= tau
    review = (p >= tau - band) & ~auto_real
    real = y == 1
    return dict(threshold=tau, review_band=band,
                gate_auroc=float(roc_auc_score(y, p)),
                gate_acc=float((auto_real == real).mean()),
                escape_rate=float((real & ~auto_real & ~review).mean()),
                real_in_review_rate=float((real & review).sum() / max(real.sum(), 1)),
                review_rate=float(review.mean()),
                noise_pass_rate=float((~real & auto_real).sum() / max((~real).sum(), 1)))


def bin_report(e, layers, bin_vocab, gate="oracle", tau=None):
    out, fs = {"per_layer": {}}, []
    for li, name in enumerate(layers):
        bins = bin_vocab[name]
        if not bins:
            continue
        m = (e["layer"] == li) & (e["y_bin"] >= 0)
        if gate == "cascade":
            m &= e["p_real"] >= tau
        yt, yp = e["y_bin"][m], e["p_bin"][m]
        if len(yt) == 0:
            out["per_layer"][name] = {"n": 0}
            continue
        f1m = float(f1_score(yt, yp, labels=list(range(len(bins))), average="macro",
                             zero_division=0))
        out["per_layer"][name] = dict(n=int(len(yt)), macro_f1=f1m, acc=float((yt == yp).mean()),
                                      classes=bins,
                                      confusion=confusion_matrix(
                                          yt, yp, labels=list(range(len(bins)))).tolist())
        fs.append(f1m)
    out["macro_f1_mean"] = float(np.mean(fs)) if fs else None
    out["unseen_bin_samples"] = int((e["y_bin"] == -1).sum())
    out["no_head_samples"] = int((e["p_bin"] == -2).sum())
    return out


def stage_train(cfg, freeze_override=None, init_from=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch = CHANNEL_SETS[cfg["channels"]]
    layers_filter = [s.strip() for s in cfg["layers"].split(",") if s.strip()] or None
    loaders = make_classification_loaders(cfg["data_dir"], ch, cfg["batch_size"],
                                          cfg["aug"] == "geom", cfg["seed"], layers_filter,
                                          cfg["num_workers"])
    rows, layers, vocab = loaders["rows"], loaders["layers"], loaders["bin_vocab"]
    log(f"[train] splits={ {k: sum(1 for r in rows if r['split']==k) for k in ('train','val','test')} } "
        f"layers={layers} bins={vocab} device={device}")
    model = ADCModel(build_backbone(cfg["arch"], len(ch), cfg), vocab).to(device)
    src = init_from if init_from is not None else cfg["init_from"]
    if src:
        load_backbone_weights(model, src)
    freeze = freeze_override if freeze_override is not None else cfg["freeze"]
    if freeze == "backbone":
        for p_ in model.backbone.parameters():
            p_.requires_grad = False
        log("[train] backbone frozen (只训 gate + bin heads)")
    weights = bin_class_weights(rows, vocab)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    for ep in range(1, cfg["epochs"] + 1):
        tr_loss = run_epoch(model, loaders["train"], device, opt, cfg["lam_bin"], weights,
                            sched, freeze_backbone=(freeze == "backbone"))
        va = evaluate(model, loaders["val"], device)
        try:
            auc = float(roc_auc_score(va["y_gate"], va["p_real"]))
        except ValueError:
            auc = float("nan")
        log(f"[train] epoch {ep:3d} | loss {tr_loss:.4f} | val gate AUC {auc:.4f}")

    tau = pick_threshold(va["p_real"], va["y_gate"], cfg["target_recall"])
    test = evaluate(model, loaders["test"], device)
    report = dict(config=plain_cfg(cfg), layers=layers, bin_vocab=vocab,
                  gate_test=gate_report(test["p_real"], test["y_gate"], tau, cfg["review_band"]),
                  gate_val=gate_report(va["p_real"], va["y_gate"], tau, cfg["review_band"]),
                  bin_test_oracle=bin_report(test, layers, vocab, "oracle"),
                  bin_test_cascade=bin_report(test, layers, vocab, "cascade", tau))
    out = rpath(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    torch.save(dict(model=model.state_dict(), arch=cfg["arch"], channels=cfg["channels"],
                    layers=layers, bin_vocab=vocab, mean=loaders["mean"], std=loaders["std"],
                    tau=tau, vit=dict(dim=cfg["vit_dim"], depth=cfg["vit_depth"],
                                      heads=cfg["vit_heads"], patch=cfg["patch_size"]),
                    config=plain_cfg(cfg)), out / "ckpt.pt")
    g = report["gate_test"]
    log(f"[train] TEST gate: auc={g['gate_auroc']:.4f} escape={g['escape_rate']:.4f} "
        f"noise_pass={g['noise_pass_rate']:.4f} review={g['review_rate']:.4f} tau={tau:.3f}")
    for L, d in report["bin_test_oracle"]["per_layer"].items():
        if d.get("n"):
            log(f"[train] TEST bin[{L}]: n={d['n']} macro_f1={d['macro_f1']:.4f}")
    log(f"[train] report -> {out/'report.json'}  ckpt -> {out/'ckpt.pt'}")
    return out / "ckpt.pt"


def _load_model_from_ckpt(cfg, ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    v = ckpt["vit"]
    ch = CHANNEL_SETS[ckpt["channels"]]

    def bb(arch, in_ch, _cfg):
        if arch == "vit_tiny":
            return ViT(in_ch, v["dim"], v["depth"], v["heads"], v["patch"])
        return build_backbone(arch, in_ch, cfg)

    model = ADCModel(bb(ckpt["arch"], len(ch), cfg), ckpt["bin_vocab"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt, ch, device


def stage_eval(cfg):
    ckpt_path = rpath(cfg["ckpt"]) if cfg["ckpt"] else rpath(cfg["out_dir"]) / "ckpt.pt"
    model, ckpt, ch, device = _load_model_from_ckpt(cfg, ckpt_path)
    layers, vocab = ckpt["layers"], ckpt["bin_vocab"]
    loaders = make_classification_loaders(
        cfg["data_dir"], ch, cfg["batch_size"], False, cfg["seed"],
        None, cfg["num_workers"], layers=layers, bin_vocab=vocab)
    tau = ckpt.get("tau", 0.5)
    test = evaluate(model, loaders["test"], device)
    report = dict(ckpt=str(ckpt_path), tau=tau,
                  gate_test=gate_report(test["p_real"], test["y_gate"], tau, cfg["review_band"]),
                  bin_test_oracle=bin_report(test, layers, vocab, "oracle"),
                  bin_test_cascade=bin_report(test, layers, vocab, "cascade", tau))
    out = rpath(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    g = report["gate_test"]
    log(f"[eval] TEST gate: auc={g['gate_auroc']:.4f} escape={g['escape_rate']:.4f} "
        f"noise_pass={g['noise_pass_rate']:.4f} review={g['review_rate']:.4f} (tau={tau:.3f})")
    log(f"[eval] -> {out/'eval_report.json'}")


# ---------------------------------------------------------------- baseline
def handcrafted_features(p: np.ndarray) -> np.ndarray:
    """diff 为主 + raw/compare 上下文的手工特征 (GBDT 门控基线用, 可解释)."""
    raw, cmp_, diff = p[0], p[1], p[2]
    a = np.abs(diff)
    feats = []
    for ch in (diff, a):
        feats += [ch.max(), ch.min(), ch.mean(), ch.std(),
                  np.quantile(ch, 0.99), np.quantile(ch, 0.01), np.quantile(ch, 0.95)]
    for t in (0.2, 0.35, 0.5, 0.75):          # 多阈值激活像素数: 尺寸/形状代理
        feats.append(float((a > t).sum()))
    c = a[8:24, 8:24].mean()                  # 中心 vs 全图能量 (defect 近中心先验)
    feats += [c, c - a.mean()]
    m = a > 0.35
    if m.sum() >= 2:                          # 二阶矩拉长度: scratch vs blob
        ys, xs = np.nonzero(m)
        wts = a[m]
        cx, cy = (xs * wts).sum() / wts.sum(), (ys * wts).sum() / wts.sum()
        cov = np.cov(np.stack([xs - cx, ys - cy]), aweights=wts)
        ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
        feats += [float(np.sqrt(ev[1] / max(ev[0], 1e-6))), float(m.mean())]
    else:
        feats += [0.0, float(m.mean())]
    feats += [raw.std(), cmp_.std(), raw.max() - cmp_.max()]
    return np.array(feats, np.float32)


def stage_baseline(cfg):
    """手工特征 + 直方图 GBDT 的门控基线; 深度模型不显著优于它时先查数据/标签."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    data_dir = rpath(cfg["data_dir"])
    rows = read_labels(data_dir / "labels.csv")
    if any(not r["split"] for r in rows):
        assign_splits(rows, seed=cfg["seed"])
    patches = load_patches(data_dir, rows)
    X = np.stack([handcrafted_features(p) for p in patches])
    y = np.array([r["is_real"] for r in rows])
    sp = np.array([r["split"] for r in rows])
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                         random_state=cfg["seed"])
    clf.fit(X[sp == "train"], y[sp == "train"])
    pv = clf.predict_proba(X[sp == "val"])[:, 1]
    pt = clf.predict_proba(X[sp == "test"])[:, 1]
    tau = pick_threshold(pv, y[sp == "val"], cfg["target_recall"])
    report = dict(n_features=int(X.shape[1]),
                  gate_val=gate_report(pv, y[sp == "val"], tau, cfg["review_band"]),
                  gate_test=gate_report(pt, y[sp == "test"], tau, cfg["review_band"]))
    out = rpath(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    g = report["gate_test"]
    log(f"[baseline] GBDT gate TEST: auc={g['gate_auroc']:.4f} escape={g['escape_rate']:.4f} "
        f"noise_pass={g['noise_pass_rate']:.4f} review={g['review_rate']:.4f} -> {out/'baseline_report.json'}")
    deep = out / "report.json"
    if deep.exists():
        d = json.loads(deep.read_text(encoding="utf-8")).get("gate_test", {})
        log(f"[baseline] 深度模型对照: gate auc={d.get('gate_auroc', float('nan')):.4f} "
            f"(不显著优于 GBDT 时, 先查数据/标签再谈模型)")
    return report


# ---------------------------------------------------------------- inference
@torch.no_grad()
def stage_infer(cfg):
    """批量推理: 全量 patch -> predictions.csv, 三段路由 + auto_real 样本的 bin 预测."""
    ckpt_path = rpath(cfg["ckpt"]) if cfg["ckpt"] else rpath(cfg["out_dir"]) / "ckpt.pt"
    model, ckpt, ch, device = _load_model_from_ckpt(cfg, ckpt_path)
    model.eval()
    layers, vocab = ckpt["layers"], ckpt["bin_vocab"]
    tau, band = ckpt["tau"], cfg["review_band"]

    data_dir = rpath(cfg["data_dir"])
    rows = read_labels(data_dir / "labels.csv")
    if any(not r["split"] for r in rows):
        assign_splits(rows, seed=cfg["seed"])
    patches = load_patches(data_dir, rows)
    ds = PatchDataset(patches, rows, None, layers, vocab, ckpt["mean"], ckpt["std"], ch)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"])

    def action_of(p: float) -> str:
        return "auto_real" if p >= tau else ("review" if p >= tau - band else "auto_noise")

    recs = []
    k = 0
    for img, _, _, layer_i in loader:          # shuffle=False: 与 ds.idx 顺序一致
        gate_logits, bin_logits = model(img.to(device), layer_i.to(device))
        p = torch.softmax(gate_logits.float(), 1)[:, 1].cpu().numpy()
        bin_logits = bin_logits.cpu()
        for j in range(len(p)):
            r = rows[ds.idx[k]]
            L = r["layer"]
            rec = dict(patch_id=r["patch_id"], layer=L, split=r["split"],
                       is_real=r["is_real"], bin_true=r["bin"] or "",
                       p_real=round(float(p[j]), 4), action=action_of(float(p[j])),
                       bin_pred="", bin_conf="")
            if rec["action"] == "auto_real" and L in vocab and vocab[L]:
                probs = torch.softmax(bin_logits[j, :len(vocab[L])], 0)
                bi = int(probs.argmax())
                rec["bin_pred"] = vocab[L][bi]
                rec["bin_conf"] = round(float(probs[bi]), 4)
            recs.append(rec)
            k += 1
    out = rpath(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    n = defaultdict(int)
    for rec in recs:
        n[rec["action"]] += 1
    log(f"[infer] {len(recs)} patches -> {csv_path} | "
        f"auto_real={n['auto_real']} review={n['review']} auto_noise={n['auto_noise']} "
        f"(tau={tau:.3f}, band={band})")
    log("[infer] review 样本即人审队列: 按 |p_real - tau| 升序排优先复判")


# ---------------------------------------------------------------- entry
def plain_cfg(cfg):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}


def run(cfg: dict):
    seed_all(cfg["seed"])
    torch.set_num_threads(max(os.cpu_count() - 1, 1))
    log(f"=== adc_pipeline | stage={cfg['stage']} ===")
    stage = cfg["stage"]
    if stage == "synth":
        stage_synth(cfg)
    elif stage == "pretrain":
        stage_pretrain(cfg)
    elif stage == "train":
        stage_train(cfg)
    elif stage == "heads":
        stage_train(cfg, freeze_override="backbone")   # 冻结 backbone 接新层/新数据
    elif stage == "baseline":
        stage_baseline(cfg)
    elif stage == "eval":
        stage_eval(cfg)
    elif stage == "infer":
        stage_infer(cfg)
    elif stage == "all":
        data_dir = rpath(cfg["data_dir"])
        if cfg["force_synth"] or not (data_dir / "labels.csv").exists():
            stage_synth(cfg)
        enc = stage_pretrain(cfg)
        cfg = dict(cfg, init_from=str(enc))            # all 模式自动接上预训练
        stage_train(cfg)
        stage_eval(cfg)
        stage_baseline(cfg)                            # GBDT 对照 (在深度 report 之后, 自动打印对比)
        stage_infer(cfg)                               # 全量 predictions.csv + 人审队列
    else:
        raise ValueError(f"未知 stage: {stage}")


def resolve_config(argv=None) -> dict:
    """无命令行参数 -> 用文件内 CONFIG; 有参数 -> 只覆盖显式传入项."""
    parser = argparse.ArgumentParser(description="ADC 单文件流水线 (详见文件头注释)")
    parser.add_argument("--stage", default=None,
                        choices=["synth", "pretrain", "train", "heads", "baseline",
                                 "eval", "infer", "all"])
    for k, v in CONFIG.items():
        if k == "stage":
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):   # --force-synth / --no-force-synth
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(flag, type=type(v), default=None)
    args = parser.parse_args(argv)
    cfg = dict(CONFIG)
    if argv is None and len(sys.argv) == 1:
        log("(无命令行参数: 使用文件内 CONFIG, 直接改它再运行即可)")
    for k in CONFIG:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    log("effective config:\n" + json.dumps(plain_cfg(cfg), ensure_ascii=False, indent=2))
    return cfg


if __name__ == "__main__":
    run(resolve_config())
