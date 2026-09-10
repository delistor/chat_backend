# -*- coding: utf-8 -*-
"""
单文件、零项目内部依赖。只需要: torch, torchvision, numpy, scikit-learn, Pillow。

这是一个经过完整实验验证的配方（MVTec-AD 15 类 image AUROC 0.984，ViT-B/14）：
  - 冻结 DINOv2 骨干（不微调，train 模式下 requires_grad=False + no_grad 前向）
  - 抽第 2/5/8/11 个 block 的原始输出（pre-LN、不做最终归一化），通道维拼接
  - 单个 FastFlow 2D flow 头在拼接特征上建模正常分布（NLL 越高越异常）
  - 训练: Adam lr=1e-3 wd=1e-5, bs=32, 150 epochs, 无增广, seed=42, fp32
  - 评分: 异常图上采样到输入尺寸 → 高斯平滑 → 去边框 m 像素 → top-N 均值
    （N 和 m 按物理尺度随输入分辨率自动缩放: 224→top100/m16/(5,5)σ4,
      448→top400/m32/(9,9)σ8。边框去除是因为卷积 padding 会在图像边缘
      产生系统性假阳性，top-N 池化对孤立单点噪声比 max 稳健。）

用法
----
# 1) 训练（train_dir 里只放正常图，递归扫描）
python dinov2_flow_standalone.py train --train-dir data/good --exp-dir runs/demo --size 448

# 2) 评估（test_dir/<缺陷类型>/*.png，其中 good/ 子目录为正常样本）
python dinov2_flow_standalone.py eval --exp-dir runs/demo --test-dir data/test

# 3) 纯打分（无标签审查模式；晶圆整图大图加 --tile 滑窗）
python dinov2_flow_standalone.py score --exp-dir runs/demo --images data/review --tile

接公司内部的 DINOv2
--------------------
--backbone timm     用 timm 下载 facebook DINOv2（默认；离线时用本地缓存）
--backbone hub      用 torch.hub facebookresearch/dinov2
--ckpt PATH         公司内部权重：按 timm 的 vit_base_patch14_dinov2 结构
                    加载（pretrained=False + load_state_dict, strict=False）
非标准结构 → 直接改 load_backbone()，返回任何标准 ViT
（有 .blocks 列表、输出 (B, tokens, C)）即可，特征抽取是 hook 实现的，不挑框架。

显存: ViT-B @448 bs32 训练峰值 ~2.9GB，@224 ~1GB；8GB 卡轻松跑。
Windows: num_workers 固定为 0（多进程 DataLoader 在 Windows 上有死锁前科）。
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OUT_INDICES = (2, 5, 8, 11)      # ViT-B 12 层中的第 3/6/9/12 层（0-based 2/5/8/11）
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


# ---------------------------------------------------------------- 骨干
def load_backbone(source="timm", ckpt=None, device="cuda"):
    """加载 DINOv2 ViT。公司内部权重优先走 --ckpt（tim­m 结构）。"""
    if source == "timm":
        import timm
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                              pretrained=(ckpt is None), dynamic_img_size=True)
        if ckpt is not None:
            sd = torch.load(ckpt, map_location="cpu")
            missing, unexpected = m.load_state_dict(sd, strict=False)
            print(f"[backbone] ckpt loaded: {len(missing)} missing / "
                  f"{len(unexpected)} unexpected keys")
    elif source == "hub":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    else:
        raise ValueError(f"unknown backbone source: {source}")
    # 冻结：本方法只训 flow 头。ViT 无 BN，eval() 只为保险。
    m.eval().requires_grad_(False)
    return m.to(device)


class FeatureExtractor(nn.Module):
    """hook 抽取多个 block 的原始输出 → 各自 reshape 成 (B,C,H,W) → 通道拼接。

    hook 实现不依赖 timm/hub 的 API，任何标准 ViT（含公司内部版）都能用。
    """

    def __init__(self, model, out_indices=OUT_INDICES):
        super().__init__()
        self.model = model
        self.out_indices = list(out_indices)
        self._feats = {}
        self._hooks = [model.blocks[i].register_forward_hook(self._mk(i))
                       for i in self.out_indices]

    def _mk(self, idx):
        def fn(_module, _inp, out, idx=idx):
            self._feats[idx] = out
        return fn

    def _prefix_tokens(self, n):
        # timm: num_prefix_tokens(1 或带 register 的 4)；hub dinov2: 1。
        for p in (getattr(self.model, "num_prefix_tokens", None), 1, 4):
            if p is None or n - p < 0:
                continue
            if math.isqrt(n - p) ** 2 == n - p:
                return p
        raise RuntimeError(f"无法推断 prefix token 数 (n={n})")

    @torch.no_grad()
    def forward(self, x):
        self._feats.clear()
        try:                                   # timm / 多数自定义 ViT 都有
            self.model.forward_features(x)
        except AttributeError:
            self.model(x)                      # 只要 hook 抓到了 block 输出即可
        outs = []
        for i in self.out_indices:
            t = self._feats[i]
            if t.dim() == 3:                   # (B, N, C) → 去掉 CLS/reg → (B,C,H,W)
                p = self._prefix_tokens(t.shape[1])
                n = t.shape[1] - p
                h = w = math.isqrt(n)
                t = t[:, p:].reshape(t.shape[0], h, w, -1).permute(0, 3, 1, 2)
            outs.append(t)
        c = outs[0].shape[1]
        assert all(o.shape[1] == c for o in outs), "各层 embed 维度不一致"
        return torch.cat(outs, dim=1)          # (B, 4C, H, W)


# ---------------------------------------------------------------- flow 头
class AffineCoupling2D(nn.Module):
    """RealNVP 式 2D 仿射耦合。末层卷积零初始化 → 初始为恒等映射，训练稳定。"""

    def __init__(self, channels, hidden):
        super().__init__()
        assert channels % 2 == 0
        h = channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(h, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 2 * h, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s, t = self.net(x1).chunk(2, dim=1)
        s = s.clamp(-15.0, 15.0)               # log-scale 限幅，防数值爆炸
        z2 = x2 * s.exp() + t
        logdet = s.sum(dim=1)                  # (B,H,W) 每位置 log|det|
        return torch.cat([x1, z2], dim=1), logdet


class FastFlow2D(nn.Module):
    """8 个耦合层，层间固定通道 shuffle。输入 (B,C,H,W)（C=拼接后的通道数）。"""

    def __init__(self, channels, hidden=512, n_coupling=8):
        super().__init__()
        self.n_coupling = n_coupling
        self.couplings = nn.ModuleList(
            [AffineCoupling2D(channels, hidden) for _ in range(n_coupling)])
        g = torch.Generator().manual_seed(SEED)   # 确定性 shuffle，随 ckpt 保存
        for i in range(n_coupling):
            self.register_buffer(f"perm_{i}", torch.randperm(channels, generator=g))

    def forward(self, x):
        z, logdet = x, 0.0
        for i, coupling in enumerate(self.couplings):
            z = z[:, getattr(self, f"perm_{i}")]
            z, ld = coupling(z)
            logdet = logdet + ld
        # 每空间位置 NLL（越高越异常）；总 logprob 用于训练
        nll_map = 0.5 * z.pow(2).sum(dim=1) - logdet          # (B,H,W)
        logprob = -nll_map.flatten(1).sum(dim=1)              # (B,)
        return nll_map, logprob


# ---------------------------------------------------------------- 数据
def build_transform(size, mode="resize"):
    # resize: 整图直接缩放（晶圆整图/宽幅推荐，配合 score --tile）
    # center-crop: 短边缩放后中心方形裁剪（物体居中的 MVTec 式数据用）
    ops = []
    if mode == "center-crop":
        ops.append(transforms.Resize(size, interpolation=Image.BICUBIC))
        ops.append(transforms.CenterCrop(size))
    else:
        ops.append(transforms.Resize((size, size), interpolation=Image.BICUBIC))
    ops += [transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class ImageListDataset(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), i


# ---------------------------------------------------------------- 评分
def scale_rule(size):
    """物理尺度规则：以 224 基准(top100/m16/(5,5)σ4)线性/平方缩放，448 精确对齐。"""
    k = size / 224.0
    top_n = max(1, round(100 * k * k))
    margin = round(16 * k)
    blur_k = 5 + 4 * (size // 224)              # 224→5, 448→9
    blur_sigma = 4.0 * k
    return top_n, margin, blur_k, blur_sigma


def image_scores(nll_maps, size):
    """(B,h,w) 原始 NLL → 上采样到输入尺寸 → 高斯平滑 → 去边框 → 池化。"""
    top_n, margin, blur_k, sigma = scale_rule(size)
    m = F.interpolate(nll_maps.unsqueeze(1), size=(size, size),
                      mode="bilinear", align_corners=False).squeeze(1)
    m = transforms.functional.gaussian_blur(
        m.unsqueeze(1), [blur_k, blur_k], [sigma, sigma]).squeeze(1)
    if margin > 0:                              # 卷积 padding 的边缘假阳性带
        m = m.clone()
        m[:, :margin, :] = -np.inf
        m[:, -margin:, :] = -np.inf
        m[:, :, :margin] = -np.inf
        m[:, :, -margin:] = -np.inf
    flat = m.flatten(1)
    top = flat.topk(min(top_n, flat.shape[1]), dim=1).values.mean(dim=1)
    return {"topN": top.cpu().numpy(),
            "max": flat.max(dim=1).values.cpu().numpy(),
            "mean": torch.where(torch.isfinite(flat), flat, 0).sum(1).cpu().numpy()}


@torch.no_grad()
def run_model(backbone, flow, images, device, bs=16):
    """images: 已预处理张量 (N,3,S,S)。返回 NLL 图与三组图级分数。"""
    maps, sc = [], {"topN": [], "max": [], "mean": []}
    backbone.eval()
    flow.eval()
    for i in range(0, len(images), bs):
        x = images[i:i + bs].to(device)
        feats = backbone(x)                     # FeatureExtractor 实例
        nll, _ = flow(feats)
        s = image_scores(nll, x.shape[-1])
        for k in sc:
            sc[k].append(s[k])
        maps.append(nll.cpu())
    return torch.cat(maps), {k: np.concatenate(v) for k, v in sc.items()}


def save_map_png(nll, path):
    a = nll.numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-12) * 255
    Image.fromarray(a.astype(np.uint8), mode="L").save(path)


# ---------------------------------------------------------------- 训练
def cmd_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()
    train_paths = list_images(args.train_dir)
    assert train_paths, f"train-dir 里没有图片: {args.train_dir}"
    print(f"[train] {len(train_paths)} images, size={args.size}, "
          f"epochs={args.epochs}, bs={args.batch_size}")

    backbone = FeatureExtractor(
        load_backbone(args.backbone, args.ckpt, device)).to(device)
    tf = build_transform(args.size, args.preprocess)
    loader = DataLoader(ImageListDataset(train_paths, tf),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    embed_dim = backbone.model.embed_dim if hasattr(backbone.model, "embed_dim") \
        else backbone(torch.zeros(1, 3, args.size, args.size, device=device)).shape[1]
    channels = embed_dim * len(OUT_INDICES)
    flow = FastFlow2D(channels, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    exp = Path(args.exp_dir)
    (exp / "maps").mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        flow.train()
        for x, _ in loader:
            with torch.no_grad():               # 骨干冻结，只对 flow 求梯度
                feats = backbone(x.to(device))
            _, logprob = flow(feats)
            loss = -logprob.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.1f}", flush=True)
            step += 1
        if (epoch + 1) % 30 == 0 or epoch == args.epochs - 1:   # 断点保护
            torch.save({"flow": flow.state_dict(),
                        "cfg": cfg_dict(args, embed_dim)}, exp / "latest.pt")
    torch.save({"flow": flow.state_dict(),
                "cfg": cfg_dict(args, embed_dim)}, exp / "final.pt")
    print(f"[train] done → {exp / 'final.pt'}")


def cfg_dict(args, embed_dim):
    return {"size": args.size, "embed_dim": embed_dim,
            "out_indices": list(OUT_INDICES), "hidden": args.hidden,
            "backbone": args.backbone, "ckpt": args.ckpt,
            "preprocess": args.preprocess}


def load_flow(exp_dir, device, source="final.pt"):
    ckpt = torch.load(Path(exp_dir) / source, map_location=device)
    cfg = ckpt["cfg"]
    backbone = FeatureExtractor(
        load_backbone(cfg["backbone"], cfg["ckpt"], device)).to(device)
    flow = FastFlow2D(cfg["embed_dim"] * len(cfg["out_indices"]),
                      hidden=cfg["hidden"]).to(device)
    flow.load_state_dict(ckpt["flow"])
    flow.eval()
    return backbone, flow, cfg


# ---------------------------------------------------------------- 评估/打分
def cmd_eval(args):
    from sklearn.metrics import roc_auc_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, flow, cfg = load_flow(args.exp_dir, device)
    tf = build_transform(cfg["size"], cfg["preprocess"])

    paths, types = [], []
    for p in list_images(args.test_dir):
        paths.append(p)
        types.append(p.parent.name)
    labels = np.array([0 if t == "good" else 1 for t in types])
    if labels.all() or not labels.any():
        raise SystemExit("test-dir 需要 good/ 子目录 + 至少一个缺陷子目录")
    print(f"[eval] {len(paths)} images "
          f"({int((labels == 0).sum())} good / {int(labels.sum())} defect)")

    images = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    maps, scores = run_model(backbone, flow, images, device, args.eval_bs)
    primary = scores["topN"]

    results = {"image_auroc": {k: float(roc_auc_score(labels, v))
                               for k, v in scores.items()},
               "per_type": {}}
    good = primary[labels == 0]
    for t in sorted(set(types) - {"good"}):
        sel = np.array([tt == t for tt in types])
        y = np.concatenate([np.ones(sel.sum()), np.zeros(len(good))])
        s = np.concatenate([primary[sel], good])
        results["per_type"][t] = float(roc_auc_score(y, s))

    print(f"[eval] image AUROC topN={results['image_auroc']['topN']:.4f} "
          f"max={results['image_auroc']['max']:.4f} "
          f"mean={results['image_auroc']['mean']:.4f}")
    for t, a in sorted(results["per_type"].items(), key=lambda kv: kv[1]):
        print(f"    {t:<30} {a:.4f}")
    exp = Path(args.exp_dir)
    json.dump(results, open(exp / "results.json", "w"), indent=2)

    # 保存最异常的 10 张的异常图，方便人工复核
    order = np.argsort(-primary)[:10]
    for i in order:
        save_map_png(maps[i], exp / "maps" / f"{paths[i].stem}_score{primary[i]:.1f}.png")
    print(f"[eval] top-10 anomaly maps → {exp / 'maps'}")


def cmd_score(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, flow, cfg = load_flow(args.exp_dir, device)
    tf = build_transform(cfg["size"], cfg["preprocess"])
    size = cfg["size"]
    paths = list_images(args.images)
    assert paths, f"没有找到图片: {args.images}"
    out_rows = []

    for p in paths:
        img = Image.open(p).convert("RGB")
        if args.tile and (img.width > size * 1.5 or img.height > size * 1.5):
            stride = int(size * 0.75)
            xs = sorted({min(x, max(img.width - size, 0))
                         for x in range(0, max(img.width - size, 0) + 1, stride)})
            ys = sorted({min(y, max(img.height - size, 0))
                         for y in range(0, max(img.height - size, 0) + 1, stride)})
            tiles = [img.crop((x, y, x + size, y + size)) for x in xs for y in ys]
            batch = torch.stack([tf(t) for t in tiles])
            maps, sc = run_model(backbone, flow, batch, device, args.eval_bs)
            best = int(np.argmax(sc["topN"]))
            score, nll = float(sc["topN"][best]), maps[best]
        else:
            batch = tf(img).unsqueeze(0)
            maps, sc = run_model(backbone, flow, batch, device, args.eval_bs)
            score, nll = float(sc["topN"][0]), maps[0]
        out_rows.append((str(p), score))
        if args.save_maps:
            Path(args.save_maps).mkdir(parents=True, exist_ok=True)
            save_map_png(nll, Path(args.save_maps) / f"{p.stem}.png")

    out_rows.sort(key=lambda r: -r[1])
    print(f"{'score':>12}  path")
    for p, s in out_rows:
        print(f"{s:12.2f}  {p}")
    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8") as f:
            f.write("score,path\n")
            f.writelines(f"{s:.4f},{p}\n" for p, s in out_rows)


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train-dir", required=True, help="只放正常图的目录（递归）")
    t.add_argument("--exp-dir", required=True)
    t.add_argument("--size", type=int, default=448,
                   help="模型输入尺寸；池化/平滑参数按物理尺度自动缩放")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-5)
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--backbone", choices=["timm", "hub"], default="timm")
    t.add_argument("--ckpt", default=None, help="公司内部 DINOv2 权重 (.pth)")
    t.add_argument("--preprocess", choices=["resize", "center-crop"],
                   default="resize")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--exp-dir", required=True)
    e.add_argument("--test-dir", required=True,
                   help="子目录=类别，good/=正常，其余=缺陷")
    e.add_argument("--eval-bs", type=int, default=16)
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("score")
    s.add_argument("--exp-dir", required=True)
    s.add_argument("--images", required=True, help="图片/目录/glob")
    s.add_argument("--tile", action="store_true",
                   help="大图滑窗（晶圆整图必开），图分 = max(窗分)")
    s.add_argument("--save-maps", default=None, help="异常图 PNG 输出目录")
    s.add_argument("--out-csv", default=None)
    s.add_argument("--eval-bs", type=int, default=16)
    s.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
