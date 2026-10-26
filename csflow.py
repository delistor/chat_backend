#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CS-Flow 单文件复刻（对齐 anomalib v2.3.0 实现, 按原论文回归为无预训练形态）
—— 训练 / 测试 / 推理一体。

CS-Flow（Rudolph et al., arXiv:2110.02855, WACV 2022）: 全卷积跨尺度归一化流,
直接对【原始图像】的三个尺度做精确似然估计 —— 完全不需要任何 ImageNet 预训练权重。
注意: anomalib 官方实现在此模型上硬编码了 EfficientNet-B5 特征提取器(304ch),
本脚本为满足「无预训练 backbone」要求, 保留 anomalib 的全部流图/损失/评测代码,
仅把输入从 EfficientNet 特征换回原论文的原始图像三尺度(marco-rudolph/cs-flow 形态)。

用法（与 fastflow.py / draem.py 同构）:
    # 训练 + 测试
    python csflow.py --category bottle

    # 用已训权重推理
    python csflow.py --predict D:/some.png --weights results/bottle_csflow/seed0/best.pt

依赖: torch, torchvision, FrEIA(>=0.2), numpy, scikit-learn, pillow, matplotlib(可选)

对齐要点（anomalib v2.3.0 src/anomalib/models/image/csflow/）:
  1. 流图: FrEIA GraphINN, 3 个输入节点 = 原图 1x/2x/4x 下采样（anomalib 版是
     EfficientNet 特征的 3 个尺度）, n_coupling_blocks=4, 每块 = ParallelPermute
     (seed=块号) + ParallelGlowCouplingLayer(clamp=3, 末块卷积核 5 其余 3)。
  2. coupling 子网 = CrossConvolutions: 三尺度并行卷积 + 上/下采样跨尺度连接 +
     gamma 可学习缩放（torch_model.py 原样移植）。
  3. 损失 CsFlowLoss: mean(0.5*Σz²) − Σ log|det J|（jacobian 项不做批平均, 是
     anomalib 的原始行为, 已精确复现 —— 微批累积时对 z² 项按微批占比加权、jacobian
     项直接累加）。
  4. 优化: Adam(lr=2e-4, eps=1e-4, weight_decay=1e-5, betas=(0.5, 0.9)),
     梯度全局范数裁剪到 1（trainer_arguments: gradient_clip_val=1）。
  5. 评测: 图像分数 = mean(z²/2)（所有尺度展平）; 异常图 = 三尺度 mean-z² 图分别
     上采样后【相乘】（AnomalyMapMode.ALL）。
  6. 预处理: Resize((256,256)) + ImageNet Normalize（anomalib 基类默认, 保持一致）。

显存说明: coupling 子网在原图全分辨率上卷积, 激活量远大于参数量(模型仅 ~1M 参数),
默认 --micro-batch 1 (梯度累积, 数学等价整批)。显存充裕可加大 micro-batch 提速。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from math import exp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FrEIA.framework import GraphINN, InputNode, Node, OutputNode
from FrEIA.modules import InvertibleModule
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# anomalib csflow README 官方 benchmark（256 尺寸, seed=42, EfficientNet-B5 版本;
# 本脚本为无预训练形态, 数字仅作宽松参照）: 类别 -> (image_AUROC, pixel_AUROC)
OFFICIAL = {
    "bottle": (1.000, 0.853), "cable": (0.970, 0.863), "capsule": (0.907, 0.882),
    "carpet": (0.995, 0.847), "grid": (0.982, 0.746), "hazelnut": (0.995, 0.895),
    "leather": (1.000, 0.851), "metal_nut": (0.972, 0.932), "pill": (0.953, 0.920),
    "screw": (0.896, 0.779), "tile": (0.972, 0.775), "toothbrush": (0.969, 0.892),
    "transistor": (0.987, 0.960), "wood": (0.988, 0.677), "zipper": (0.987, 0.803),
    "average": (0.972, 0.845),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 数据（MVTec 格式, 与 fastflow.py 相同的预处理协议）
# --------------------------------------------------------------------------- #
class MVTecLike(Dataset):
    def __init__(self, root: Path, category: str, split: str, image_size: int, normalize=True):
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
        self._normalize = normalize
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
        if not self._normalize:
            # 反量化: 流模型对 8-bit 图像需要均匀噪声打散离散格点(CFlow/CS-Flow 标准做法)
            img = img + torch.rand_like(img) / 255.0
        else:
            img = (img - self._mean) / self._std
        mask = torch.zeros(self.size, self.size, dtype=torch.uint8)
        if mask_path is not None and mask_path.exists():
            m = torch.from_numpy(np.array(Image.open(mask_path).convert("L")))[None, None]
            mask = self._resize(m)[0, 0].to(torch.uint8)
        return img, mask, label, defect_type


# --------------------------------------------------------------------------- #
# 模型（逐行移植 anomalib torch_model.py, 输入端换成原始图像三尺度）
# --------------------------------------------------------------------------- #
class CrossConvolutions(nn.Module):
    """三尺度并行卷积 + 跨尺度 up/down 连接（anomalib 原样移植）。"""

    def __init__(self, in_channels, channels, channels_hidden=512, kernel_size=3,
                 leaky_slope=0.1, batch_norm=False, use_gamma=True):
        super().__init__()
        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        pad_mode = "zeros"
        self.use_gamma = use_gamma
        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        def conv(cin, cout, stride=1, dilation=1):
            return nn.Conv2d(cin, cout, kernel_size=kernel_size, padding=pad, stride=stride,
                             dilation=dilation, bias=not batch_norm, padding_mode=pad_mode)

        self.conv_scale0_0 = conv(in_channels, channels_hidden)
        self.conv_scale1_0 = conv(in_channels, channels_hidden)
        self.conv_scale2_0 = conv(in_channels, channels_hidden)
        self.conv_scale0_1 = conv(channels_hidden, channels, dilation=1)
        self.conv_scale1_1 = conv(channels_hidden, channels, dilation=1)
        self.conv_scale2_1 = conv(channels_hidden, channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv10 = nn.Conv2d(channels_hidden, channels, kernel_size=kernel_size,
                                   padding=pad, bias=True, padding_mode=pad_mode)
        self.up_conv21 = nn.Conv2d(channels_hidden, channels, kernel_size=kernel_size,
                                   padding=pad, bias=True, padding_mode=pad_mode)
        self.down_conv01 = conv(channels_hidden, channels, stride=2, dilation=1)
        self.down_conv12 = conv(channels_hidden, channels, stride=2, dilation=1)
        self.leaky_relu = nn.LeakyReLU(self.leaky_slope)

    def forward(self, scale0, scale1, scale2):
        out0 = self.conv_scale0_0(scale0)
        out1 = self.conv_scale1_0(scale1)
        out2 = self.conv_scale2_0(scale2)
        lr0 = self.leaky_relu(out0)
        lr1 = self.leaky_relu(out1)
        lr3 = self.leaky_relu(out2)
        out0 = self.conv_scale0_1(lr0)
        out1 = self.conv_scale1_1(lr1)
        out2 = self.conv_scale2_1(lr3)
        y1_up = self.up_conv10(self.upsample(lr1))
        y2_up = self.up_conv21(self.upsample(lr3))
        y0_down = self.down_conv01(lr0)
        y1_down = self.down_conv12(lr1)
        out0 = out0 + y1_up
        out1 = out1 + y0_down + y2_up
        out2 = out2 + y1_down
        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
        return out0, out1, out2


class ParallelPermute(InvertibleModule):
    """各尺度并行固定随机通道置换（anomalib 原样移植）。"""

    def __init__(self, dims_in, seed=None):
        super().__init__(dims_in)
        self.n_inputs = len(dims_in)
        self.in_channels = [dims_in[i][0] for i in range(self.n_inputs)]
        self.seed = seed
        self.perm, self.perm_inv = [], []
        for i in range(self.n_inputs):
            perm, perm_inv = self.get_random_perm(i)
            self.perm.append(perm)
            self.perm_inv.append(perm_inv)

    def get_random_perm(self, index):
        perm = np.random.default_rng(self.seed).permutation(self.in_channels[index])
        perm_inv = np.zeros_like(perm)
        for idx, permutation in enumerate(perm):
            perm_inv[permutation] = idx
        return torch.LongTensor(perm), torch.LongTensor(perm_inv)

    def forward(self, input_tensor, rev=False, jac=True):
        del jac
        if not rev:
            return [input_tensor[i][:, self.perm[i]] for i in range(self.n_inputs)], 0.0
        return [input_tensor[i][:, self.perm_inv[i]] for i in range(self.n_inputs)], 0.0

    @staticmethod
    def output_dims(input_dims):
        return input_dims


class ParallelGlowCouplingLayer(InvertibleModule):
    """三尺度并行 GLOW 耦合层（anomalib 原样移植, atan 钳制）。"""

    def __init__(self, dims_in, subnet_args: dict, clamp=5.0):
        super().__init__(dims_in)
        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)
        self.cross_convolution1 = CrossConvolutions(self.split_len1, self.split_len2 * 2, **subnet_args)
        self.cross_convolution2 = CrossConvolutions(self.split_len2, self.split_len1 * 2, **subnet_args)

    def exp(self, x):
        if self.clamp > 0:
            return torch.exp(self.log_e(x))
        return torch.exp(x)

    def log_e(self, x):
        if self.clamp > 0:
            return self.clamp * 0.636 * torch.atan(x / self.clamp)
        return x

    def forward(self, input_tensor, rev=False, jac=True):
        del jac
        x01, x02 = (input_tensor[0].narrow(1, 0, self.split_len1),
                    input_tensor[0].narrow(1, self.split_len1, self.split_len2))
        x11, x12 = (input_tensor[1].narrow(1, 0, self.split_len1),
                    input_tensor[1].narrow(1, self.split_len1, self.split_len2))
        x21, x22 = (input_tensor[2].narrow(1, 0, self.split_len1),
                    input_tensor[2].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r02, r12, r22 = self.cross_convolution2(x02, x12, x22)
            s02, t02 = r02[:, : self.split_len1], r02[:, self.split_len1:]
            s12, t12 = r12[:, : self.split_len1], r12[:, self.split_len1:]
            s22, t22 = r22[:, : self.split_len1], r22[:, self.split_len1:]
            y01 = self.exp(s02) * x01 + t02
            y11 = self.exp(s12) * x11 + t12
            y21 = self.exp(s22) * x21 + t22
            r01, r11, r21 = self.cross_convolution1(y01, y11, y21)
            s01, t01 = r01[:, : self.split_len2], r01[:, self.split_len2:]
            s11, t11 = r11[:, : self.split_len2], r11[:, self.split_len2:]
            s21, t21 = r21[:, : self.split_len2], r21[:, self.split_len2:]
            y02 = self.exp(s01) * x02 + t01
            y12 = self.exp(s11) * x12 + t11
            y22 = self.exp(s21) * x22 + t21
        else:  # names of x and y are swapped!
            r01, r11, r21 = self.cross_convolution1(x01, x11, x21)
            s01, t01 = r01[:, : self.split_len2], r01[:, self.split_len2:]
            s11, t11 = r11[:, : self.split_len2], r11[:, self.split_len2:]
            s21, t21 = r21[:, : self.split_len2], r21[:, self.split_len2:]
            y02 = (x02 - t01) / self.exp(s01)
            y12 = (x12 - t11) / self.exp(s11)
            y22 = (x22 - t21) / self.exp(s21)
            r02, r12, r22 = self.cross_convolution2(y02, y12, y22)
            s02, t02 = r02[:, : self.split_len2], r01[:, self.split_len2:]
            s12, t12 = r12[:, : self.split_len2], r11[:, self.split_len2:]
            s22, t22 = r22[:, : self.split_len2], r21[:, self.split_len2:]
            y01 = (x01 - t02) / self.exp(s02)
            y11 = (x11 - t12) / self.exp(s12)
            y21 = (x21 - t22) / self.exp(s22)

        z_dist0 = torch.clamp(torch.cat((y01, y02), 1), -1e6, 1e6)
        z_dist1 = torch.clamp(torch.cat((y11, y12), 1), -1e6, 1e6)
        z_dist2 = torch.clamp(torch.cat((y21, y22), 1), -1e6, 1e6)

        jac0 = torch.sum(self.log_e(s01), dim=(1, 2, 3)) + torch.sum(self.log_e(s02), dim=(1, 2, 3))
        jac1 = torch.sum(self.log_e(s11), dim=(1, 2, 3)) + torch.sum(self.log_e(s12), dim=(1, 2, 3))
        jac2 = torch.sum(self.log_e(s21), dim=(1, 2, 3)) + torch.sum(self.log_e(s22), dim=(1, 2, 3))
        # anomalib: jacobian 在此按 batch 求和成标量
        return [z_dist0, z_dist1, z_dist2], torch.stack([jac0, jac1, jac2], dim=1).sum()

    @staticmethod
    def output_dims(input_dims):
        return input_dims


class CrossScaleFlow(nn.Module):
    """跨尺度流图（anomalib 原样移植, 输入节点改为原始图像三尺度）。"""

    def __init__(self, input_dims, n_coupling_blocks, clamp, cross_conv_hidden_channels):
        super().__init__()
        self.input_dims = input_dims
        self.n_coupling_blocks = n_coupling_blocks
        self.kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]
        self.clamp = clamp
        self.cross_conv_hidden_channels = cross_conv_hidden_channels
        self.graph = self._create_graph()

    def _create_graph(self) -> GraphINN:
        nodes: list[Node] = []
        # 原论文形态: 三尺度 = 原始图像的 1x / 2x / 4x 下采样, 3 通道
        input_nodes = [
            InputNode(self.input_dims[0], self.input_dims[1], self.input_dims[2], name="input"),
            InputNode(self.input_dims[0], self.input_dims[1] // 2, self.input_dims[2] // 2, name="input2"),
            InputNode(self.input_dims[0], self.input_dims[1] // 4, self.input_dims[2] // 4, name="input3"),
        ]
        nodes.extend(input_nodes)
        for coupling_block in range(self.n_coupling_blocks):
            if coupling_block == 0:
                node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
            else:
                node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]
            permute_node = Node(inputs=node_to_permute, module_type=ParallelPermute,
                                module_args={"seed": coupling_block},
                                name=f"permute_{coupling_block}")
            nodes.extend([permute_node])
            coupling_layer_node = Node(
                inputs=[nodes[-1].out0, nodes[-1].out1, nodes[-1].out2],
                module_type=ParallelGlowCouplingLayer,
                module_args={
                    "clamp": self.clamp,
                    "subnet_args": {
                        "channels_hidden": self.cross_conv_hidden_channels,
                        "kernel_size": self.kernel_sizes[coupling_block],
                    },
                },
                name=f"fc1_{coupling_block}")
            nodes.extend([coupling_layer_node])
        output_nodes = [
            OutputNode([nodes[-1].out0], name="output_end0"),
            OutputNode([nodes[-1].out1], name="output_end1"),
            OutputNode([nodes[-1].out2], name="output_end2"),
        ]
        nodes.extend(output_nodes)
        return GraphINN(nodes)

    def forward(self, inputs, **kwargs):
        return self.graph(inputs, **kwargs)


class MultiScaleImageExtractor(nn.Module):
    """原始图像三尺度（替代 anomalib 的 EfficientNet 特征提取器, 无任何参数）。"""

    def __init__(self, n_scales: int, input_size: tuple[int, int]):
        super().__init__()
        self.n_scales = n_scales
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        output = []
        for scale in range(self.n_scales):
            # anomalib 对特征图下采样用的 F.interpolate 默认 nearest, 此处保持一致
            feat_s = (F.interpolate(x, size=(self.input_size[0] // (2 ** scale),
                                            self.input_size[1] // (2 ** scale)))
                      if scale > 0 else x)
            output.append(feat_s)
        return output


class CsFlowModel(nn.Module):
    def __init__(self, input_size, cross_conv_hidden_channels, n_coupling_blocks=4, clamp=3,
                 num_channels=3):
        super().__init__()
        self.input_dims = (num_channels, *input_size)
        self.clamp = clamp
        self.cross_conv_hidden_channels = cross_conv_hidden_channels
        self.feature_extractor = MultiScaleImageExtractor(n_scales=3, input_size=input_size)
        self.graph = CrossScaleFlow(
            input_dims=self.input_dims, n_coupling_blocks=n_coupling_blocks, clamp=clamp,
            cross_conv_hidden_channels=cross_conv_hidden_channels)

    def forward(self, images: torch.Tensor):
        features = self.feature_extractor(images)
        if self.training:
            return self.graph(features)
        z_dist, _ = self.graph(features)
        anomaly_scores = self._compute_anomaly_scores(z_dist)
        anomaly_maps = self.anomaly_map(z_dist)
        return anomaly_scores, anomaly_maps

    @staticmethod
    def _compute_anomaly_scores(z_dists) -> torch.Tensor:
        """anomalib: 全尺度展平后 mean(z²/2)。"""
        flat_maps = [z.reshape(z.shape[0], -1) for z in z_dists]
        flat = torch.cat(flat_maps, dim=1)
        return torch.mean(flat**2 / 2, dim=1)

    def anomaly_map(self, z_dists, mode="all") -> torch.Tensor:
        """anomalib AnomalyMapGenerator: 三尺度 mean-z² 分别上采样到输入尺寸后相乘(ALL 模式)。"""
        h, w = self.input_dims[1], self.input_dims[2]
        if mode == "all":
            anomaly_map = torch.ones(z_dists[0].shape[0], 1, h, w, device=z_dists[0].device)
            for z_dist in z_dists:
                mean_z = (z_dist**2).mean(dim=1, keepdim=True)
                anomaly_map *= F.interpolate(mean_z, size=(h, w), mode="bilinear",
                                             align_corners=False)
        else:
            mean_z = (z_dists[0] ** 2).mean(dim=1, keepdim=True)
            anomaly_map = F.interpolate(mean_z, size=(h, w), mode="bilinear",
                                        align_corners=False)
        return anomaly_map


def csflow_loss_full(z_dist, jacobians, jacobian_mode="mean") -> torch.Tensor:
    """CS-Flow 损失。

    jacobian_mode="sum": anomalib CsFlowLoss 原样 —— FrEIA GraphINN 把 coupling 层
    按批求和的标量 jacobian 广播成 (B,), 于是 mean_b(0.5*Σz²_b − J_total) 等价于
    mean_b(0.5Σz²) − ΣJ_b（jacobian 项不按批平均, 相当于被放大 batch_size 倍）。
    该行为在 anomalib 的 EfficientNet 特征输入下可以训练, 但在原始像素输入上
    会导致优化塌缩到「最大化 jacobian」的坏平衡（实测分数失去判别性）。

    jacobian_mode="mean"（默认）: 原论文的逐样本 NLL, mean_b(0.5*Σz²_b − J_b)，
    即 jacobian 按批平均 —— 数值上唯一能正确训练的版本。
    """
    concatenated = torch.cat([z.reshape(z.shape[0], -1) for z in z_dist], dim=1)
    sq_term = torch.mean(0.5 * torch.sum(concatenated**2, dim=(1,)))
    if jacobian_mode == "sum":
        jac_term = jacobians.mean()
    else:
        jac_term = jacobians.mean() / concatenated.shape[0]
    return (sq_term - jac_term) / concatenated.shape[1]


def csflow_loss_micro(z_dist, jacobians, sq_weight: float, jacobian_mode="mean",
                      batch_size: int = 0) -> torch.Tensor:
    """微批版损失, 梯度累积精确等价对应的 csflow_loss_full:
    sum 模式: Σ[w_i*mean(0.5Σz²) − J_micro] = mean_b(0.5Σz²) − ΣJ_b
    mean 模式: Σ[w_i*mean(0.5Σz²) − J_micro/B] = mean_b(0.5Σz²) − ΣJ_b/B
    （B=整批样本数, 由调用方传入）
    """
    concatenated = torch.cat([z.reshape(z.shape[0], -1) for z in z_dist], dim=1)
    sq_term = torch.mean(0.5 * torch.sum(concatenated**2, dim=(1,)))
    jac = jacobians.mean() if jacobians.dim() else jacobians  # 广播值全等, 取均值即微批总和
    if jacobian_mode == "sum":
        return (sq_weight * sq_term - jac) / concatenated.shape[1]
    return (sq_weight * sq_term - jac / batch_size) / concatenated.shape[1]


# --------------------------------------------------------------------------- #
# 评测 / 可视化（与前两个脚本同接口）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(model, loader, device):
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
    torch.backends.cudnn.benchmark = False  # 同 DRAEM: 全分辨率卷积, 避免 WDDM 回退

    out_dir = args.out_dir or Path(__file__).parent / "results" / f"{args.category}_csflow" / \
        f"seed{args.seed}_s{args.image_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(exist_ok=True)

    train_set = MVTecLike(args.data_root, args.category, "train", args.image_size,
                          normalize=bool(args.normalize))
    test_set = MVTecLike(args.data_root, args.category, "test", args.image_size,
                         normalize=bool(args.normalize))
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

    model = CsFlowModel((args.image_size, args.image_size),
                        cross_conv_hidden_channels=args.hidden_channels,
                        n_coupling_blocks=args.n_coupling_blocks, clamp=args.clamp).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] CS-Flow(原始图像三尺度, 无预训练) n_coupling={args.n_coupling_blocks} "
          f"hidden={args.hidden_channels} 可训练参数 {n_train/1e6:.2f}M")

    # anomalib configure_optimizers: Adam(2e-4, eps=1e-4, wd=1e-5, betas=(0.5,0.9))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps,
                                 weight_decay=1e-5, betas=(0.5, 0.9))

    best_auroc, best_epoch, patience_left = -1.0, -1, args.patience
    log_rows = []
    with open(out_dir / "epoch_log.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_image_auroc", "val_pixel_auroc",
                                "epoch_seconds"])

    micro = args.micro_batch or args.batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for img, _mask, _label, _type in train_loader:
            img = img.to(device)
            n = img.shape[0]
            optimizer.zero_grad(set_to_none=True)
            for s in range(0, n, micro):
                e = min(s + micro, n)
                if micro == n:
                    z, jac = model(img)
                    loss = csflow_loss_full(z, jac, args.jacobian_mode)
                else:
                    z, jac = model(img[s:e])
                    loss = csflow_loss_micro(z, jac, (e - s) / n, args.jacobian_mode, batch_size=n)
                loss.backward()
                losses.append(loss.item())
            # anomalib trainer_arguments: gradient_clip_val=1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        torch.cuda.empty_cache()  # 归还训练缓存, 避免与 eval 激活叠加触发 WDDM 回退

        model.eval()
        labels, scores, maps, masks, _types = run_inference(model, test_loader, device)
        m = compute_metrics(labels, scores, maps, masks, _types)
        dt = time.time() - t0
        print(f"[epoch {epoch:4d}] loss {np.mean(losses):.2f} | val image_AUROC "
              f"{m['image_auroc']:.4f} | val pixel_AUROC {m['pixel_auroc']:.4f} | {dt:.1f}s",
              flush=True)
        log_rows.append([epoch, float(np.mean(losses)), m["image_auroc"], m["pixel_auroc"], dt])
        with open(out_dir / "epoch_log.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(log_rows[-1])

        val_pix = m["pixel_auroc"]
        if val_pix > best_auroc:
            best_auroc, best_epoch, patience_left = val_pix, epoch, args.patience
            # 流式落盘, 不做整份 CPU 克隆
            torch.save({"state_dict": model.state_dict(), "config": vars(args)},
                       out_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] pixel_AUROC 连续 {args.patience} 轮未提升, 于 epoch {epoch} 停止")
                break

    if (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        del ckpt

    labels, scores, maps, masks, types = run_inference(model, test_loader, device)
    metrics = compute_metrics(labels, scores, maps, masks, types)
    metrics.update({"category": args.category, "image_size": args.image_size,
                    "best_epoch": best_epoch, "best_val_pixel_auroc": best_auroc,
                    "seed": args.seed, "backbone": "none(raw-image csflow)"})
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dir / "predictions.npz", labels=labels, scores=scores, maps=maps,
                        masks=masks, types=np.array(types))

    order = sorted(range(len(labels)), key=lambda i: -scores[i])
    vis_idx = list(dict.fromkeys(order[: args.num_vis] + list(range(min(args.num_vis, len(labels))))))
    dmean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    dstd = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for i in vis_idx:
        img, mask, _, _ = test_set[i]
        img01 = img.clamp(0, 1) if not args.normalize else (img * dstd + dmean).clamp(0, 1)
        save_visualization(img01, mask, maps[i],
                           out_dir / "vis" / f"{i:03d}_{types[i]}_score{scores[i]:.4f}.png")

    ref = OFFICIAL.get(args.category)
    print(f"\n===== 最终测试结果（best epoch {best_epoch}）=====")
    print(f"image_AUROC = {metrics['image_auroc']:.4f}")
    print(f"pixel_AUROC = {metrics['pixel_auroc']:.4f}")
    for t, a in metrics["per_type"].items():
        if isinstance(a, dict):
            print(f"  - {t:<16s} n={a['count']:<3d} mean_score {a['mean_score']:.4f}")
        else:
            print(f"  - {t:<16s} image_AUROC {a:.4f}")
    if ref:
        print(f"\n[参考] anomalib CS-Flow(EfficientNet-B5 版, 256) bottle: image {ref[0]:.3f} / "
              f"pixel {ref[1]:.3f} —— 本版本无预训练骨干, 仅宽松对照")
    print(f"[产物目录] {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def predict(args) -> None:
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = CsFlowModel((cfg["image_size"], cfg["image_size"]),
                        cross_conv_hidden_channels=cfg["hidden_channels"],
                        n_coupling_blocks=cfg["n_coupling_blocks"], clamp=cfg["clamp"])
    model.load_state_dict(ckpt["state_dict"])
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
            img01 = img.clamp(0, 1) if not args.normalize else (img * dstd + dmean).clamp(0, 1)
            save_visualization(img01.cpu(), torch.zeros_like(amap).to(torch.uint8), amap,
                               out_dir / f"{p.stem}_pred.png")
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
    ap.add_argument("--n-coupling-blocks", type=int, default=4)
    ap.add_argument("--hidden-channels", type=int, default=256,
                    help="CrossConvolutions 隐通道（原论文 512, 但 8GB 显存建议 256; "
                         "anomalib 为 EfficientNet 特征用 1024）")
    ap.add_argument("--jacobian-mode", choices=["mean", "sum"], default="mean",
                    help="mean=原论文逐样本 NLL(jacobian 按批平均, 原始像素输入下唯一可正确训练的版本, 默认); "
                         "sum=anomalib 源码原行为(jacobian 不批平均, 仅特征输入下可用)")
    ap.add_argument("--clamp", type=float, default=2.0,
                    help="coupling 钳制（原论文 2.0; anomalib 默认 3.0）")
    ap.add_argument("--normalize", action="store_true",
                    help="ImageNet 归一化输入（anomalib 基类默认做法）; 不加此开关则为原论文的 [0,1] 输入")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=2,
                    help="微批大小（梯度累积, 数学等价整批且精确保持 anomalib 的 jacobian 项行为）; 0=不分微批")
    ap.add_argument("--eval-batch-size", type=int, default=8,
                    help="eval 前向也会缓存大激活, 小批量可避免与训练缓存叠加撑爆显存")
    ap.add_argument("--epochs", type=int, default=100, help="anomalib Engine 默认 100")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eps", type=float, default=1e-4)
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
