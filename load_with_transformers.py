"""
使用transformers直接加载中文CLIP模型，无需modelscope
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from PIL import Image
from torchvision import transforms
import json


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
    
    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        
        # 使用ModuleDict来匹配checkpoint的命名格式
        self.transformer = nn.ModuleDict({
            "resblocks": nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])
        })
        
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
    
    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        for block in self.transformer["resblocks"]:
            x = block(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_post(x[:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj
        
        return x


class ChineseCLIP(nn.Module):
    """
    中文CLIP模型，使用transformers组件构建
    """
    def __init__(self, vision_config_path="vision_model_config.json", 
                 text_config_path="text_model_config.json",
                 checkpoint_path="pytorch_model.bin"):
        super().__init__()
        
        # 加载配置
        with open(vision_config_path, 'r', encoding='utf-8') as f:
            vision_config_dict = json.load(f)
        with open(text_config_path, 'r', encoding='utf-8') as f:
            text_config_dict = json.load(f)
        
        self.embed_dim = vision_config_dict.get('embed_dim', 512)
        
        # 构建视觉编码器 (ViT-Base-Patch16)
        self.visual = VisionTransformer(
            input_resolution=vision_config_dict.get('image_resolution', 224),
            patch_size=vision_config_dict.get('vision_patch_size', 16),
            width=vision_config_dict.get('vision_width', 768),
            layers=vision_config_dict.get('vision_layers', 12),
            heads=12,
            output_dim=self.embed_dim
        )
        
        # 构建文本编码器配置 (RoBERTa-wwm-ext-base-chinese)
        self.text_config = BertConfig(
            vocab_size=text_config_dict.get('vocab_size', 21128),
            hidden_size=text_config_dict.get('text_hidden_size', 768),
            num_hidden_layers=text_config_dict.get('text_num_hidden_layers', 12),
            num_attention_heads=text_config_dict.get('text_num_attention_heads', 12),
            intermediate_size=text_config_dict.get('text_intermediate_size', 3072),
            hidden_act=text_config_dict.get('text_hidden_act', 'gelu'),
            hidden_dropout_prob=text_config_dict.get('text_hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=text_config_dict.get('text_attention_probs_dropout_prob', 0.1),
            max_position_embeddings=text_config_dict.get('text_max_position_embeddings', 512),
            type_vocab_size=text_config_dict.get('text_type_vocab_size', 2),
            initializer_range=text_config_dict.get('text_initializer_range', 0.02),
            layer_norm_eps=1e-12,
        )
        
        # 初始化文本模型
        self.text = BertModel(self.text_config)
        
        # 投影层 - 文本投影层是线性层，视觉投影使用visual.proj
        self.text_projection = nn.Linear(text_config_dict.get('text_hidden_size', 768), self.embed_dim)
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        
        # 加载权重
        self.load_weights(checkpoint_path)
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    def load_weights(self, checkpoint_path):
        """从checkpoint加载权重"""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        
        # 创建新的state_dict，转换key格式
        new_state_dict = {}
        for k, v in state_dict.items():
            # 去掉module.前缀
            if k.startswith("module."):
                k = k[7:]
            
            # 视觉模型权重 - 转换mlp的命名格式
            if k.startswith("visual."):
                new_k = k.replace(".mlp.c_fc.", ".mlp.0.")
                new_k = new_k.replace(".mlp.c_proj.", ".mlp.2.")
                new_state_dict[new_k] = v
            
            # 文本模型权重 - checkpoint中是bert.*，需要转换为text.*
            elif k.startswith("bert."):
                new_k = "text." + k[5:]  # 去掉bert.，加上text.
                new_state_dict[new_k] = v
            
            # text_projection在checkpoint中是一个参数，不是线性层
            elif k == "text_projection":
                # 保存文本投影权重，稍后手动加载
                self._text_proj_weight = v
            
            # logit_scale - 保存以便后续处理
            elif k == "logit_scale":
                self._logit_scale_value = v
        
        # 加载权重
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        
        # 手动设置text_projection权重
        if hasattr(self, '_text_proj_weight'):
            # checkpoint中的text_projection形状是 [hidden_size, embed_dim] = [768, 512]
            # nn.Linear的权重形状是 [out_features, in_features] = [embed_dim, hidden_size] = [512, 768]
            # 所以需要转置
            self.text_projection.weight.data = self._text_proj_weight.t()
            self.text_projection.bias.data.zero_()
            delattr(self, '_text_proj_weight')
            # 从missing_keys中移除text_projection相关的key
            missing_keys = [k for k in missing_keys if 'text_projection' not in k]
        
        # 手动设置logit_scale
        if hasattr(self, '_logit_scale_value'):
            self.logit_scale.data = self._logit_scale_value
            delattr(self, '_logit_scale_value')
            # 从missing_keys中移除logit_scale
            missing_keys = [k for k in missing_keys if k != 'logit_scale']
        
        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        print("权重加载完成")
    
    def encode_image(self, images):
        """
        编码图像
        images: PIL.Image 或 List[PIL.Image] 或 torch.Tensor [B, C, H, W]
        返回: 归一化的图像特征 [B, embed_dim]
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, list):
            images = torch.stack([self.image_transform(img) for img in images])
        
        image_embeds = self.visual(images)
        # visual.proj 已经是投影矩阵了，输出就是最终的图像特征
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds
    
    def encode_text(self, input_ids, attention_mask=None):
        """
        编码文本
        input_ids: [B, seq_len] token ids
        attention_mask: [B, seq_len] 注意力掩码
        返回: 归一化的文本特征 [B, embed_dim]
        """
        text_outputs = self.text(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS] token的输出
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds
    
    def forward(self, images=None, input_ids=None, attention_mask=None):
        """
        前向传播
        """
        outputs = {}
        if images is not None:
            outputs['image_embeds'] = self.encode_image(images)
        if input_ids is not None:
            outputs['text_embeds'] = self.encode_text(input_ids, attention_mask)
        return outputs


def load_chinese_clip(checkpoint_path="pytorch_model.bin", 
                      vision_config_path="vision_model_config.json",
                      text_config_path="text_model_config.json",
                      vocab_path="vocab.txt"):
    """
    加载中文CLIP模型和tokenizer
    """
    # 加载模型
    model = ChineseCLIP(vision_config_path, text_config_path, checkpoint_path)
    model.eval()
    
    # 加载tokenizer (使用BertTokenizer， vocab.txt是RoBERTa-wwm-ext-base-chinese的词表)
    tokenizer = BertTokenizer(vocab_path, do_lower_case=True)
    
    return model, tokenizer


if __name__ == "__main__":
    # 示例用法
    print("加载模型...")
    model, tokenizer = load_chinese_clip()
    
    # 测试文本编码
    print("\n=== 测试文本编码 ===")
    input_texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]
    inputs = tokenizer(input_texts, padding=True, truncation=True, 
                      max_length=77, return_tensors="pt")
    
    with torch.no_grad():
        text_embeds = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
        print(f"文本特征形状: {text_embeds.shape}")
        print(f"文本特征示例 (第一个文本的前5维): {text_embeds[0, :5]}")
    
    # 测试图像编码 (如果resources/pokemon.jpeg存在)
    print("\n=== 测试图像编码 ===")
    import os
    image_path = "resources/pokemon.jpeg"
    if os.path.exists(image_path):
        input_img = Image.open(image_path)
        with torch.no_grad():
            image_embeds = model.encode_image(input_img)
            print(f"图像特征形状: {image_embeds.shape}")
            print(f"图像特征示例 (前5维): {image_embeds[0, :5]}")
        
        # 计算图文相似度
        print("\n=== 图文匹配 ===")
        with torch.no_grad():
            logit_scale = torch.exp(model.logit_scale)
            logits = logit_scale * image_embeds @ text_embeds.t()
            probs = logits.softmax(dim=-1)
            print(f"图文匹配概率:")
            for i, text in enumerate(input_texts):
                print(f"  {text}: {probs[0, i].item():.4f}")
    else:
        print(f"图片不存在: {image_path}")
        print("跳过图像编码测试")
    
    print("\n模型加载和测试完成!")
