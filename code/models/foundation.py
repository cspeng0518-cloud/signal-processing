"""电磁通用感知基座模型组装（对应赛题任务1/2/3）

EMFoundation = IQTokenizer（时域分支）
             + 共享 Backbone（Transformer/Mamba 可选）
             + 三个任务头（体制/调制/参数）

权重共用核算（赛题红线 >80%）：
  共享部分 = tokenizer + backbone
  任务私有 = 三个 head 的 MLP
  主干:head 参数量比例通常 > 95:5，天然满足约束。

MAE 预训练（任务无关）：mask 掉部分 patch token，backbone 编码可见 token，
轻量 decoder 重建被掩码 patch 的原始 IQ 值。
"""

import torch
import torch.nn as nn

from .tokenizer import IQTokenizer
from .frequency import FrequencyTokenizer, CrossDomainFusion
from .backbone import build_backbone
from .heads import ClassificationHead, RegressionHead

# 官方类别数（task_schema.json）
N_PROTOCOL = 8
N_MODULATION = 23
N_PARAMS = 7


class MAEDecoder(nn.Module):
    """轻量 MAE 解码器（仅预训练使用，微调/推理时丢弃）"""

    def __init__(self, d_model=128, patch_size=8, n_layers=2, nhead=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, 2 * patch_size)  # 回到 IQ patch 维

    def forward(self, x):
        return self.proj(self.decoder(x))

class EMFoundation(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        backbone_name = config.get("backbone", "transformer")
        d_model = config.get("d_model", 128)
        patch_size = config.get("patch_size", 8)
        sr_bins = config.get("sr_bins", 64)
        mask_ratio = config.get("mask_ratio", 0.75)

        # ---- 共享部分（各任务权重共用的核心）----
        self.tokenizer = IQTokenizer(d_model, patch_size, sr_bins)
        self.backbone = build_backbone(
            backbone_name,
            d_model=d_model,
            n_layers=config.get("n_layers", 12),
            nhead=config.get("nhead", 4),
        )
        # ---- MAE 预训练专用 ----
        self.mae_decoder = MAEDecoder(
            d_model, patch_size, config.get("mae_decoder_layers", 2)
        )
        self.mask_ratio = mask_ratio
        self.use_frequency = bool(config.get("use_frequency", True))
        if self.use_frequency:
            self.frequency_tokenizer = FrequencyTokenizer(d_model, patch_size)
            self.cross_domain_fusion = CrossDomainFusion(d_model, config.get("nhead", 4))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.patch_size = patch_size
        # ---- 任务私有 head（task_heads 唯一持有，避免 safetensors 共享内存冲突）----
        self.task_heads = nn.ModuleDict({
            "protocol": ClassificationHead(d_model, N_PROTOCOL,
                                            config.get("head_hidden", 256)),
            "modulation": ClassificationHead(d_model, N_MODULATION,
                                             config.get("head_hidden", 256)),
            "parameter": RegressionHead(d_model, N_PARAMS,
                                        config.get("head_hidden", 256)),
        })
        # parameter 辅助头：通信/雷达模式二分类
        # （官方数据 valid 严格二分: 1110000 通信 / 0001111 雷达，
        #   推理时由该头硬决定不适用参数输出 0.0）
        self.task_heads["parameter_pattern"] = ClassificationHead(d_model, 2,
                                                                  config.get("head_hidden", 256))
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def shared_parameter_count(self):
        """共享（任务间共用）参数量"""
        total = sum(p.numel() for p in self.tokenizer.parameters()) + sum(p.numel() for p in self.backbone.parameters())
        if self.use_frequency:
            total += sum(p.numel() for p in self.frequency_tokenizer.parameters())
            total += sum(p.numel() for p in self.cross_domain_fusion.parameters())
        return total

    def head_parameter_count(self):
        """任务私有参数量"""
        return sum(p.numel() for h in self.task_heads.values() for p in h.parameters())

    def shared_ratio(self):
        """权重共用率（赛题要求 >80%）"""
        s, h = self.shared_parameter_count(), self.head_parameter_count()
        return s / (s + h)

    # ------------------------------------------------------------------
    def _random_mask(self, n_patch, B, device):
        """按样本随机掩码，返回 keep_ids [B,K]（CLS/SR 永不掩码，EMind V-C）"""
        K = max(int(n_patch * (1 - self.mask_ratio)), 1)
        # 对每个样本在 [2, 2+n_patch) 内取 K 个 patch 下标
        rand = torch.rand(B, n_patch, device=device)
        ids = torch.argsort(rand, dim=1)[:, :K] + 2  # 偏移跳过 CLS/SR
        return ids

    def forward_mae(self, iq, sr):
        """真正的 masked reconstruction：编码可见 patch，解码并仅监督遮蔽 patch。"""
        tokens, n_patch = self.tokenizer(iq, sr)
        B, _, D = tokens.shape
        keep_patch = self._random_mask(n_patch, B, tokens.device) - 2
        visible_index = torch.cat([
            torch.arange(2, device=iq.device).unsqueeze(0).expand(B, -1), keep_patch + 2], dim=1)
        visible = torch.gather(tokens, 1, visible_index.unsqueeze(-1).expand(-1,-1,D))
        if self.use_frequency:
            freq = self.frequency_tokenizer(iq)
            visible = self.cross_domain_fusion(visible, freq)
        encoded = self.backbone(visible)
        # 遮蔽位置只使用 learned mask token + positional encoding，绝不泄露原始 patch
        dec = self.mask_token.expand(B, n_patch + 2, D).clone()
        dec.scatter_(1, visible_index.unsqueeze(-1).expand(-1,-1,D), encoded)
        dec = self.tokenizer.pos_encoding(dec)
        reconstructed = self.mae_decoder(dec)[:, 2:]
        target = self.tokenizer.to_patches(self.tokenizer.normalize(iq))
        masked = torch.ones(B, n_patch, device=iq.device, dtype=torch.bool)
        masked.scatter_(1, keep_patch, False)
        return reconstructed[masked].reshape(B, -1, target.size(-1)), target[masked].reshape(B, -1, target.size(-1))

    # ------------------------------------------------------------------
    def encode(self, iq, sr):
        """共享编码：任务头使用的通用前向（无掩码，全序列）"""
        tokens, _ = self.tokenizer(iq, sr)
        if self.use_frequency:
            freq = self.frequency_tokenizer(iq)
            tokens = self.cross_domain_fusion(tokens, freq)
        return self.backbone(tokens)

    def forward(self, iq, sr, task):
        """多任务前向：task ∈ {'protocol','modulation','parameter','parameter_pattern'}

        Returns:
            protocol/modulation: logits [B, n_classes]
            parameter: 归一化参数 [B, 7]，范围 (0,1)
            parameter_pattern: 模式 logits [B, 2]（0=通信, 1=雷达）
        """
        ctx = self.encode(iq, sr)
        return self.task_heads[task](ctx)
