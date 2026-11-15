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
        return sum(p.numel() for p in self.tokenizer.parameters()) + \
               sum(p.numel() for p in self.backbone.parameters())

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
        """MAE 预训练前向：掩码 -> 编码可见 token -> 重建被掩码 patch

        Returns:
            recon: [B, n_masked_patch, 2*patch_size] 重建的 IQ patch
            target: [B, n_masked_patch, 2*patch_size] 原始归一化 patch
        """
        tokens, n_patch = self.tokenizer(iq, sr)  # [B, P+2, D]
        B = tokens.size(0)
        keep_ids = self._random_mask(n_patch, B, tokens.device)
        ctx = self.backbone(tokens, keep_ids=keep_ids)  # 掩码位置为 0

        # 被掩码位置的目标：归一化后的原始 patch
        iq_n = self.tokenizer.normalize(iq)
        patches = self.tokenizer.to_patches(iq_n)  # [B,P,2*ps]
        # 被掩码 patch 下标 = keep_ids - 2
        masked_ids = keep_ids - 2
        target = torch.gather(
            patches, 1, masked_ids.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        )
        # 重建：只解码被掩码位置的 token（位置嵌入信息由位置编码提供）
        masked_tokens = torch.gather(ctx, 1, keep_ids.unsqueeze(-1).expand(-1, -1, ctx.size(-1)))
        recon = self.mae_decoder(masked_tokens)
        return recon, target

    # ------------------------------------------------------------------
    def encode(self, iq, sr):
        """共享编码：任务头使用的通用前向（无掩码，全序列）"""
        tokens, _ = self.tokenizer(iq, sr)
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
