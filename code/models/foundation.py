"""电磁通用感知基座模型组装（对应赛题任务1/2/3）

EMFoundation = IQTokenizer（时域分支）
             + 共享 Backbone（Transformer / Hybrid(MLA+GDA+Mamba) / Mamba 可选）
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

from .tokenizer import IQTokenizer, token_max_len
from .frequency import FrequencyTokenizer, CrossDomainFusion
from .backbone import (build_backbone, RoPETransformerEncoder,
                       SinTransformerEncoder)
from .heads import ClassificationHead, RegressionHead

# 官方类别数（task_schema.json）
N_PROTOCOL = 8
N_MODULATION = 23
N_PARAMS = 7


class MAEDecoder(nn.Module):
    """轻量 MAE 解码器（仅预训练使用，微调/推理时丢弃）

    pos_encoding 与主干保持一致："sin" 时输入 token 已含可加性位置编码；
    "rope" 时解码注意力内部旋转（全序列 0..L-1 位置），输入无需叠加位置向量。
    """

    def __init__(self, d_model=128, patch_size=8, n_layers=2, nhead=4,
                 pos_encoding="sin", rope_max_len=512):
        super().__init__()
        if pos_encoding == "rope":
            self.decoder = RoPETransformerEncoder(
                d_model=d_model, n_layers=n_layers, nhead=nhead,
                dim_feedforward=4 * d_model, dropout=0.1,
                rope_max_len=rope_max_len)
        else:
            # 两分支均为 BaseEncoder（自建层栈 + final norm），forward 完全同构
            self.decoder = SinTransformerEncoder(
                d_model=d_model, n_layers=n_layers, nhead=nhead,
                dim_feedforward=4 * d_model, dropout=0.1)
        self.proj = nn.Linear(d_model, 2 * patch_size)  # 回到 IQ patch 维

    def forward(self, x):
        return self.proj(self.decoder(x))

class EMFoundation(nn.Module):
    """接收配置的 model 部分（{"model": {...}} 的内层 dict，
    由 training/common.normalize_config 保证新旧格式归一）。"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        backbone_name = config.get("backbone", "transformer")
        d_model = config.get("d_model", 128)
        patch_size = config.get("patch_size", 8)
        sr_bins = config.get("sr_bins", 64)
        mask_ratio = config.get("mask_ratio", 0.75)
        pos_encoding = config.get("pos_encoding", "sin")  # "sin" | "rope"
        rope_max_len = token_max_len(patch_size)

        # ---- 共享部分（各任务权重共用的核心）----
        self.tokenizer = IQTokenizer(d_model, patch_size, sr_bins,
                                     pos_encoding=pos_encoding)
        self.backbone = build_backbone(
            backbone_name,
            d_model=d_model,
            n_layers=config.get("n_layers", 12),
            nhead=config.get("nhead", 4),
            pos_encoding=pos_encoding,
            rope_max_len=rope_max_len,
            # MLP 通道维（ffn_hidden 未配置时各 backbone 用默认 4*d_model）
            dim_feedforward=config.get("ffn_hidden", 4 * d_model),
            # hybrid backbone 专属参数（其余 backbone 在工厂内弹出不感知）；
            # layer_types 为 GLM-5.2 式逐层列表，hybrid_layers 为旧键名
            layer_types=config.get("layer_types",
                                   config.get("hybrid_layers", ["mla", "gda", "mamba"])),
            mla_kv_lora_rank=config.get("mla_kv_lora_rank", 32),
            mla_q_lora_rank=config.get("mla_q_lora_rank", 32),
        )
        # ---- MAE 预训练专用 ----
        self.mae_decoder = MAEDecoder(
            d_model, patch_size, config.get("mae_decoder_layers", 2),
            nhead=config.get("nhead", 4), pos_encoding=pos_encoding,
            rope_max_len=rope_max_len,
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
        # 头部容量按任务独立配置, 三级回退:
        #   head_config.<task>.{hidden,layers}  → 全局 head_hidden/head_layers → 256/1
        # 任务难度不同允许不同参数量（parameter 7 参数联合回归计算量最大）;
        # parameter_pattern 未单独指定时跟随 parameter 档（同属 parameter 任务）。
        head_cfg = config.get("head_config", {})
        g_hidden = config.get("head_hidden", 256)
        g_layers = config.get("head_layers", 1)

        def _dims(task):
            t = head_cfg.get(task, {}) or {}
            return t.get("hidden", g_hidden), t.get("layers", g_layers)

        h_p, l_p = _dims("protocol")
        h_m, l_m = _dims("modulation")
        h_pa, l_pa = _dims("parameter")
        _pat = head_cfg.get("parameter_pattern", {}) or {}
        h_pat = _pat.get("hidden", h_pa)
        l_pat = _pat.get("layers", l_pa)

        self.task_heads = nn.ModuleDict({
            "protocol": ClassificationHead(d_model, N_PROTOCOL, h_p,
                                           hidden_layers=l_p),
            "modulation": ClassificationHead(d_model, N_MODULATION, h_m,
                                              hidden_layers=l_m),
            "parameter": RegressionHead(d_model, N_PARAMS, h_pa,
                                        hidden_layers=l_pa),
        })
        # parameter 辅助头：通信/雷达模式二分类
        # （官方数据 valid 严格二分: 1110000 通信 / 0001111 雷达，
        #   推理时由该头硬决定不适用参数输出 0.0）
        self.task_heads["parameter_pattern"] = ClassificationHead(d_model, 2,
                                                                  h_pat,
                                                                  hidden_layers=l_pat)
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
        """标准 MAE 前向：只编码可见 token，解码端为掩码位置插入可学习 mask token，
        仅对被掩码 patch 计算重建损失。

        Returns:
            recon:  [B, n_masked, 2*patch_size] 被掩码 patch 的重建值
            target: [B, n_masked, 2*patch_size] 被掩码 patch 的归一化原始值
        """
        tokens, n_patch = self.tokenizer(iq, sr)  # sin 模式下 token 已含位置编码
        B, _, D = tokens.shape

        # ---- 1. 采样掩码，收集可见 token（CLS/SR 永不掩码）----
        keep_patch = self._random_mask(n_patch, B, tokens.device) - 2   # [B,K] patch 下标
        visible_index = torch.cat(
            [torch.arange(2, device=tokens.device).unsqueeze(0).expand(B, -1),
             keep_patch + 2], dim=1)                                     # [B,K+2] token 下标
        visible = torch.gather(tokens, 1, visible_index.unsqueeze(-1).expand(-1, -1, D))
        if self.use_frequency:
            visible = self.cross_domain_fusion(visible, self.frequency_tokenizer(iq))
        # 仅编码可见 token（MAE 加速路径）；pos_ids 传原始下标，
        # 依赖位置的实现（RoPE）据此保持真实位置，其余实现忽略
        encoded = self.backbone(visible, pos_ids=visible_index)

        # ---- 2. 解码端全序列：可见位用 encoded，掩码位用 mask token ----
        dec = self.mask_token.expand(B, n_patch + 2, D).clone()
        if self.tokenizer.pos_encoding is not None:
            # 可加性位置编码：encoded 已含 tokenizer 阶段的 PE，只给 mask token
            # 补位置编码，避免可见 token 被重复叠加两次位置编码；
            # rope 模式无叠加，位置由解码注意力携带
            dec = dec + self.tokenizer.pos_encoding.pe[:, : n_patch + 2]
        dec.scatter_(1, visible_index.unsqueeze(-1).expand(-1, -1, D), encoded)
        reconstructed = self.mae_decoder(dec)[:, 2:]                     # 丢掉 CLS/SR 位

        # ---- 3. 仅监督被掩码 patch ----
        target = self.tokenizer.to_patches(self.tokenizer.normalize(iq))  # [B,P,2*ps]
        masked = torch.ones(B, n_patch, device=tokens.device, dtype=torch.bool)
        masked.scatter_(1, keep_patch, False)
        return (reconstructed[masked].reshape(B, -1, target.size(-1)),
                target[masked].reshape(B, -1, target.size(-1)))

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
