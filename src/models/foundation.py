"""EMFoundation：电磁通用感知基座模型（赛题任务 1/2/3 的模型侧实现）

整体结构
--------
          IQ [B,N,2]
              │
      SignalFrontEnd（dsp.py，物理先验前端，无参数，fp32）
        ├── 时域 10 通道 / STFT / 40 维统计量 ─────────► TimeTokenizer ──► [CLS][STAT][TIME×T]
        └── 非线性谱+共轭循环谱 / 循环自相关 CAF / 去斜谱 /
            包络+复自相关 / 三通道星座图 / 分布直方图 ──► FrequencyStream ──► [FREQ×Lf]
                                                   │
              CrossDomainFusion（时域 query ← 频域 key/value，门控残差）
                                                   │
              backbone: HybridEncoder（FA : MLA = 1 : 2，RoPE）      ← MAE 预训练 + 微调
                                                   │
              neck: HybridEncoder（Conformer × neck_layers，序列尾部拼接频域 token）
                                                   │                  ← 仅微调/推理
              task_heads: 任务提示 + 注意力池化 + MLP（protocol / modulation / parameter）

预训练（MAE，只编码可见 token，省显存的关键）
------------------------------------------
  1. 按有效长度对时域 patch 随机掩码（默认 75%），CLS/STAT 永不掩码；
  2. 被掩码 patch 对应的采样点在送入前端前置零 —— 频域流 / STAT / STFT
     只能看到可见部分（修复旧版“整段 FFT 泄漏被掩码内容”的缺陷）；
  3. gather 可见 token（约 25%）进入主干，pos_ids 携带原始位置（RoPE）；
  4. 轻量解码器（mae_decoder_dim，默认 256）在全长序列上插入 mask token，
     重建被掩码 patch 的单位功率 IQ + 对应 STFT 帧（时域 + 时频双目标）。
  Conformer neck 不参与预训练（卷积需要连续序列，与 gather 路径不兼容）。

权重共用核算（赛题红线 >80%）
----------------------------
  共享 = tokenizer（除任务提示） + frequency stream + fusion + backbone + neck
  私有 = task_heads + 任务提示嵌入
  仅预训练 = mae_decoder（微调/推理不构建、不导出、不计入）
"""

import torch
import torch.nn as nn

from .backbone import HybridEncoder
from .dsp import (DEFAULT_CAF_LAGS, DEFAULT_CCAF_LAGS, DEFAULT_DECHIRP_RATES,
                  SignalFrontEnd)
from .frequency import CrossDomainFusion, FrequencyStream
from .heads import ClsHead, ParamHead
from .layers.hybrid import CONTIGUOUS_ONLY, normalize_layer_type
from .tokenizer import N_PREFIX, TimeTokenizer

N_PROTOCOL, N_MODULATION = 8, 23

# 调制族（层次化辅助监督）：0=PSK 1=QAM 2=FSK/CPM 3=模拟/其他 4=雷达
MOD_FAMILY = [
    3, 0, 0, 0, 1, 1, 1, 1,      # OOK BPSK QPSK 8PSK 16QAM 32QAM 64QAM 256QAM
    0, 2, 2, 2, 1, 0, 3, 2,      # OQPSK 2FSK 4FSK MSK 128QAM AQPSK CSS GFSK
    0, 3,                        # pi_4DQPSK FM
    4, 4, 4, 4, 4,               # Radar/*
]
N_FAMILY = 5

DEFAULT_BACKBONE_LAYERS = ["full_attention", "mla", "mla"]


class MAEDecoder(nn.Module):
    """轻量 MAE 解码器（仅预训练）：d_model -> dec_dim，RoPE 全注意力，输出 IQ+STFT 重建。"""

    def __init__(self, d_model, dec_dim, n_layers, out_dim, rope_max_len):
        super().__init__()
        nhead = max(1, dec_dim // 64)
        self.embed = nn.Linear(d_model, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = HybridEncoder(dec_dim, max(1, n_layers), nhead, ["full_attention"],
                                    dim_feedforward=4 * dec_dim, dropout=0.0,
                                    use_rope=True, rope_max_len=rope_max_len)
        self.pred = nn.Linear(dec_dim, out_dim)

    def forward(self, encoded, pos_ids, keep_tok, key_padding_mask=None):
        """
        encoded:  [B,K',D] 主干输出（可见 token + 前缀；含被标记为无效的填充槽位）
        pos_ids:  [B,K'] 这些 token 在全长序列中的下标
        keep_tok: [B,L] 全长序列中真正可见的位置（其余位置填 mask token）
        """
        x = self.embed(encoded)
        B, L = keep_tok.shape
        buf = x.new_zeros(B, L, x.size(-1))
        buf.scatter_(1, pos_ids.unsqueeze(-1).expand(-1, -1, x.size(-1)), x)
        dec = torch.where(keep_tok.unsqueeze(-1), buf, self.mask_token.to(buf.dtype))
        return self.pred(self.blocks(dec, key_padding_mask=key_padding_mask))


class EMFoundation(nn.Module):
    """Args:
        config:       配置的 model 部分（{"model": {...}} 的内层 dict）
        with_decoder: 是否构建 MAE 解码器（预训练 True；微调/推理 False）
    """

    def __init__(self, config, with_decoder=False):
        super().__init__()
        self.config = c = config
        d = c.get("d_model", 512)
        nhead = c.get("nhead", 8)
        self.patch = c.get("patch_size", 8)
        self.seq_len = c.get("seq_len", 1024)
        ffn = c.get("ffn_hidden", 4 * d)
        dropout = c.get("dropout", 0.1)
        stft_win = c.get("stft_win", 32)
        self.trim_padding = c.get("trim_padding", True)
        rope_max_len = self.seq_len // self.patch + N_PREFIX + 8

        # ---------------- 共享：前端 + 双流 tokenizer + 融合 ----------------
        acf_lags = c.get("acf_lags", self.seq_len // 2)
        use_dechirp = c.get("use_dechirp", True)
        self.front = SignalFrontEnd(
            self.seq_len, self.patch, stft_win, acf_lags,
            const_bins=c.get("const_bins", 64), const_range=c.get("const_range", 2.5),
            dft_impl=c.get("dft_impl", "matmul"),
            length_quantum=c.get("length_quantum", 512),
            caf_lags=c.get("caf_lags", DEFAULT_CAF_LAGS),
            ccaf_lags=c.get("ccaf_lags", DEFAULT_CCAF_LAGS),
            dechirp_rates=c.get("dechirp_rates", DEFAULT_DECHIRP_RATES) if use_dechirp else (),
            dechirp_pool=c.get("dechirp_pool", 2), hist_bins=c.get("hist_bins", 64),
            dcmp_lag=c.get("dcmp_lag", 4), if_smooth=c.get("if_smooth", 5),
            min_symbols=c.get("min_symbols", 4), cyc_whiten=c.get("cyc_whiten", 31),
            fine_cfo_bins=c.get("fine_cfo_bins", 12.0))
        self.tokenizer = TimeTokenizer(
            d, self.patch, stft_win, c.get("stem_width", 64), self.front.n_time_channels,
            use_stft=c.get("use_stft", True), use_stats=c.get("use_stats", True),
            task_prompt=c.get("task_prompt", True), n_stats=self.front.n_stats)
        self.freq_stream = FrequencyStream(
            d, nhead, self.front.view_shapes(),
            tokens={"spec": c.get("spec_tokens", 32), "caf": c.get("caf_tokens", 16),
                    "dechirp": c.get("dechirp_tokens", 16), "acf": c.get("acf_tokens", 16)},
            const_patch=c.get("const_patch", 16),
            use={"spec": c.get("use_spec", True), "acf": c.get("use_acf", True),
                 "const": c.get("use_const", True), "caf": c.get("use_caf", True),
                 "dechirp": use_dechirp, "hist": c.get("use_hist", True)},
            n_layers=c.get("freq_layers", 2), dim_feedforward=ffn, dropout=dropout)
        self.fusion = CrossDomainFusion(d, nhead, dropout=0.0)

        # ---------------- 共享：预训练主干（FA + MLA） ----------------
        layer_types = c.get("layer_types") or c.get("hybrid_layers") or DEFAULT_BACKBONE_LAYERS
        bad = [t for t in layer_types if normalize_layer_type(t) in CONTIGUOUS_ONLY]
        if bad:
            raise ValueError(f"主干 layer_types 不能包含 {bad}：MAE 只编码可见 token（gather），"
                             "卷积类层需要连续序列。Conformer 请配置在 neck_layer_types。")
        self.backbone = HybridEncoder(
            d, c.get("n_layers", 24), nhead, layer_types, dim_feedforward=ffn,
            dropout=dropout, drop_path=c.get("drop_path", 0.1), use_rope=True,
            rope_max_len=rope_max_len, mla_kv_lora_rank=c.get("mla_kv_lora_rank", 128),
            mla_q_lora_rank=c.get("mla_q_lora_rank", 128),
            grad_checkpoint=c.get("grad_checkpoint", False))

        # ---------------- 共享：微调 neck（Conformer） ----------------
        n_neck = int(c.get("neck_layers", 2))
        self.neck_with_freq = bool(c.get("neck_with_freq", True)) and self.freq_stream.enabled
        self.neck = None
        if n_neck > 0:
            self.neck = HybridEncoder(
                d, n_neck, nhead, c.get("neck_layer_types", ["conformer"]),
                dim_feedforward=c.get("neck_ffn_hidden", ffn), dropout=dropout,
                drop_path=c.get("neck_drop_path", 0.1), use_rope=False,
                conv_kernel=c.get("conv_kernel", 15),
                mla_kv_lora_rank=c.get("mla_kv_lora_rank", 128),
                mla_q_lora_rank=c.get("mla_q_lora_rank", 128),
                grad_checkpoint=c.get("grad_checkpoint", False))
            if self.neck_with_freq:
                self.neck_freq_type = nn.Parameter(torch.zeros(1, 1, d))

        # ---------------- 任务私有头 ----------------
        hc = c.get("head_config", {}) or {}
        ph = c.get("pool_heads", 4)

        def dims(t, h, l):
            x = hc.get(t, {}) or {}
            return x.get("hidden", h), x.get("layers", l)

        hp, lp = dims("protocol", 512, 1)
        hm, lm = dims("modulation", 768, 2)
        hq, lq = dims("parameter", 768, 2)
        pspec = c.get("param_head_spec") or {"n_reg": 7, "cls_sizes": []}
        self.task_heads = nn.ModuleDict({
            "protocol": ClsHead(d, N_PROTOCOL, hp, lp, pool_heads=ph),
            "modulation": ClsHead(d, N_MODULATION, hm, lm, pool_heads=ph,
                                  n_aux=N_FAMILY if c.get("mod_family_aux", True) else 0,
                                  proj_dim=c.get("supcon_dim", 0)),
            "parameter": ParamHead(d, pspec["n_reg"], pspec["cls_sizes"], hq, lq,
                                   pool_heads=ph),
        })

        # ---------------- 仅预训练：MAE 解码器 ----------------
        self.mask_ratio = c.get("mask_ratio", 0.75)
        self.stft_loss_weight = c.get("stft_loss_weight", 0.5)
        self.mae_decoder = None
        if with_decoder:
            self.mae_decoder = MAEDecoder(d, c.get("mae_decoder_dim", 256),
                                          c.get("mae_decoder_layers", 2),
                                          2 * self.patch + stft_win, rope_max_len)
        self.codec = None            # ParamCodec，由训练/推理脚本注入
        self.register_buffer("mod_family", torch.tensor(MOD_FAMILY), persistent=False)
        self._init_weights()

    # ------------------------------------------------------------- 初始化
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if hasattr(self.tokenizer, "fuse"):          # 门控融合保持“偏向时域”的专门初始化
            self.tokenizer.fuse.reset_gate()

    # ------------------------------------------------------------- 参数核算
    def _private_ids(self):
        return {id(p) for p in self.tokenizer.private_parameters()}

    def shared_parameter_count(self):
        priv = self._private_ids()
        mods = [self.tokenizer, self.freq_stream, self.fusion, self.backbone]
        n = sum(p.numel() for m in mods for p in m.parameters() if id(p) not in priv)
        if self.neck is not None:
            n += sum(p.numel() for p in self.neck.parameters())
            if self.neck_with_freq:
                n += self.neck_freq_type.numel()
        return n

    def head_parameter_count(self):
        n = sum(p.numel() for p in self.task_heads.parameters())
        return n + sum(p.numel() for p in self.tokenizer.private_parameters())

    def shared_ratio(self):
        s, h = self.shared_parameter_count(), self.head_parameter_count()
        return s / (s + h)

    def deploy_parameter_count(self):
        return self.shared_parameter_count() + self.head_parameter_count()

    def deploy_state_dict(self):
        """导出/推理用权重（去掉仅预训练使用的 MAE 解码器）。"""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("mae_decoder.")}

    # ------------------------------------------------------------- 工具
    def _token_valid(self, fe):
        """[B,P] bool：每个时域 patch 是否在有效长度内。"""
        B = fe["valid"].size(0)
        return fe["valid"].view(B, -1, self.patch)[:, :, 0] > 0

    def _n_time(self, tok_valid):
        """本 batch 需要保留的时域 token 数（trim_padding：裁到批内最长有效长度）。"""
        if not self.trim_padding:
            return tok_valid.size(1)
        return max(1, int(tok_valid.sum(1).max().item()))

    @staticmethod
    def _prefix_pad(time_pad, n_prefix=N_PREFIX):
        pad = torch.cat([time_pad.new_zeros(time_pad.size(0), n_prefix), time_pad], dim=1)
        return pad if bool(pad.any()) else None

    # ------------------------------------------------------------- 前向
    def encode(self, iq, task=None):
        """共享编码。Returns (ctx [B,L,D], key_padding_mask [B,L] 或 None)。"""
        fe = self.front(iq)
        prefix, time_tok = self.tokenizer(fe, task)
        tok_valid = self._token_valid(fe)
        T = self._n_time(tok_valid)
        x = torch.cat([prefix, time_tok[:, :T].to(prefix.dtype)], dim=1)
        pad = self._prefix_pad(~tok_valid[:, :T])
        freq = self.freq_stream(fe)
        x = self.fusion(x, freq)
        ctx = self.backbone(x, key_padding_mask=pad)
        if self.neck is not None:
            time_slice = (N_PREFIX, N_PREFIX + T)
            if self.neck_with_freq and freq is not None:
                ctx = torch.cat([ctx, (freq + self.neck_freq_type).to(ctx.dtype)], dim=1)
                if pad is not None:
                    pad = torch.cat([pad, pad.new_zeros(pad.size(0), freq.size(1))], dim=1)
            ctx = self.neck(ctx, key_padding_mask=pad, time_slice=time_slice)
        return ctx, pad

    def head_outputs(self, iq, task, return_all=False):
        ctx, pad = self.encode(iq, task)
        head = self.task_heads[task]
        if task == "parameter":
            return head(ctx, pad)
        return head(ctx, pad, return_all=return_all)

    def forward(self, iq, task="protocol"):
        """返回 logits（分类）或参数头输出 dict（parameter）。"""
        return self.head_outputs(iq, task)

    @torch.no_grad()
    def predict(self, iq, task):
        """推理：分类 -> 类别下标 [B]；参数 -> SI 原始单位 [B,7]（不适用参数已置 0）。"""
        out = self.head_outputs(iq, task)
        if task != "parameter":
            return out.argmax(-1)
        if self.codec is None:
            raise RuntimeError("parameter 推理需要 model.codec（weights/param_codec.json）")
        si = self.codec.decode(out)
        radar = out["pattern"].argmax(-1).bool()                     # 0=通信 1=雷达
        mask = torch.zeros_like(si)
        mask[:, :3] = (~radar).float()[:, None]
        mask[:, 3:] = radar.float()[:, None]
        return si * mask

    # ------------------------------------------------------------- MAE 预训练
    def forward_pretrain(self, iq):
        """MAE 预训练前向。Returns (loss, {"loss_time": ..., "loss_stft": ...})。"""
        if self.mae_decoder is None:
            raise RuntimeError("预训练需要 EMFoundation(cfg, with_decoder=True)")
        front, P = self.front, self.patch
        patches, stft_t, p_raw, valid = front.pretrain_targets(iq)   # [B,Pn,2P] [B,Pn,W]
        B = iq.size(0)
        tok_valid = valid.view(B, -1, P)[:, :, 0] > 0                  # [B,Pn]
        T = self._n_time(tok_valid)
        tok_valid = tok_valid[:, :T]
        n_valid = tok_valid.sum(1)                                     # [B]

        # ---- 1. 按有效长度逐样本采样可见 patch（padding patch 排在最后，永不可见）----
        k_i = torch.clamp(torch.round(n_valid.float() * (1 - self.mask_ratio)), min=1).long()
        k_i = torch.minimum(k_i, n_valid.clamp_min(1))
        K = int(k_i.max().item())
        scores = torch.rand(B, T, device=iq.device)
        scores = torch.where(tok_valid, scores, torch.full_like(scores, 2.0))
        keep_patch = scores.argsort(dim=1)[:, :K]                      # [B,K] patch 下标
        keep_ok = torch.arange(K, device=iq.device)[None, :] < k_i[:, None]   # [B,K]
        keepmask = torch.zeros(B, T, device=iq.device).scatter_(
            1, keep_patch, keep_ok.float()) > 0.5                      # 真正可见的 patch

        # ---- 2. 被掩码采样点置零后再进前端（频域/统计/STFT 不泄漏被掩码内容）----
        sample_keep = keepmask.repeat_interleave(P, dim=1).float()
        if sample_keep.size(1) < iq.size(1):
            sample_keep = torch.cat([sample_keep, sample_keep.new_zeros(
                B, iq.size(1) - sample_keep.size(1))], dim=1)
        fe = front(iq, sample_keep=sample_keep, power=p_raw)
        prefix, time_tok = self.tokenizer(fe, None)
        time_tok = time_tok[:, :T]
        D = time_tok.size(-1)
        vis = torch.gather(time_tok, 1, keep_patch.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([prefix, vis.to(prefix.dtype)], dim=1)          # [B,2+K,D]
        pos = torch.cat([torch.arange(N_PREFIX, device=iq.device).expand(B, -1),
                         keep_patch + N_PREFIX], dim=1)                # [B,2+K]
        enc_pad = self._prefix_pad(~keep_ok)
        x = self.fusion(x, self.freq_stream(fe))
        enc = self.backbone(x, key_padding_mask=enc_pad, pos_ids=pos)

        # ---- 3. 解码：全长序列，可见位放编码结果，其余放 mask token ----
        keep_tok = torch.cat([keepmask.new_ones(B, N_PREFIX), keepmask], dim=1)  # [B,2+T]
        dec_pad = self._prefix_pad(~tok_valid)
        out = self.mae_decoder(enc, pos, keep_tok, dec_pad)[:, N_PREFIX:].float()
        pred_t, pred_s = out.split([2 * P, stft_t.size(-1)], dim=-1)

        # ---- 4. 只监督“有效且被掩码”的 patch ----
        w = (tok_valid & ~keepmask).float()
        denom = w.sum().clamp_min(1.0)
        lt = (pred_t - patches[:, :T]).pow(2).mean(-1)
        ls = (pred_s - stft_t[:, :T]).pow(2).mean(-1)
        loss_t = (lt * w).sum() / denom
        loss_s = (ls * w).sum() / denom
        loss = loss_t + self.stft_loss_weight * loss_s
        return loss, {"loss_time": loss_t.detach(), "loss_stft": loss_s.detach()}
