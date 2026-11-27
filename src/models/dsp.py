"""物理先验信号前端（无可学习参数，纯 torch 实数运算）

设计目标
--------
1. 把雷达/通信信号处理里"专家知识"做成确定性的前端，给 Transformer 提供多视角输入：
   时域多通道 / 时频 STFT / 非线性谱 + 共轭循环谱 / 非共轭循环自相关(CAF) / 去斜谱 /
   复自相关 / 三通道星座图 / 分布直方图 / 统计量。
2. **全部用实数张量 + 矩阵乘法实现 DFT**（不依赖 torch.fft 与复数 dtype）：
     - 昇腾 NPU 的 Cube 单元对 GEMM 最友好，复数/FFT 算子支持不完整；
     - ONNX / ATC / TensorRT 导出时只有 MatMul，没有 DFT 算子兼容问题；
     - 多路 DFT 一律把"路"并到 batch 维做一次大矩阵乘（[B,C,N]@[N,N]）。
   通过 impl="fft" 可切换回 torch.fft 做数值对照（tests/test_frontend_features.py）。
3. 前端强制 fp32（autocast 关闭），避免 bf16 下频谱/相位精度崩坏。

处理链（forward）
----------------
  iq ─ 有效长度/单位功率归一化 ─ x
     ├─ 功率谱 -> 噪底/占用带宽 -> 粗频偏(整 bin) + 带外置零 ─ xc
     ├─ 非共轭循环自相关 CAF  c_d(α)=|Σ x[n]x*[n-d] e^{-j2παn}| / Σ|x[n]x*[n-d]|
     │     └─ 循环域剖面 CDP(α)=max_d c_d(α) -> 码元速率 Rs（谐波回退 + 抛物线插值）
     ├─ 符号同步：Oerder–Meyr 闭式定时 + 线性插值抽取符号 y_k
     │     └─ M 次方(p=2/4/8)相干度 ρ_p + 精频偏/相位 -> 去旋转（符号域与采样域 xf）
     ├─ 时域通道（xf 上）：瞬时频率、平滑瞬时频率、调频率、延迟共轭积 x[n]x*[n-D]
     ├─ 非线性谱 |X|,|FFT(xc²)|,|FFT(xc⁴)| + 共轭循环谱 |FFT(xc[n]xc[n-d])|
     ├─ 去斜谱 |FFT(xc·e^{-jπ m n²/N})|（m 为一组候选调频率，含 m=0 基准）
     ├─ 包络 ACF + 复自相关 R(τ)（幅度 + 归一化实/虚部，保留相位编码信息）
     ├─ 星座图三通道：全部采样点 / 稳定点加权（抑制符号间过渡点）/ 符号同步后的符号点
     └─ 直方图：平滑瞬时频率 / 延迟相位差 / 符号间相位差 / 采样幅度 / 符号幅度环 / 符号相位

主要输出（SignalFrontEnd.forward 返回 dict）
------------------------------------------
  valid    [B,N]        有效采样点掩码（512 点样本右侧补零部分为 0）
  length   [B]          有效长度（已向上取整到 length_quantum 整数倍）
  x        [B,N,2]      单位功率归一化原始信号（保留载频偏移 -> frequency_shift 可估计）
  xc       [B,N,2]      精频偏校正 + 带外抑制后的信号（xf）
  time_ch  [B,10,N]     时域多通道，见 TIME_CHANNELS
  stft     [B,T,W]      对数 STFT 幅度（帧与 patch 一一对齐，T=N/patch）
  spec     [B,3+|ccaf|,N] 对数谱（fftshift）：|X|,|X²|,|X⁴|,共轭循环谱(每个滞后一路)
  caf      [B,|caf|,N/2] 非共轭循环自相关相干度 ∈[0,1]（α∈[0,0.5)）
  dechirp  [B,J,N/pool] 去斜对数谱（J=候选调频率个数，含 m=0），J=0 时为 None
  acf      [B,4,A]      包络 ACF、|R|/R0、Re R/R0、Im R/R0
  const    [B,3,G,G]    星座密度（log1p），见 CONST_CHANNELS
  hist     [B,6,H]      分布直方图（log1p 归一化），见 HIST_NAMES
  stats    [B,40]       统计量（symlog 压缩），见 STAT_NAMES

参考:
  Swami & Sadler, "Hierarchical digital modulation classification using cumulants", TCOM 2000
  Gardner, "Exploitation of spectral redundancy in cyclostationary signals", SPM 1991
  Oerder & Meyr, "Digital filter and square timing recovery", TCOM 1988
  Levanon & Mozeson, "Radar Signals", Wiley 2004（相位编码/LFM 的自相关与去斜）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- 常量
TIME_CHANNELS = (
    "i", "q",                 # 单位功率原始 I/Q（含载频偏移）
    "i_c", "q_c",             # 精频偏校正 + 带外抑制后的 I/Q
    "amp",                    # 包络
    "ifreq",                  # 瞬时频率 angle(x[n]x*[n-1])/π（= 相邻相位差序列）
    "ifreq_smooth",           # 复域滑动平均后的瞬时频率（FSK 电平 / 相位码跳变更清晰）
    "chirp",                  # 平滑瞬时频率的一阶差分（调频率；LFM 为常数）
    "dcmp_i", "dcmp_q",       # 延迟共轭积 xf[n]·xf*[n-D]（对 CFO 不敏感的相位差视图）
)
CONST_CHANNELS = ("all_samples", "stable_samples", "symbols")
HIST_NAMES = (
    "ifreq_smooth",           # 平滑瞬时频率分布（2FSK 双峰 / 4FSK 四峰 / GFSK 连续）
    "phase_diff_lag",         # 延迟 D 的相位差分布
    "phase_diff_symbol",      # 符号间相位差分布（π/4-DQPSK 的 ±π/4、±3π/4）
    "amp_sample",             # 采样点幅度分布（OOK 双峰、恒包络单峰）
    "amp_symbol",             # 符号幅度环分布（16/32/64/128/256QAM 的环结构）
    "phase_symbol",           # 符号相位分布（去旋转后，PSK 阶数）
)
STAT_NAMES = (
    # ---- 基础 16 维 ----
    "log_power", "length", "cfo", "occupied_bw", "snr", "log_floor",
    "M20", "C40", "C41", "C42", "M63", "papr", "duty", "amp_kurt", "if_std", "amp_var",
    # ---- 循环平稳 / 符号同步 ----
    "sym_rate", "rate_conf", "log2_sps", "log_nsym", "bw_over_rate",
    "rho2_sym", "rho4_sym", "rho8_sym", "sync_gate", "fine_cfo",
    # ---- 符号域高阶累积量 ----
    "C40_sym", "C42_sym", "C60_sym", "C63_sym", "C80_sym", "sym_amp_std",
    # ---- 频率/去斜/复自相关 ----
    "ifs_std", "ifs_kurt", "dechirp_gain", "dechirp_best", "acf_psl", "acf_r1",
    # ---- 采样域 M 次方谱线相干度 ----
    "rho2_smp", "rho4_smp",
)
N_STATS = len(STAT_NAMES)

DEFAULT_CAF_LAGS = (0, 1, 2, 4, 8, 16, 32)
DEFAULT_CCAF_LAGS = (1, 2, 4, 8)
_DC = (0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_DECHIRP_RATES = (0.0,) + tuple(-m for m in reversed(_DC)) + _DC   # 19 路，约 √2 倍间隔
POWERS = (2, 4, 8)


# ----------------------------------------------------------------------------- 小工具
def _symlog(x):
    """sign(x)*log1p(|x|)：把跨数量级的统计量压到 O(1) 区间。"""
    return torch.sign(x) * torch.log1p(x.abs())


def _fftshift(x):
    n = x.shape[-1]
    return torch.roll(x, shifts=n // 2, dims=-1)


def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


def _cpow(r, i, p):
    """复数整数次幂（p ∈ {1,2,4,8}，反复平方）。"""
    while p > 1:
        r, i = _cmul(r, i, r, i)
        p //= 2
    return r, i


def _delay(t, d):
    """沿最后一维右移 d 个采样（左侧补零）：out[n] = t[n-d]。"""
    if d == 0:
        return t
    return F.pad(t, (d, 0))[..., : t.shape[-1]]


def _advance(t, d=1):
    """沿最后一维左移 d 个采样（右侧补零）：out[n] = t[n+d]。"""
    return F.pad(t, (0, d))[..., d:]


def _moving_sum(t, k):
    """[B,N] 居中滑动求和（k 为奇数，两端补零）。"""
    if k <= 1:
        return t
    return F.avg_pool1d(F.pad(t[:, None], (k // 2, k // 2)), k, stride=1)[:, 0] * k


def _soft_index(u, bins, circular=False):
    """u∈[0,1) -> 线性插值分箱 (i0, i1, w0, w1)。箱中心在 (i+0.5)/bins；
    circular=True 时首尾相邻（相位直方图 -π 与 +π 连续），否则边缘箱吸收越界部分。
    软分箱让直方图对输入连续：浮点舍入不会让样本在相邻箱之间"跳变"。"""
    pos = u * bins - 0.5
    i0f = pos.floor()
    f = pos - i0f
    i0 = i0f.long()
    i1 = i0 + 1
    if circular:
        i0, i1 = i0 % bins, i1 % bins
    else:
        i0, i1 = i0.clamp(0, bins - 1), i1.clamp(0, bins - 1)
    return i0, i1, 1.0 - f, f


def _hist1d(v, w, lo, hi, bins, circular=False):
    """加权 1D 软直方图 -> 概率分布 [B,bins]；越界样本并入边缘箱（保持连续）。v/w: [B,M]。"""
    u = (v - lo) / (hi - lo)
    i0, i1, w0, w1 = _soft_index(u.clamp(0, 1 - 1e-6), bins, circular)
    h = torch.zeros(v.size(0), bins, device=v.device, dtype=w.dtype)
    h.scatter_add_(1, i0, w * w0)
    h.scatter_add_(1, i1, w * w1)
    return h / h.sum(-1, keepdim=True).clamp_min(1e-12)


def _hist2d(xr, xi, w, R, G):
    """加权 2D 双线性软直方图（[-R,R]²，G×G）-> log1p(密度×G²)，[B,G,G]；越界样本并入边缘箱。"""
    ur, ui = (xr + R) / (2 * R), (xi + R) / (2 * R)
    x0, x1, wx0, wx1 = _soft_index(ur.clamp(0, 1 - 1e-6), G)
    y0, y1, wy0, wy1 = _soft_index(ui.clamp(0, 1 - 1e-6), G)
    dens = torch.zeros(xr.size(0), G * G, device=xr.device, dtype=w.dtype)
    for yy, wy in ((y0, wy0), (y1, wy1)):
        for xx, wx in ((x0, wx0), (x1, wx1)):
            dens.scatter_add_(1, yy * G + xx, w * wy * wx)
    dens = dens / dens.sum(-1, keepdim=True).clamp_min(1e-12) * (G * G)
    return torch.log1p(dens).view(-1, G, G)


def _gather1(t, idx):
    """t [B,M], idx [B] -> [B]"""
    return torch.gather(t, 1, idx[:, None])[:, 0]


def valid_mask_from_iq(iq, quantum=512):
    """由右侧补零推断有效长度（官方 512 点样本被补零到 1024）。

    有效长度向上取整到 quantum 的整数倍（默认 512 -> 只可能是 512 / 1024），
    与数据侧长度分桶（dataset.PreparedOfficialDataset.lengths）规则完全一致；
    避免脉冲雷达等"尾部恰好为 0"的 1024 点样本被截成奇怪长度。

    Returns:
        valid [B,N] float, length [B] long（quantum 的整数倍，>= quantum）
    """
    B, N, _ = iq.shape
    quantum = max(1, min(int(quantum), N))
    nz = (iq.abs().sum(-1) > 0).float()                         # [B,N]
    idx = torch.arange(1, N + 1, device=iq.device, dtype=torch.float32)
    last = (nz * idx).amax(-1)                                  # 最后一个非零点下标+1
    length = torch.ceil(last / quantum) * quantum
    length = length.clamp(min=quantum, max=N).long()
    valid = (torch.arange(N, device=iq.device)[None, :] < length[:, None]).float()
    return valid, length


class RealDFT(nn.Module):
    """实数矩阵乘实现的 DFT / IDFT（NPU / ONNX 友好）。

    X[k] = Σ_n x_n e^{-j2πkn/N}:
      Xr = xr@C + xi@S ;  Xi = xi@C - xr@S   (C[n,k]=cos, S[n,k]=sin)
    n_out < n 时只计算前 n_out 个频点（例如只要 [0, N/2) 的循环频率，矩阵乘减半）。
    impl="fft" 时改用 torch.fft（仅用于 CUDA 对照/调试）。
    """

    def __init__(self, n, impl="matmul", n_out=None):
        super().__init__()
        self.n = n
        self.n_out = n if n_out is None else int(n_out)
        self.impl = impl
        k_in = torch.arange(n, dtype=torch.int64)
        k_out = torch.arange(self.n_out, dtype=torch.int64)
        # 先对 n 取模再乘 2π/n：大 N 时角度精确（避免 float 大角度的 cos 误差）
        ang = 2 * math.pi * (torch.outer(k_in, k_out) % n).double() / n
        self.register_buffer("C", torch.cos(ang).float(), persistent=False)
        self.register_buffer("S", torch.sin(ang).float(), persistent=False)

    def forward(self, xr, xi, inverse=False):
        """xr/xi: [..., n] -> (Xr, Xi) [..., n_out]；inverse=True 时含 1/n 归一化（需 n_out==n）。"""
        if inverse and self.n_out != self.n:
            raise ValueError("截断输出的 RealDFT 不支持逆变换")
        if self.impl == "fft":
            z = torch.complex(xr.float(), xi.float())
            Z = torch.fft.ifft(z, dim=-1) if inverse else torch.fft.fft(z, dim=-1)
            return Z.real[..., : self.n_out], Z.imag[..., : self.n_out]
        C, S = self.C, self.S
        if inverse:   # e^{+j}: Xr = xr C - xi S ; Xi = xi C + xr S
            return (xr @ C - xi @ S) / self.n, (xi @ C + xr @ S) / self.n
        return xr @ C + xi @ S, xi @ C - xr @ S


# ============================================================================= 前端
class SignalFrontEnd(nn.Module):
    """多视角确定性信号前端（见模块文档）。全部运算在 fp32 下执行，无可学习参数。

    Args:
        seq_len / patch_size / stft_win: 序列长度、patch 大小、STFT 窗长（帧移 = patch）
        acf_lags:      自相关保留的滞后数 A
        const_bins / const_range: 星座图 G×G 网格与范围 [-R,R]
        caf_lags:      非共轭循环自相关的滞后集合（含 0 即 |x|² 的循环谱）
        ccaf_lags:     共轭循环谱的滞后集合（x[n]x[n-d]，叠加到 spec 通道）
        dechirp_rates: 去斜候选调频率 m（瞬时频率在 N 点内扫过 m 个 cycles/sample；
                       自动保证 m=0 为第一个通道；空列表 = 关闭去斜）
        dechirp_pool:  去斜谱沿频率轴的最大池化倍数（峰值保持）
        hist_bins:     直方图分箱数
        dcmp_lag:      延迟共轭积 / 相位差直方图的滞后 D
        if_smooth:     平滑瞬时频率的复域滑动窗（奇数）
        min_symbols:   估计码元速率时，最少要求的观测符号数（限制 sps 上界 = L/min_symbols）
        cyc_whiten:    码元速率估计时循环谱局部噪底白化窗长（bin，奇数）
        fine_cfo_bins: 符号域精频偏搜索范围（粗校正后残余频偏的上界，单位 DFT bin；另封顶 0.25 cycles/symbol）
    """

    def __init__(self, seq_len=1024, patch_size=8, stft_win=32, acf_lags=512,
                 const_bins=64, const_range=2.5, floor_ratio=4.0, dft_impl="matmul",
                 length_quantum=512, caf_lags=DEFAULT_CAF_LAGS, ccaf_lags=DEFAULT_CCAF_LAGS,
                 dechirp_rates=DEFAULT_DECHIRP_RATES, dechirp_pool=2, hist_bins=64,
                 dcmp_lag=4, if_smooth=5, min_symbols=4, cyc_whiten=31, fine_cfo_bins=12.0):
        super().__init__()
        assert seq_len % patch_size == 0
        assert stft_win >= patch_size and (stft_win - patch_size) % 2 == 0
        assert seq_len % 2 == 0
        self.N = seq_len
        self.patch = patch_size
        self.quantum = min(int(length_quantum), seq_len)
        assert self.quantum % patch_size == 0, "length_quantum 必须是 patch_size 的整数倍"
        self.win = stft_win
        self.acf_lags = int(acf_lags)
        assert 2 <= self.acf_lags <= seq_len
        self.G = const_bins
        self.R = const_range
        self.floor_ratio = floor_ratio
        self.caf_lags = tuple(int(d) for d in caf_lags)
        self.ccaf_lags = tuple(int(d) for d in ccaf_lags)
        assert len(self.caf_lags) > 0, "caf_lags 不能为空（码元速率估计依赖它）"
        assert all(0 <= d < seq_len // 2 for d in self.caf_lags + self.ccaf_lags)
        rates = [float(m) for m in dechirp_rates]
        self.dechirp_rates = tuple([0.0] + [m for m in rates if m != 0.0]) if rates else ()
        self.dechirp_pool = int(dechirp_pool)
        assert seq_len % self.dechirp_pool == 0
        self.hist_bins = int(hist_bins)
        self.dcmp_lag = int(dcmp_lag)
        self.if_smooth = int(if_smooth) | 1                       # 强制奇数
        self.min_symbols = float(min_symbols)
        self.cyc_whiten = int(cyc_whiten) | 1                     # 循环谱局部白化窗（奇数）
        self.fine_cfo_bins = float(fine_cfo_bins)

        self.dft = RealDFT(seq_len, dft_impl)
        self.dft_half = RealDFT(seq_len, dft_impl, n_out=seq_len // 2)   # 循环频率 α∈[0,0.5)
        self.dft_acf = RealDFT(2 * seq_len, dft_impl)             # 线性自相关需 2N 点
        self.dft_win = RealDFT(stft_win, dft_impl)
        n = torch.arange(stft_win, dtype=torch.float64)
        self.register_buffer("hann", (0.5 - 0.5 * torch.cos(2 * math.pi * n / stft_win)).float(),
                             persistent=False)
        f = (torch.arange(seq_len, dtype=torch.float32) - seq_len // 2) / seq_len
        self.register_buffer("freq_axis", f, persistent=False)   # fftshift 后的归一化频率
        self.register_buffer("n_idx", torch.arange(seq_len, dtype=torch.float32), persistent=False)
        if self.dechirp_rates:
            nn_ = torch.arange(seq_len, dtype=torch.float64)
            m = torch.tensor(self.dechirp_rates, dtype=torch.float64)[:, None]
            # φ[n] = π m n² / N，先对 2π 取模保证精度
            phi = torch.remainder(math.pi * m * nn_[None] ** 2 / seq_len, 2 * math.pi)
            self.register_buffer("chirp_c", torch.cos(phi).float(), persistent=False)
            self.register_buffer("chirp_s", torch.sin(phi).float(), persistent=False)
        self.register_buffer("power_bias", torch.tensor([1.0, 0.97, 0.94]), persistent=False)

    # ------------------------------------------------------------------ 形状信息
    @property
    def n_time_channels(self):
        return len(TIME_CHANNELS)

    @property
    def n_stats(self):
        return N_STATS

    def n_tokens(self):
        return self.N // self.patch

    def view_shapes(self):
        """频域流各视图的 (通道数, 轴长)；const 为 (通道数, G)，dechirp 关闭时为 None。"""
        return {
            "spec": (3 + len(self.ccaf_lags), self.N),
            "caf": (len(self.caf_lags), self.N // 2),
            "dechirp": (len(self.dechirp_rates), self.N // self.dechirp_pool)
            if self.dechirp_rates else None,
            "acf": (4, self.acf_lags),
            "const": (len(CONST_CHANNELS), self.G),
            "hist": (len(HIST_NAMES), self.hist_bins),
        }

    # ------------------------------------------------------------------ 基础视图
    def _normalize(self, iq, valid, power=None):
        """单位功率归一化（按有效长度），返回 x [B,N,2]、原始功率 p [B]。"""
        L = valid.sum(-1).clamp_min(1.0)
        p_raw = ((iq ** 2).sum(-1) * valid).sum(-1) / L
        p = p_raw if power is None else power
        x = iq / torch.sqrt(p + 1e-20)[:, None, None]
        return x * valid[..., None], p_raw

    def _spectrum(self, xr, xi, L):
        """|X|^2/L 的 fftshift 功率谱 + 平滑 + 噪底 + 占用掩码。"""
        Xr, Xi = self.dft(xr, xi)
        psd = (Xr ** 2 + Xi ** 2) / L[:, None]
        psd_s = _fftshift(psd)                                   # [B,N] 频率 -N/2..N/2-1
        k = 9
        sm = F.avg_pool1d(F.pad(psd_s[:, None], (k // 2, k // 2), mode="circular"),
                          k, stride=1)[:, 0]
        # 噪底 = 最低 50% 频点的均值（topk 可导出 ONNX，避免 median）
        low = torch.topk(sm, self.N // 2, dim=-1, largest=False).values
        floor = low.mean(-1).clamp_min(1e-12)                    # [B]
        occ = (sm > floor[:, None] * self.floor_ratio).float()
        # 膨胀一下避免截断带边
        occ = F.max_pool1d(F.pad(occ[:, None], (4, 4), mode="circular"), 9, stride=1)[:, 0]
        none = (occ.sum(-1, keepdim=True) == 0).float()
        occ = (occ + none).clamp(max=1.0)                        # 全空时退化为全通
        return Xr, Xi, psd_s, sm, floor, occ

    def _carrier_correct(self, Xr, Xi, sm, floor, occ, valid):
        """频域加权质心估计 CFO -> 频域整数 bin 平移 + 带外置零 -> IDFT。"""
        w = F.relu(sm - floor[:, None]) * occ
        cfo = (w * self.freq_axis[None]).sum(-1) / w.sum(-1).clamp_min(1e-12)   # cycles/sample
        shift = torch.round(cfo * self.N).long()                                  # [B]
        Xr_s, Xi_s = _fftshift(Xr) * occ, _fftshift(Xi) * occ
        idx = (torch.arange(self.N, device=Xr.device)[None, :] + shift[:, None]) % self.N
        Xr_s = torch.gather(Xr_s, 1, idx)
        Xi_s = torch.gather(Xi_s, 1, idx)
        # 反 fftshift（N 偶数时 shift N/2 自逆）后 IDFT
        xr, xi = self.dft(_fftshift(Xr_s), _fftshift(Xi_s), inverse=True)
        xr, xi = xr * valid, xi * valid
        L = valid.sum(-1).clamp_min(1.0)
        p = ((xr ** 2 + xi ** 2) * valid).sum(-1) / L
        s = torch.rsqrt(p + 1e-12)[:, None]
        return xr * s, xi * s, cfo

    def _stft(self, xr, xi):
        """帧与 patch 对齐的对数 STFT：[B, N/patch, win]（fftshift）。"""
        pad = (self.win - self.patch) // 2
        fr = F.pad(xr, (pad, pad)).unfold(-1, self.win, self.patch) * self.hann
        fi = F.pad(xi, (pad, pad)).unfold(-1, self.win, self.patch) * self.hann
        Sr, Si = self.dft_win(fr, fi)
        return _fftshift(torch.log1p(Sr ** 2 + Si ** 2))

    def _logspec(self, yr, yi, mask):
        """对数归一化谱（fftshift）与谱线相干度。

        yr/yi/mask: [..., N]。返回 logP [..., N] = log1p(|Y|²/Σ|y|²)，
        rho [...] = max|Y| / Σ|y| ∈ [0,1]（纯谱线 -> 1，噪声 -> ~1/√N）。
        """
        yr, yi = yr * mask, yi * mask
        Yr, Yi = self.dft(yr, yi)
        pw = (yr ** 2 + yi ** 2).sum(-1, keepdim=True)
        P = (Yr ** 2 + Yi ** 2) / pw.clamp_min(1e-12)
        mag_sum = torch.sqrt(yr ** 2 + yi ** 2 + 1e-20).mul(mask).sum(-1)
        rho = torch.sqrt(P.amax(-1) * pw[..., 0]) / mag_sum.clamp_min(1e-12)
        return _fftshift(torch.log1p(P)), rho

    # ------------------------------------------------------------------ 循环平稳
    def _cyclic(self, xr, xi, valid):
        """非共轭循环自相关相干度 c_d(α)，[B,|caf_lags|,N/2]，α=k/N∈[0,0.5)；
        以及各滞后的时域相干度 coh_d = |Σp_d|/Σ|p_d| [B,|caf_lags|]（≈|R(d)|/R(0)）。

        p_d[n] = x[n]x*[n-d]（去均值后 DFT），c_d(α) = |P_d(α)| / Σ|p_d|。
        线性调制在 α = k·Rs 处有谱线（与载频、定时、初相无关），雷达脉冲串在 1/PRI 处。
        """
        prs, pis, ms = [], [], []
        for d in self.caf_lags:
            sr, si, sv = _delay(xr, d), _delay(xi, d), _delay(valid, d)
            m = valid * sv
            prs.append((xr * sr + xi * si) * m)
            pis.append((xi * sr - xr * si) * m)
            ms.append(m)
        pr, pi_, m = torch.stack(prs, 1), torch.stack(pis, 1), torch.stack(ms, 1)   # [B,C,N]
        cnt = m.sum(-1, keepdim=True).clamp_min(1.0)
        absum = torch.sqrt(pr ** 2 + pi_ ** 2 + 1e-20).mul(m).sum(-1, keepdim=True)
        sr, si = pr.sum(-1, keepdim=True), pi_.sum(-1, keepdim=True)
        coh = (torch.sqrt(sr ** 2 + si ** 2 + 1e-20) / absum.clamp_min(1e-12))[..., 0]  # |R(d)|/Σ|p|
        pr = (pr - sr / cnt) * m
        pi_ = (pi_ - si / cnt) * m
        Zr, Zi = self.dft_half(pr, pi_)
        return torch.sqrt(Zr ** 2 + Zi ** 2 + 1e-20) / absum.clamp_min(1e-12), coh

    def _symbol_rate(self, caf, coh, L):
        """白化循环域剖面上找码元速率（cycles/sample）。

        CDP(α) = mean_{相干 d} c_d(α) / 局部均值_d(α)：
          * 逐滞后局部噪底白化——谱线只占 1~2 个 bin，而乘积的随机连续谱较宽；
          * 只用"相干"滞后（|R(d)|/R(0) > 0.2）——d 超过脉冲相关长度后 p_d 是独立符号
            之积，只有随机连续谱、没有谱线；
          * 跨滞后平均——真实谱线在所有相干滞后的同一 α 上出现（相干累加），随机峰不对齐。
        搜索区间 α ∈ [4/L, 0.5)（低于 4/L 受有限长窗泄漏影响，局部均值也只在区间内统计）；
        最高峰若在半频处也有 ≥80% 的峰，则回退到半频（防止锁到 2Rs 谐波）；抛物线插值到亚 bin。
        Returns: rate [B], conf [B]（白化峰值 / 带内均值 的对数）
        """
        B, C, H = caf.shape
        k = torch.arange(H, device=caf.device)
        kmin = torch.ceil(4.0 * self.N / L).clamp(min=2).long()      # [B]
        band = (k[None] >= kmin[:, None]) & (k[None] <= H - 2)
        bandf = band.to(caf.dtype)
        W = self.cyc_whiten
        pool = lambda t: F.avg_pool1d(F.pad(t, (W // 2, W // 2)), W, stride=1)
        num = pool((caf * bandf[:, None]).reshape(B * C, 1, H)).view(B, C, H)
        den = pool(bandf[:, None]).view(B, 1, H)
        loc = num / den.clamp_min(1e-6)
        use = (coh > 0.2).to(caf.dtype)
        use[:, 0] = 1.0
        cdp = (caf / (loc + 1e-4) * use[..., None]).sum(1) / use.sum(1, keepdim=True)   # [B,H]
        kstar = torch.where(band, cdp, torch.full_like(cdp, -1.0)).argmax(-1)
        for _ in range(2):
            kh = torch.div(kstar + 1, 2, rounding_mode="floor")
            ok = (kh >= kmin) & (_gather1(cdp, kh) >= 0.8 * _gather1(cdp, kstar))
            kstar = torch.where(ok, kh, kstar)
        a = _gather1(cdp, (kstar - 1).clamp(min=0))
        b = _gather1(cdp, kstar)
        c = _gather1(cdp, (kstar + 1).clamp(max=H - 1))
        dd = a - 2 * b + c
        safe = dd.abs() > 1e-9
        delta = torch.where(safe, 0.5 * (a - c) / torch.where(safe, dd, torch.ones_like(dd)),
                            torch.zeros_like(dd)).clamp(-0.5, 0.5)
        rate = (kstar.float() + delta) / self.N
        mean_band = (cdp * bandf).sum(-1) / bandf.sum(-1).clamp_min(1.0)
        conf = torch.log(b / (mean_band + 1e-6) + 1e-6)
        return rate, conf

    def _symbol_sync(self, xcr, xci, valid, L, rate):
        """符号同步：Oerder–Meyr 定时 + 线性插值抽取 + M 次方精频偏/相位去旋转。

        Returns dict:
          yr, yi, ym [B,N]  符号（单位功率、已去旋转）与有效掩码（第 k 个位置 = 第 k 个符号）
          ks [B] 符号数, sps [B], tau [B], rho [B,3]（p=2/4/8 的谱线相干度）
          fr [B] 残余频偏（cycles/symbol）, theta [B] 相位, gate [B] 是否应用了去旋转
        """
        N = self.N
        n = self.n_idx
        sps = (1.0 / rate.clamp_min(1e-6)).clamp(min=1.0)
        sps = torch.minimum(sps, L / self.min_symbols)
        # Oerder–Meyr：τ = -sps/(2π)·arg Σ|x[n]|² e^{-j2πn/sps}（符号中心处 |x|² 最大）
        e = (xcr ** 2 + xci ** 2) * valid
        ph = 2 * math.pi * n[None] / sps[:, None]
        s_r = (e * torch.cos(ph)).sum(-1)
        s_i = -(e * torch.sin(ph)).sum(-1)
        tau = torch.remainder(-sps * torch.atan2(s_i, s_r) / (2 * math.pi), sps)
        t = tau[:, None] + n[None] * sps[:, None]                     # 第 k 个符号的时刻
        ym = (t < (L - 1.0)[:, None]).float()
        t = t.clamp(0.0, N - 1.001)
        i0 = t.floor().long()
        fr_ = t - i0.float()
        i1 = (i0 + 1).clamp(max=N - 1)
        yr = (torch.gather(xcr, 1, i0) * (1 - fr_) + torch.gather(xcr, 1, i1) * fr_) * ym
        yi = (torch.gather(xci, 1, i0) * (1 - fr_) + torch.gather(xci, 1, i1) * fr_) * ym
        ks = ym.sum(-1).clamp_min(1.0)
        s = torch.rsqrt((yr ** 2 + yi ** 2).sum(-1) / ks + 1e-12)[:, None]
        yr, yi = yr * s, yi * s

        # M 次方相干度：ρ_p = max_f |Σ y^p e^{-j2πfk}| / Σ|y|^p，同时给出频偏与相位
        ay = torch.sqrt(yr ** 2 + yi ** 2 + 1e-20) * ym
        rhos, fs, phis = [], [], []
        # 粗校正（谱质心）残余频偏实测可达 ±6 bin 以上（周期图起伏使质心有偏），因此精搜索
        # 覆盖 ±fine_cfo_bins 个 bin，换算到 p 次方的符号域频率 |f| ≤ p·bins·sps/N；
        # 同时封顶 |f| ≤ 0.25 cycles/symbol：π/4-DQPSK 的 y⁴ 在 0.5 cycles/symbol 有一根"交替"
        # 谱线，不封顶会把 π/4 交替旋转当成频偏去掉，使其退化成 QPSK
        f_ax = ((torch.arange(N, device=yr.device) + N // 2) % N - N // 2).float() / N   # [N]
        for p in POWERS:
            zr, zi = _cpow(yr, yi, p)
            Zr, Zi = self.dft(zr, zi)
            fmax = (p * self.fine_cfo_bins * sps / N).clamp(min=1.0 / N, max=0.25)    # [B]
            inside = (f_ax[None].abs() <= fmax[:, None]).to(yr.dtype)
            mag = torch.sqrt(Zr ** 2 + Zi ** 2 + 1e-20) * inside
            pk, kp = mag.max(-1)
            rhos.append(pk / ((ay ** p).sum(-1) + 1e-6))
            phis.append(torch.atan2(_gather1(Zi, kp), _gather1(Zr, kp)))
            fs.append(f_ax[kp])
        rho = torch.stack(rhos, -1)                                   # [B,3]
        best = (rho * self.power_bias).argmax(-1)
        pw = torch.tensor(POWERS, device=yr.device, dtype=yr.dtype)[best]
        f_sel = _gather1(torch.stack(fs, -1), best)
        phi_sel = _gather1(torch.stack(phis, -1), best)
        # p=4 时补 π：方形 QAM / QPSK 的 E[y⁴] 为负实数，补偿后 QAM 网格与坐标轴对齐
        theta = (phi_sel - math.pi * (pw == 4).to(yr.dtype)) / pw
        f_sym = f_sel / pw
        gate = (rho.amax(-1) > 3.0 / torch.sqrt(ks)).to(yr.dtype)     # 谱线显著才去旋转
        theta, f_sym = theta * gate, f_sym * gate
        ang = 2 * math.pi * f_sym[:, None] * n[None] + theta[:, None]
        c, sn = torch.cos(ang), torch.sin(ang)
        yr, yi = (yr * c + yi * sn) * ym, (yi * c - yr * sn) * ym
        return {"yr": yr, "yi": yi, "ym": ym, "ks": ks, "sps": sps, "tau": tau, "rho": rho,
                "fr": f_sym, "theta": theta, "gate": gate}

    def _fine_correct(self, xcr, xci, valid, sym):
        """把符号域估计的残余频偏/相位换算到采样域：xf[n] = xc[n]·e^{-j(2π fr (n-τ)/sps + θ)}。"""
        ang = (2 * math.pi * sym["fr"][:, None] * (self.n_idx[None] - sym["tau"][:, None])
               / sym["sps"][:, None] + sym["theta"][:, None])
        c, s = torch.cos(ang), torch.sin(ang)
        return (xcr * c + xci * s) * valid, (xci * c - xcr * s) * valid

    # ------------------------------------------------------------------ 其余视图
    def _acf(self, amp, xfr, xfi, valid, L):
        """包络 ACF（老视图）+ 复自相关 R(τ)=Σ x[n+τ]x*[n]（保留相位编码），[B,4,A]。"""
        A = self.acf_lags
        e = (amp - (amp * valid).sum(-1, keepdim=True) / L[:, None]) * valid
        zr = torch.stack([e, xfr * valid], 1)                         # [B,2,N]
        zi = torch.stack([torch.zeros_like(e), xfi * valid], 1)
        zr, zi = F.pad(zr, (0, self.N)), F.pad(zi, (0, self.N))      # 2N 零填充 -> 线性相关
        Er, Ei = self.dft_acf(zr, zi)
        P = Er ** 2 + Ei ** 2
        Rr, Ri = self.dft_acf(P, torch.zeros_like(P), inverse=True)
        env = Rr[:, 0, :A]
        # 恒包络信号（FSK/PSK）包络方差≈0，加相对下限避免除零放大噪声
        env = env / (env[:, :1] + 1e-3 * L[:, None])
        R0 = Rr[:, 1, :1].clamp_min(1e-12)
        cr, ci = Rr[:, 1, :A] / R0, Ri[:, 1, :A] / R0
        cm = torch.sqrt(cr ** 2 + ci ** 2 + 1e-20)
        return torch.stack([env, cm, cr, ci], 1), cm

    def _dechirp(self, xcr, xci, valid):
        """去斜谱：对每个候选调频率 m 乘 e^{-jπ m n²/N} 后做 DFT（离散 chirplet/FrFT 近似）。

        LFM / CSS 在匹配的 m 上能量聚成一根谱线（处理增益 ~ 10lgL），Frank 码（阶梯近似 LFM）
        出现带副峰的聚焦，Barker/多相 Barker 不聚焦。Returns map [B,J,N/pool], gain [B], best [B]。
        """
        c, s = self.chirp_c, self.chirp_s                              # [J,N]
        ar = xcr[:, None] * c + xci[:, None] * s
        ai = xci[:, None] * c - xcr[:, None] * s
        mask = valid[:, None].expand_as(ar)
        logP, _ = self._logspec(ar, ai, mask)                          # [B,J,N]
        peaks = logP.amax(-1)                                          # [B,J]
        best_val, best_j = peaks.max(-1)
        gain = best_val - peaks[:, 0]
        rates = torch.tensor(self.dechirp_rates, device=xcr.device, dtype=xcr.dtype)
        best = rates[best_j] / rates.abs().max().clamp_min(1e-6)
        B, J, N = logP.shape
        m = F.max_pool1d(logP.reshape(B * J, 1, N), self.dechirp_pool, self.dechirp_pool)
        return m.view(B, J, -1), gain, best

    # ------------------------------------------------------------------ 统计量
    @staticmethod
    def _cumulants_sym(yr, yi, ym, ks):
        """符号域高阶累积量（单位功率）：|C40|, C42, |C60|, C63, |C80|。"""
        mean = lambda t: (t * ym).sum(-1) / ks
        y2 = _cmul(yr, yi, yr, yi)
        y4 = _cmul(*y2, *y2)
        y6 = _cmul(*y4, *y2)
        y8 = _cmul(*y4, *y4)
        a2 = yr ** 2 + yi ** 2
        M20 = (mean(y2[0]), mean(y2[1]))
        M40 = (mean(y4[0]), mean(y4[1]))
        M60 = (mean(y6[0]), mean(y6[1]))
        M80 = (mean(y8[0]), mean(y8[1]))
        M42, M63 = mean(a2 ** 2), mean(a2 ** 3)
        M20sq = _cmul(*M20, *M20)
        M20cu = _cmul(*M20sq, *M20)
        M20q = _cmul(*M20sq, *M20sq)
        C40 = (M40[0] - 3 * M20sq[0], M40[1] - 3 * M20sq[1])
        C42 = M42 - (M20[0] ** 2 + M20[1] ** 2) - 2.0
        t = _cmul(*M20, *M40)
        C60 = (M60[0] - 15 * t[0] + 30 * M20cu[0], M60[1] - 15 * t[1] + 30 * M20cu[1])
        C63 = M63 - 9.0 * M42 + 12.0                      # 圆对称近似（M21=1）
        a = _cmul(*M60, *M20)
        b = _cmul(*M40, *M40)
        c = _cmul(*M40, *M20sq)
        C80 = (M80[0] - 28 * a[0] - 35 * b[0] + 420 * c[0] - 630 * M20q[0],
               M80[1] - 28 * a[1] - 35 * b[1] + 420 * c[1] - 630 * M20q[1])
        mag = lambda z: torch.sqrt(z[0] ** 2 + z[1] ** 2 + 1e-20)
        amp = torch.sqrt(a2 + 1e-20)
        a_mu = mean(amp)
        a_std = torch.sqrt(mean((amp - a_mu[:, None]) ** 2) + 1e-12)
        return mag(C40), C42, mag(C60), C63, mag(C80), a_std

    def _base_stats(self, p_raw, length, cfo, occ, sm, floor, xr, xi, xcr, xci, ifreq, valid, L):
        mean = lambda t: (t * valid).sum(-1) / L
        x2r, x2i = _cmul(xcr, xci, xcr, xci)
        a2 = xcr ** 2 + xci ** 2
        M20 = torch.sqrt(mean(x2r) ** 2 + mean(x2i) ** 2)
        x4r, x4i = _cmul(x2r, x2i, x2r, x2i)
        m40r, m40i = mean(x4r), mean(x4i)
        m20r, m20i = mean(x2r), mean(x2i)
        s2r, s2i = _cmul(m20r, m20i, m20r, m20i)
        C40 = torch.sqrt((m40r - 3 * s2r) ** 2 + (m40i - 3 * s2i) ** 2)
        M41r, M41i = mean(x2r * a2), mean(x2i * a2)
        C41 = torch.sqrt((M41r - 3 * m20r) ** 2 + (M41i - 3 * m20i) ** 2)  # M21=1
        M42 = mean(a2 ** 2)
        C42 = M42 - (m20r ** 2 + m20i ** 2) - 2.0
        M63 = mean(a2 ** 3)
        amp = torch.sqrt(xr ** 2 + xi ** 2 + 1e-12)
        pw = xr ** 2 + xi ** 2
        papr = (pw * valid).amax(-1) / mean(pw).clamp_min(1e-12)
        duty = mean((pw > 0.5 * mean(pw)[:, None]).float())
        a_mu = mean(amp)
        a_var = mean((amp - a_mu[:, None]) ** 2)
        kurt = mean((amp - a_mu[:, None]) ** 4) / (a_var ** 2 + 1e-12)
        if_std = torch.sqrt(mean((ifreq - mean(ifreq)[:, None]) ** 2))
        bw = occ.mean(-1)
        sig = ((sm - floor[:, None]).clamp_min(0) * occ).sum(-1)
        snr = 10 * torch.log10(sig / (floor * self.N) + 1e-6)
        return {
            "log_power": torch.log10(p_raw + 1e-20), "length": length.float() / self.N,
            "cfo": cfo * 10, "occupied_bw": bw, "snr": snr / 10, "log_floor": torch.log10(floor),
            "M20": M20, "C40": C40, "C41": C41, "C42": C42, "M63": M63 / 10, "papr": papr / 10,
            "duty": duty, "amp_kurt": kurt / 3, "if_std": if_std, "amp_var": a_var,
        }

    # ------------------------------------------------------------------ 前向
    def forward(self, iq, sample_keep=None, power=None):
        """
        Args:
            iq: [B,N,2] 原始 IQ（右侧补零到 N）
            sample_keep: [B,N] 可选，预训练掩码（0=被掩码采样点），全局视图从掩码后信号计算，
                         防止 MAE 通过全局频谱"偷看"被掩码内容
            power: [B] 可选，归一化功率覆盖（预训练用整段信号功率，保持重建目标尺度一致）
        """
        with torch.autocast(device_type=iq.device.type, enabled=False):
            iq = iq.float()
            valid, length = valid_mask_from_iq(iq, self.quantum)
            if sample_keep is not None:
                iq = iq * sample_keep[..., None]
            L = valid.sum(-1).clamp_min(1.0)
            x, p_raw = self._normalize(iq, valid, power)
            if power is not None:
                p_raw = power
            xr, xi = x[..., 0], x[..., 1]

            # ---- 1. 谱 / 粗频偏 / 带外抑制 ----
            Xr, Xi, psd_s, sm, floor, occ = self._spectrum(xr, xi, L)
            xcr, xci, cfo = self._carrier_correct(Xr, Xi, sm, floor, occ, valid)
            amp = torch.sqrt(xr ** 2 + xi ** 2 + 1e-12) * valid
            pr = xr[:, 1:] * xr[:, :-1] + xi[:, 1:] * xi[:, :-1]
            pi_ = xi[:, 1:] * xr[:, :-1] - xr[:, 1:] * xi[:, :-1]
            ifreq = F.pad(torch.atan2(pi_, pr + 1e-12) / math.pi, (1, 0)) * valid

            # ---- 2. 循环平稳：码元速率 ----
            caf, lag_coh = self._cyclic(xr, xi, valid)                         # [B,C,N/2]
            rate, rate_conf = self._symbol_rate(caf, lag_coh, L)

            # ---- 3. 符号同步 + 精频偏 ----
            sym = self._symbol_sync(xcr, xci, valid, L, rate)
            xfr, xfi = self._fine_correct(xcr, xci, valid, sym)
            yr, yi, ym, ks = sym["yr"], sym["yi"], sym["ym"], sym["ks"]

            # ---- 4. 时域通道 ----
            vd = valid * _delay(valid, 1)
            p1r = (xfr * _delay(xfr, 1) + xfi * _delay(xfi, 1)) * vd
            p1i = (xfi * _delay(xfr, 1) - xfr * _delay(xfi, 1)) * vd
            ifs = torch.atan2(_moving_sum(p1i, self.if_smooth),
                              _moving_sum(p1r, self.if_smooth) + 1e-12) / math.pi * valid
            chirp = ((_advance(ifs) - _delay(ifs, 1)) * (0.5 * self.N / 8)).clamp(-4, 4) * valid
            D = self.dcmp_lag
            vD = valid * _delay(valid, D)
            dcr = (xfr * _delay(xfr, D) + xfi * _delay(xfi, D)) * vD
            dci = (xfi * _delay(xfr, D) - xfr * _delay(xfi, D)) * vD
            time_ch = torch.stack([xr, xi, xfr, xfi, amp, ifreq, ifs, chirp, dcr, dci], dim=1)
            stft = self._stft(xr, xi)                                          # [B,T,W]

            # ---- 5. 非线性谱 + 共轭循环谱（同一频率轴，堆叠为通道） ----
            x2r, x2i = _cmul(xcr, xci, xcr, xci)
            x4r, x4i = _cmul(x2r, x2i, x2r, x2i)
            nl_r, nl_i, nl_m = [x2r, x4r], [x2i, x4i], [valid, valid]
            for d in self.ccaf_lags:
                sv = valid * _delay(valid, d)
                qr, qi = _cmul(xcr, xci, _delay(xcr, d), _delay(xci, d))
                nl_r.append(qr)
                nl_i.append(qi)
                nl_m.append(sv)
            logs, rho_smp = self._logspec(torch.stack(nl_r, 1), torch.stack(nl_i, 1),
                                          torch.stack(nl_m, 1))                 # [B,2+C,N]
            spec = torch.cat([torch.log1p(psd_s)[:, None], logs], 1)           # [B,3+C,N]

            # ---- 6. 自相关 ----
            acf, acf_mag = self._acf(amp, xfr, xfi, valid, L)

            # ---- 7. 星座图三通道 ----
            dr = _advance(xfr) - _delay(xfr, 1)
            di = _advance(xfi) - _delay(xfi, 1)
            dd = (dr ** 2 + di ** 2) * valid
            stab = torch.exp(-dd / ((dd.sum(-1) / L)[:, None] + 1e-6)) * valid
            const = torch.stack([_hist2d(xfr, xfi, valid, self.R, self.G),
                                 _hist2d(xfr, xfi, stab, self.R, self.G),
                                 _hist2d(yr, yi, ym, self.R, self.G)], 1)       # [B,3,G,G]

            # ---- 8. 直方图 ----
            Hb = self.hist_bins
            wif = (xfr ** 2 + xfi ** 2) * valid
            sw = wif.sum(-1).clamp_min(1e-12)
            if_mu = (wif * ifs).sum(-1) / sw
            if_c = (ifs - if_mu[:, None]) * valid
            if_var = (wif * if_c ** 2).sum(-1) / sw
            if_std_w = torch.sqrt(if_var + 1e-12)
            if_kurt = (wif * if_c ** 4).sum(-1) / sw / (if_var ** 2 + 1e-12)
            ym1 = ym * _delay(ym, 1)
            sdr, sdi = _cmul(yr, yi, _delay(yr, 1), -_delay(yi, 1))
            hists = [
                _hist1d(if_c / (2.5 * if_std_w[:, None]), wif, -1.0, 1.0, Hb),
                _hist1d(torch.atan2(dci, dcr + 1e-12), torch.sqrt(dcr ** 2 + dci ** 2 + 1e-20) * vD,
                        -math.pi, math.pi, Hb, circular=True),
                _hist1d(torch.atan2(sdi, sdr + 1e-12), torch.sqrt(sdr ** 2 + sdi ** 2 + 1e-20) * ym1,
                        -math.pi, math.pi, Hb, circular=True),
                _hist1d(torch.sqrt(xfr ** 2 + xfi ** 2 + 1e-20), valid, 0.0, 3.0, Hb),
                _hist1d(torch.sqrt(yr ** 2 + yi ** 2 + 1e-20), ym, 0.0, self.R, Hb),
                _hist1d(torch.atan2(yi, yr + 1e-12), ym, -math.pi, math.pi, Hb, circular=True),
            ]
            hist = torch.log1p(torch.stack(hists, 1) * Hb)                     # [B,6,H]

            # ---- 9. 去斜谱 ----
            if self.dechirp_rates:
                dechirp, dc_gain, dc_best = self._dechirp(xcr, xci, valid)
            else:
                dechirp, dc_gain, dc_best = None, torch.zeros_like(L), torch.zeros_like(L)

            # ---- 10. 统计量 ----
            st = self._base_stats(p_raw, length, cfo, occ, sm, floor, xr, xi, xcr, xci,
                                  ifreq, valid, L)
            c40s, c42s, c60s, c63s, c80s, a_std = self._cumulants_sym(yr, yi, ym, ks)
            A = acf_mag.size(-1)
            below = (acf_mag < 0.5).float()
            first = torch.where(below.sum(-1) > 0, below.argmax(-1),
                                torch.full_like(L, A).long())
            tau_ax = torch.arange(A, device=iq.device)[None]
            side = (tau_ax >= (2 * first).clamp(min=2)[:, None]).float()
            psl = (acf_mag * side).amax(-1)
            st.update({
                "sym_rate": rate * 10, "rate_conf": rate_conf,
                "log2_sps": torch.log2(sym["sps"]) / 4,
                "log_nsym": torch.log10(L * rate + 1.0),
                "bw_over_rate": st["occupied_bw"] / rate.clamp_min(1.0 / self.N),
                "rho2_sym": sym["rho"][:, 0], "rho4_sym": sym["rho"][:, 1],
                "rho8_sym": sym["rho"][:, 2], "sync_gate": sym["gate"],
                "fine_cfo": sym["fr"] / sym["sps"] * self.N,
                "C40_sym": c40s, "C42_sym": c42s, "C60_sym": c60s, "C63_sym": c63s,
                "C80_sym": c80s, "sym_amp_std": a_std,
                "ifs_std": if_std_w, "ifs_kurt": if_kurt / 3,
                "dechirp_gain": dc_gain, "dechirp_best": dc_best,
                "acf_psl": psl, "acf_r1": acf_mag[:, 1],
                "rho2_smp": rho_smp[:, 0], "rho4_smp": rho_smp[:, 1],
            })
            stats = _symlog(torch.stack([st[k] for k in STAT_NAMES], dim=-1))
            stats = torch.nan_to_num(stats, nan=0.0, posinf=10.0, neginf=-10.0)
        return {
            "valid": valid, "length": length, "x": x,
            "xc": torch.stack([xfr, xfi], -1), "time_ch": time_ch, "stft": stft,
            "spec": spec, "caf": caf, "dechirp": dechirp, "acf": acf, "const": const,
            "hist": hist, "stats": stats,
        }

    @torch.no_grad()
    def pretrain_targets(self, iq):
        """MAE/SimMIM 重建目标（整段未掩码信号）：
        patch 目标 [B,T,2*patch]（单位功率归一化 IQ）与对数 STFT [B,T,W]，以及整段功率。"""
        with torch.autocast(device_type=iq.device.type, enabled=False):
            iq = iq.float()
            valid, _ = valid_mask_from_iq(iq, self.quantum)
            x, p_raw = self._normalize(iq, valid)
            B, N, _ = x.shape
            patches = x.reshape(B, N // self.patch, 2 * self.patch)
            stft = self._stft(x[..., 0], x[..., 1])
        return patches, stft, p_raw, valid
