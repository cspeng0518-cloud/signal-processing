"""信号前端（models/dsp.py）物理特征回归测试（CPU，约 20 秒）。

用带 RRC 成形 / 载频偏移 / 定时偏移 / AWGN 的合成信号，校验每个新视图"在物理上确实做对了"，
而不只是形状正确：
  * 码元速率估计（白化循环域剖面）；
  * M 次方相干度能区分 BPSK / QPSK / 8PSK / QAM；
  * 符号同步星座把 16QAM 从"一团"恢复成网格；
  * π/4-DQPSK 的符号相位差集中在 ±π/4、±3π/4（精频偏搜索范围受限的回归）；
  * LFM 去斜聚焦增益显著高于 Barker；
  * 2FSK / 4FSK 的平滑瞬时频率峰度可分；
  * 全零 / 极弱 / 恒定输入、MAE 掩码路径不出 NaN；单样本与混批结果一致；
  * matmul DFT 与 torch.fft 数值一致（含截断输出）。
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.dsp import (HIST_NAMES, STAT_NAMES, RealDFT, SignalFrontEnd)  # noqa: E402

S = list(STAT_NAMES)


# ----------------------------------------------------------------------------- 合成信号
def _rrc_pulse(t, beta):
    t = np.asarray(t, float)
    out = np.empty_like(t)
    for i, ti in np.ndenumerate(t):
        if abs(ti) < 1e-9:
            out[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * ti) - 1) < 1e-6:
            out[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                          + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            out[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) \
                / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return out


def _const(mod):
    if mod == "BPSK":
        return np.array([1, -1], complex)
    if mod == "QPSK":
        return np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    if mod == "8PSK":
        return np.exp(1j * np.arange(8) * np.pi / 4)
    k = int(np.sqrt(int(mod[:-3])))
    g = np.arange(k) - (k - 1) / 2
    c = (g[:, None] + 1j * g[None]).ravel()
    return c / np.sqrt((np.abs(c) ** 2).mean())


def _awgn(x, snr, rng):
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    return x + (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) * np.sqrt(10 ** (-snr / 10) / 2)


def _linear(mod, rng, N=1024, sps=8.0, beta=0.35, cfo=0.0, snr=25, symbols=None):
    nsym = int(N / sps) + 40
    s = symbols if symbols is not None else _const(mod)[rng.integers(0, len(_const(mod)), nsym)]
    t = np.arange(N) + rng.uniform(0, sps) + 20 * sps
    x = np.zeros(N, complex)
    for k in range(nsym):
        m = np.abs(t - k * sps) < 8 * sps
        if m.any():
            x[m] += s[k] * _rrc_pulse((t[m] - k * sps) / sps, beta)
    x = x * np.exp(1j * (2 * np.pi * cfo * np.arange(N) + rng.uniform(0, 2 * np.pi)))
    return _awgn(x, snr, rng)


def _pi4dqpsk(rng, N=1024, sps=8.0, snr=25):
    nsym = int(N / sps) + 40
    d = np.array([1, 3, -1, -3]) * np.pi / 4
    return _linear("QPSK", rng, N, sps, snr=snr,
                   symbols=np.exp(1j * np.cumsum(d[rng.integers(0, 4, nsym)])))


def _fsk(M, rng, N=1024, sps=16, h=1.0, snr=15):
    lv = (2 * rng.integers(0, M, N // sps + 2) - (M - 1)).astype(float)
    f = np.repeat(lv, sps)[:N] * h / (2 * sps)
    return _awgn(np.exp(1j * (2 * np.pi * np.cumsum(f + 0.03) + rng.uniform(0, 6.28))), snr, rng)


def _radar(kind, rng, N=512, snr=0):
    if kind == "LFM":
        n = np.arange(N)
        x = np.exp(1j * np.pi * (0.4 / N) * (n - N / 2) ** 2)
    else:   # Barker-13，每码片 24 点
        x = np.repeat(np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], complex), 24)
        x = np.pad(x, (0, N - len(x)))
    return _awgn(x * np.exp(1j * rng.uniform(0, 6.28)), snr, rng)


def _iq(xs, N=1024):
    out = np.zeros((len(xs), N, 2), np.float32)
    for i, x in enumerate(xs):
        out[i, : len(x), 0], out[i, : len(x), 1] = x.real, x.imag
    return torch.from_numpy(out)


def _stat(out, name):
    v = out["stats"][:, S.index(name)]
    return (torch.sign(v) * torch.expm1(v.abs())).numpy()          # 反 symlog


FE = SignalFrontEnd()


# ----------------------------------------------------------------------------- 测试
def test_real_dft_matches_fft_including_truncated_output():
    torch.manual_seed(0)
    xr, xi = torch.randn(3, 256), torch.randn(3, 256)
    for n_out in (None, 128):
        a = RealDFT(256, "matmul", n_out)(xr, xi)
        b = RealDFT(256, "fft", n_out)(xr, xi)
        assert torch.allclose(a[0], b[0], atol=1e-3) and torch.allclose(a[1], b[1], atol=1e-3)


def test_output_shapes_follow_view_shapes():
    rng = np.random.default_rng(0)
    out = FE(_iq([_linear("QPSK", rng), _radar("LFM", rng)]))
    vs = FE.view_shapes()
    assert out["time_ch"].shape == (2, FE.n_time_channels, 1024)
    assert out["stats"].shape == (2, FE.n_stats)
    for k in ("spec", "caf", "dechirp", "acf", "hist"):
        assert tuple(out[k].shape[1:]) == tuple(vs[k]), k
    assert tuple(out["const"].shape[1:]) == (vs["const"][0], vs["const"][1], vs["const"][1])


def test_symbol_rate_estimation():
    rng = np.random.default_rng(7)
    sps_true = [2.5, 4.0, 8.0, 16.0] * 3
    mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "QPSK"] * 2
    xs = [_linear(m, rng, sps=s, cfo=rng.uniform(-0.08, 0.08)) for m, s in zip(mods, sps_true)]
    rate = _stat(FE(_iq(xs)), "sym_rate") / 10
    rel = np.abs(1 / rate - np.array(sps_true)) / np.array(sps_true)
    assert (rel < 0.02).mean() >= 0.9, rel


def test_mth_power_coherence_identifies_psk_order():
    rng = np.random.default_rng(1)
    xs = [_linear(m, rng, sps=4.0, cfo=0.03) for m in ("BPSK", "QPSK", "8PSK", "16QAM")]
    out = FE(_iq(xs))
    rho = np.stack([_stat(out, k) for k in ("rho2_sym", "rho4_sym", "rho8_sym")], 1)
    assert rho[0, 0] > 0.9                                   # BPSK: y² 谱线
    assert rho[1, 1] > 0.85 and rho[1, 0] < 0.4              # QPSK: y⁴ 有、y² 无
    assert rho[2, 2] > 0.6 and rho[2, 1] < 0.4               # 8PSK: y⁸ 有、y⁴ 无
    assert 0.25 < rho[3, 1] < 0.7                            # 16QAM: y⁴ 部分相干


def test_symbol_constellation_recovers_16qam_grid():
    rng = np.random.default_rng(3)
    out = FE(_iq([_linear("16QAM", rng, sps=8.0, cfo=0.05, snr=30)]))
    c_all, c_sym = out["const"][0, 0], out["const"][0, 2]
    # 符号同步通道的能量集中度应远高于全部采样点通道（16 个簇 vs 一团）
    frac = lambda c: float(torch.topk(c.flatten(), 64).values.sum() / c.sum())
    assert frac(c_sym) > 1.5 * frac(c_all)


def test_pi4dqpsk_symbol_phase_jumps():
    rng = np.random.default_rng(5)
    out = FE(_iq([_pi4dqpsk(rng)]))
    h = out["hist"][0, HIST_NAMES.index("phase_diff_symbol")].numpy()
    centers = -np.pi + (np.arange(len(h)) + 0.5) * 2 * np.pi / len(h)
    near = np.zeros_like(h, dtype=bool)
    for a in (np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4):
        near |= np.abs(np.angle(np.exp(1j * (centers - a)))) < 0.2
    p = np.expm1(h)
    assert p[near].sum() / p.sum() > 0.6


def test_dechirp_focuses_lfm_not_barker():
    rng = np.random.default_rng(2)
    out = FE(_iq([_radar("LFM", rng, snr=10), _radar("Barker", rng, snr=10)]))
    gain = _stat(out, "dechirp_gain")
    assert gain[0] > 0.8 and gain[0] > gain[1] + 0.3
    psl = _stat(out, "acf_psl")
    assert psl[1] < 0.25                                     # Barker-13 自相关旁瓣低


def test_fsk_order_from_if_kurtosis():
    rng = np.random.default_rng(4)
    out = FE(_iq([_fsk(2, rng), _fsk(4, rng)]))
    k = _stat(out, "ifs_kurt") * 3
    assert k[0] < 1.6 < k[1]                                 # 双峰峰度≈1，四电平更高


def test_degenerate_inputs_and_masked_path_are_finite():
    rng = np.random.default_rng(0)
    X = np.zeros((4, 1024, 2), np.float32)
    X[1] = 1e-4 * rng.normal(size=(1024, 2))
    X[2, :, 0] = 1.0
    X[3, :512] = rng.normal(size=(512, 2))
    X = torch.from_numpy(X)
    for out in (FE(X), FE(X, sample_keep=(torch.rand(4, 1024) > 0.75).float(),
                         power=torch.ones(4))):
        for k, v in out.items():
            if v is not None:
                assert torch.isfinite(v.float()).all(), k


def test_batch_invariance():
    rng = np.random.default_rng(6)
    X = _iq([_linear("QPSK", rng, sps=4.0), _radar("LFM", rng), _fsk(2, rng)])
    full = FE(X)
    for i in range(3):
        one = FE(X[i:i + 1])
        for k in ("time_ch", "spec", "caf", "dechirp", "acf", "hist", "stats", "const"):
            assert (one[k][0] - full[k][i]).abs().max() < 5e-3, k


def test_feature_switches():
    fe = SignalFrontEnd(seq_len=256, acf_lags=128, dechirp_rates=(), ccaf_lags=())
    out = fe(_iq([_linear("QPSK", np.random.default_rng(0), N=256)], N=256))
    assert out["dechirp"] is None and fe.view_shapes()["dechirp"] is None
    assert out["spec"].shape[1] == 3
    assert torch.isfinite(out["stats"]).all()
    assert math.isclose(float(out["stats"][0, S.index("dechirp_gain")]), 0.0)
