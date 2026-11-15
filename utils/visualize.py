"""电磁信号可视化工具（独立于 code/，不依赖训练框架）

从 datasets/ 下的 Parquet 文件加载 IQ 信号，为每条样本生成 2x2 组合图:
  左上: 时域波形（I/Q 双通道）
  右上: 星座图（相位累积校正前散点）
  左下: 频谱图（FFT 幅度谱，fs 归一化频率轴）
  右下: 时频图（STFT spectrogram）

用法:
    python utils/visualize.py --parquet datasets/train_modulation.parquet --n 4
    python utils/visualize.py --parquet datasets/train_parameter.parquet --n 2 --out figures
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，直接落盘
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

# 复用 code/utils 的数据读取（iq bytes 反序列化）
CODE_UTILS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "utils"))
sys.path.insert(0, os.path.dirname(CODE_UTILS))
from utils.dataset import bytes_to_iq  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

FS = 1_000_000.0  # 与 data_gen.py 一致的采样率


# ============================================================
# 数据加载
# ============================================================

def load_samples(parquet_path, n=4, indices=None):
    """从 Parquet 加载样本，返回 [{sample_id, task, iq, label/params}]"""
    df = pd.read_parquet(parquet_path)
    if indices is None:
        step = max(len(df) // n, 1)
        indices = list(range(0, len(df), step))[:n]
    samples = []
    for i in indices:
        row = df.iloc[i]
        s = {
            "sample_id": row["sample_id"],
            "task": row["task"],
            "iq": bytes_to_iq(row["iq"]),  # [N,2] float32
        }
        if "label" in df.columns:
            s["label"] = row["label"]
        if "param_symbol_rate" in df.columns:
            s["params"] = {k.replace("param_", ""): row[k]
                           for k in df.columns if k.startswith("param_")}
        samples.append(s)
    return samples


# ============================================================
# 单图绘制函数（每个返回一个 Axes）
# ============================================================

def plot_time_domain(ax, iq, fs=FS):
    """时域波形: I/Q 双通道"""
    t = np.arange(iq.shape[0]) / fs * 1e3  # ms
    ax.plot(t, iq[:, 0], lw=0.8, label="I")
    ax.plot(t, iq[:, 1], lw=0.8, label="Q", alpha=0.8)
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("幅度")
    ax.set_title("时域波形")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)


def plot_constellation(ax, iq):
    """星座图: IQ 散点（每 4 点抽 1，避免过密）"""
    ax.scatter(iq[::4, 0], iq[::4, 1], s=2, c=np.arange(len(iq[::4])), cmap="viridis", alpha=0.6)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("星座图")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def plot_spectrum(ax, iq, fs=FS):
    """频谱图: FFT 幅度谱（dB）"""
    x = iq[:, 0] + 1j * iq[:, 1]
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    X = np.fft.fftshift(np.fft.fft(x, n_fft))
    f = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / fs)) / 1e3  # kHz
    mag = 20 * np.log10(np.abs(X) / n_fft + 1e-12)
    ax.plot(f, mag, lw=0.8)
    ax.set_xlabel("频率 (kHz)")
    ax.set_ylabel("幅度 (dB)")
    ax.set_title("频谱图")
    ax.grid(alpha=0.3)


def plot_spectrogram(ax, iq, fs=FS):
    """时频图: STFT spectrogram"""
    x = iq[:, 0] + 1j * iq[:, 1]
    nperseg = min(128, len(x))
    f, t, Sxx = sp_signal.spectrogram(np.abs(x), fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    pcm = ax.pcolormesh(t * 1e3, f / 1e3, 10 * np.log10(Sxx + 1e-12),
                        shading="auto", cmap="viridis")
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("频率 (kHz)")
    ax.set_title("时频图 (STFT)")
    plt.colorbar(pcm, ax=ax, label="功率 (dB)")


# ============================================================
# 组合大图（2x2）
# ============================================================

def plot_sample_grid(sample, out_path, fs=FS):
    """单条样本的 2x2 组合图，保存到 out_path"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    title = f"sample_id={sample['sample_id']}  task={sample['task']}"
    if "label" in sample and sample.get("label"):
        title += f"  label={sample['label']}"
    if sample.get("params"):
        parts = [f"{k}={v:.3g}" for k, v in sample["params"].items()]
        title += "\n" + "  ".join(parts)
    fig.suptitle(title, fontsize=11)
    plot_time_domain(axes[0, 0], sample["iq"], fs)
    plot_constellation(axes[0, 1], sample["iq"])
    plot_spectrum(axes[1, 0], sample["iq"], fs)
    plot_spectrogram(axes[1, 1], sample["iq"], fs)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_batch(samples, out_dir, prefix="sample", fs=FS):
    """批量生成组合图，返回输出文件路径列表"""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for s in samples:
        name = f"{prefix}_{s['sample_id']}.png"
        p = os.path.join(out_dir, name)
        plot_sample_grid(s, p, fs)
        paths.append(p)
        print(f"[viz] saved {p}")
    return paths


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="IQ 信号可视化")
    parser.add_argument("--parquet", required=True, help="Parquet 文件路径")
    parser.add_argument("--n", type=int, default=4, help="展示样本数")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="指定样本下标（优先于 --n）")
    parser.add_argument("--out", default="figures", help="输出目录")
    args = parser.parse_args()

    samples = load_samples(args.parquet, n=args.n, indices=args.indices)
    print(f"loaded {len(samples)} samples from {args.parquet}")
    paths = plot_batch(samples, args.out)
    print(json.dumps({"status": "ok", "figures": paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()
