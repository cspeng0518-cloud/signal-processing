"""生成少量模拟电磁信号数据（用于跑通代码框架）

按赛题官方 Parquet schema 生成:
  - task: system / modulation / parameter
  - sample_id: 唯一标识
  - iq: [N, 2] float32 数组 (I/Q 双通道)
  - 训练集额外字段: label(分类) / 参数字段(回归) + valid_* 有效性标记

信号类型:
  - 体制识别 8 类: WiFi, Bluetooth, 5G, Radar, ...
  - 调制识别 23 类: OOK, BPSK, QPSK, 16QAM, LFM, Barker Code, ...
  - 参数估计 7 个: symbol_rate, bandwidth, frequency_shift, pulse_width, pri, time_delay, number_of_pulses
"""

import os
import numpy as np
import pandas as pd

SEED = 2023
rng = np.random.default_rng(SEED)

# ---- 类别定义 ----
SYSTEM_CLASSES = [  # 体制类型 8 类
    "wifi", "bluetooth", "5g", "radar", "lora", "zwave", "gps", "dtmb",
]
MODULATION_CLASSES = [  # 调制类型 23 类
    "ook", "bpsk", "qpsk", "8psk", "16qam", "32qam", "64qam",
    "fsk", "gfsk", "msk", "cpfsk", "pam4", "am_dsb", "am_ssb",
    "lfm", "nlfm", "barker", "frank", "polyphase", "costas",
    "t1", "t2", "t3",
]
PARAM_NAMES = [  # 物理域 7 参数
    "symbol_rate", "bandwidth", "frequency_shift",
    "pulse_width", "pri", "time_delay", "number_of_pulses",
]

# 调制 -> 体制 的简单映射（体制识别是调制识别的粗粒度版本）
MOD2SYSTEM = {
    "ook": "wifi", "gfsk": "bluetooth", "qpsk": "5g", "16qam": "5g",
    "64qam": "5g", "bpsk": "zwave", "fsk": "lora", "msk": "lora",
    "lfm": "radar", "nlfm": "radar", "barker": "radar", "frank": "radar",
    "polyphase": "radar", "costas": "radar", "pam4": "gps", "am_dsb": "dtmb",
    "am_ssb": "dtmb", "8psk": "gps", "32qam": "gps", "cpfsk": "gps",
    "t1": "radar", "t2": "radar", "t3": "radar",
}

FS = 1_000_000.0  # 统一采样率 1 MHz（采样率 token 用）


# ============================================================
# 波形生成器：生成复基带信号 s[n] = I[n] + jQ[n]
# ============================================================

def _noise(n, snr_db):
    """加性高斯白噪声，SNR 转噪声功率"""
    power = 10 ** (-snr_db / 10)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(power / 2)


def gen_comm_waveform(mod, n, symbol_rate):
    """通信信号：按码元生成随机符号再上采样"""
    n_sym = max(int(n * symbol_rate / FS), 1)
    t_sym = np.arange(n_sym) / symbol_rate
    # 随机符号
    if mod == "ook":
        syms = (rng.random(n_sym) > 0.5).astype(complex)
    elif mod in ("bpsk",):
        syms = rng.choice([1 + 0j, -1 + 0j], n_sym)
    elif mod == "qpsk":
        syms = np.exp(1j * rng.choice([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4], n_sym))
    elif mod in ("8psk",):
        syms = np.exp(1j * rng.integers(0, 8, n_sym) * np.pi / 4)
    elif mod.endswith("qam"):
        m = int(mod[:-3])
        k = int(np.sqrt(m))
        real = rng.integers(0, k, n_sym) - (k - 1) / 2
        imag = rng.integers(0, k, n_sym) - (k - 1) / 2
        syms = (real + 1j * imag) / np.sqrt((k**2 - 1) / 6 * 2)
    elif mod == "pam4":
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(5)
        syms = rng.choice(levels, n_sym)
    elif mod in ("fsk", "gfsk", "cpfsk"):
        freqs = rng.choice([-2, -1, 1, 2], n_sym) * symbol_rate
        phase = 2 * np.pi * np.cumsum(freqs) / symbol_rate
        syms = np.exp(1j * phase)
    elif mod == "msk":
        bits = rng.choice([-1, 1], n_sym)
        phase = np.pi / 2 * np.cumsum(bits)
        syms = np.exp(1j * phase)
    elif mod == "am_dsb":
        syms = (1 + 0.6 * rng.standard_normal(n_sym)) * np.exp(1j * rng.uniform(0, 2*np.pi, 1))
        syms = np.abs(syms) * np.exp(1j * 0)
    elif mod == "am_ssb":
        syms = 0.6 * rng.standard_normal(n_sym) + 0.5j * rng.standard_normal(n_sym)
    else:  # 未知调制 -> QPSK 兜底
        syms = np.exp(1j * rng.choice([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4], n_sym))

    # 矩形脉冲成型 + 上采样到 n 点
    t = np.arange(n) / FS
    idx = np.clip((t * symbol_rate).astype(int), 0, n_sym - 1)
    return syms[idx]


def gen_radar_waveform(wave, n, bw, pulse_width_s, pri_s, time_delay_s, n_pulses):
    """雷达信号：脉冲串（LFM chirp / Barker / 杂类波形）

    返回 (复信号, 有效点数标记)
    """
    t = np.arange(n) / FS
    sig = np.zeros(n, dtype=complex)
    fs = FS
    # 脉宽内采样点数
    n_pw = int(pulse_width_s * fs)
    if n_pw < 4:
        n_pw = 4
    t_pw = np.arange(n_pw) / fs  # 脉内时间轴

    # 脉内波形
    if wave == "lfm":
        k = bw / pulse_width_s  # 调频斜率
        intra = np.exp(1j * np.pi * k * t_pw**2)
    elif wave == "nlfm":
        k = bw / pulse_width_s
        intra = np.exp(1j * np.pi * k * t_pw**2 * (t_pw / pulse_width_s))
    elif wave == "barker":
        code = [1,1,1,1,1,0,1,1,0,1,0,1,1]  # Barker-13
        n_chip = max(n_pw // len(code), 1)
        chip_t = np.repeat(np.array(code) * 2 - 1, n_chip)
        chip_t = np.tile(chip_t, int(np.ceil(n_pw / len(chip_t))))[:n_pw]
        intra = chip_t.astype(complex)
    else:  # frank/polyphase/costas/t1/t2/t3 -> 相位编码简化
        phases = rng.uniform(0, 2 * np.pi, 32)
        n_chip = max(n_pw // 32, 1)
        chip_t = np.repeat(np.exp(1j * phases), n_chip)
        chip_t = np.tile(chip_t, int(np.ceil(n_pw / len(chip_t))))
        intra = chip_t
    # 保证脉内波形长度 >= n_pw（截断/补齐）
    if len(intra) < n_pw:
        intra = np.tile(intra, int(np.ceil(n_pw / len(intra))))
    intra = intra[:n_pw]

    # 脉冲串排布
    start = int(time_delay_s * fs)
    for p in range(int(n_pulses)):
        s = start + int(p * pri_s * fs)
        if s >= n:
            break
        e = min(s + n_pw, n)
        sig[s:e] = intra[: e - s]
    return sig


# ============================================================
# 样本生成
# ============================================================

def make_one_sample(idx, task):
    """生成一条样本，返回 dict"""
    length = int(rng.choice([512, 1024]))
    snr_db = float(rng.uniform(0, 20))
    d = {"sample_id": f"sample_{idx:06d}", "task": task}

    if task == "parameter":
        # 雷达参数估计样本
        bw = float(rng.uniform(1e5, 4e5))
        pulse_width = float(rng.uniform(5e-6, 50e-6))
        pri = float(pulse_width * rng.uniform(2, 5))
        time_delay = float(rng.uniform(0, 100e-6))
        n_pulses = int(rng.integers(2, 6))
        freq_shift = float(rng.uniform(-2e5, 2e5))
        symbol_rate = float(rng.uniform(1e4, 2e5))
        sig = gen_radar_waveform("lfm", length, bw, pulse_width, pri, time_delay, n_pulses)
        sig = sig * np.exp(1j * 2 * np.pi * freq_shift * np.arange(length) / FS)
        params = [symbol_rate, bw, freq_shift, pulse_width, pri, time_delay, float(n_pulses)]
        d["iq"] = (sig + _noise(length, snr_db))
        for name, v in zip(PARAM_NAMES, params):
            d[f"param_{name}"] = v
            d[f"valid_{name}"] = True
        return d

    # 分类样本：随机选一个调制（雷达波形也算"调制"类别）
    mod = MODULATION_CLASSES[int(rng.integers(0, len(MODULATION_CLASSES)))]
    if mod in ("lfm", "nlfm", "barker", "frank", "polyphase", "costas", "t1", "t2", "t3"):
        sig = gen_radar_waveform(mod, length, bw=float(rng.uniform(1e5, 4e5)),
                                 pulse_width_s=float(rng.uniform(5e-6, 50e-6)),
                                 pri_s=float(rng.uniform(2e-5, 2e-4)),
                                 time_delay_s=float(rng.uniform(0, 1e-4)),
                                 n_pulses=int(rng.integers(2, 6)))
    else:
        sig = gen_comm_waveform(mod, length, symbol_rate=float(rng.uniform(1e4, 2e5)))
        sig = sig * np.exp(1j * 2 * np.pi * float(rng.uniform(-1e5, 1e5)) * np.arange(length) / FS)

    sig = sig + _noise(length, snr_db)
    d["iq"] = sig
    if task == "modulation":
        d["label"] = mod
    else:  # system
        d["label"] = MOD2SYSTEM[mod]
    return d


def iq_to_pairs(sig):
    """复信号 -> [N,2] 实数数组（官方 iq 字段格式）"""
    return np.stack([sig.real, sig.imag], axis=1).astype(np.float32)


def iq_to_bytes(sig):
    """[N,2] float32 数组 -> 扁平 bytes（Parquet 列存储兼容）"""
    arr = np.ascontiguousarray(iq_to_pairs(sig), dtype=np.float32)
    return arr.tobytes()


def build_split(n_samples, out_dir, split_name):
    os.makedirs(out_dir, exist_ok=True)
    for task in ["system", "modulation", "parameter"]:
        rows = []
        for i in range(n_samples):
            d = make_one_sample(n_samples * 1000 + i, task)
            row = {"task": d["task"], "sample_id": d["sample_id"],
                   "iq": iq_to_bytes(np.asarray(d["iq"]))}
            if "label" in d:
                row["label"] = d["label"]
            if "param_symbol_rate" in d:
                for name in PARAM_NAMES:
                    row[f"param_{name}"] = d[f"param_{name}"]
                    row[f"valid_{name}"] = d[f"valid_{name}"]
            rows.append(row)
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, f"{split_name}_{task}.parquet")
        df.to_parquet(path)
        print(f"[{split_name}] task={task}: {len(df)} samples -> {path}")


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets")
    base = os.path.abspath(base)
    # 少量数据：train 200/任务, val 40/任务（框架演示足够）
    build_split(200, base, "train")
    build_split(40, base, "val")
    # 测试集（无标签，模拟官方盲测输入）
    for task in ["system", "modulation", "parameter"]:
        rows = []
        for i in range(20):
            d = make_one_sample(900000 + i, task)
            row = {"task": d["task"], "sample_id": d["sample_id"],
                   "iq": iq_to_bytes(np.asarray(d["iq"]))}
            rows.append(row)
        pd.DataFrame(rows).to_parquet(os.path.join(base, f"test_{task}.parquet"))
        print(f"[test] task={task}: 20 samples (no labels)")
    print("Done. datasets at:", base)
