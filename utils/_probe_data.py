"""快速探查官方训练数据（临时脚本，探查用后删除）"""
import collections
import pyarrow.parquet as pq
import numpy as np

FILES = [
    r"datasets\01\rf_signal_benchmark_train_v1.0\rf_signal_benchmark_train_v1.0_part_001_of_003.parquet",
]

PARAMS = ["symbol_rate", "bandwidth", "frequency_shift", "pulse_width",
          "pri", "time_delay", "number_of_pulses"]

tc = collections.Counter()      # task 分布
mc = collections.Counter()      # modulation 标签
pc = collections.Counter()      # protocol 标签
lens = collections.Counter()    # iq 长度
valid_m = collections.Counter() # modulation_valid
valid_p = collections.Counter() # protocol_valid
amps = []                       # 峰值幅度样本

for path in FILES:
    f = pq.ParquetFile(path)
    n_rg = min(f.num_row_groups, 60)  # 每文件抽 60 行组（约 6 万行）
    for i in range(n_rg):
        df = f.read_row_group(i).to_pandas()
        tc.update(df["task"])
        lens.update(df["iq"].apply(len))
        m = df[df["task"] == "modulation"]
        mc.update(m["modulation_label"])
        valid_m.update(m["modulation_valid"])
        p = df[df["task"] == "protocol"]
        pc.update(p["protocol_label"])
        valid_p.update(p["protocol_valid"])
        # 幅度抽样（每行组取 2 条）
        for r in df["iq"].iloc[::512][:2]:
            a = np.asarray(r.tolist(), dtype=np.float32)  # list<list<float>> -> ndarray
            amps.append(float(np.sqrt((a ** 2).sum(1)).max()))

print("抽样 task 分布:", dict(tc), "（每文件60行组×1024行）")
print("iq 长度分布:", dict(lens))
print("modulation_valid:", dict(valid_m), " protocol_valid:", dict(valid_p))
print()
print("modulation 标签分布:")
for k, v in mc.most_common():
    print(f"  {k}: {v}")
print()
print("protocol 标签分布:")
for k, v in pc.most_common():
    print(f"  {k}: {v}")
print()
print(f"峰值幅度: min={min(amps):.4f} max={max(amps):.4f} mean={np.mean(amps):.4f}")
print()

# ---- parameter 任务参数统计（来自 02/03 分块）----
PFILES = [
    r"datasets\02\rf_signal_benchmark_train_v1.0\rf_signal_benchmark_train_v1.0_part_002_of_003.parquet",
    r"datasets\03\rf_signal_benchmark_train_v1.0\rf_signal_benchmark_train_v1.0_part_003_of_003.parquet",
]
pvals = {p: [] for p in PARAMS}
pvalid = collections.Counter()
for pf in PFILES:
    f = pq.ParquetFile(pf)
    step = max(1, f.num_row_groups // 40)  # 均匀抽 40 个行组
    for i in range(0, f.num_row_groups, step):
        df = f.read_row_group(i).to_pandas()
        par = df[df["task"] == "parameter"]
        if len(par) == 0:
            continue
        for p in PARAMS:
            pv = [x for x, v in zip(par[p].tolist(), par[p + "_valid"].tolist()) if v]
            pvals[p].extend(pv)
            pvalid[f"{p}_true"] += len(pv)
            pvalid[f"{p}_false"] += int(len(par) - len(pv))

print("parameter 任务（60行组抽样，仅统计 valid=True 的值）:")
for p in PARAMS:
    a = np.asarray(pvals[p], dtype=np.float64)
    print(f"  {p:<16s} n={len(a):>6d} min={a.min():.6g} p25={np.percentile(a,25):.6g} "
          f"med={np.median(a):.6g} p75={np.percentile(a,75):.6g} max={a.max():.6g} "
          f"valid率={100*pvalid[p+'_true']/(pvalid[p+'_true']+pvalid[p+'_false']):.0f}%")

# ---- protocol 任务位置探查 ----
pro_rows = []
f2 = pq.ParquetFile(r"datasets\02\rf_signal_benchmark_train_v1.0\rf_signal_benchmark_train_v1.0_part_002_of_003.parquet")
i, found = 0, 0
while i < f2.num_row_groups and found < 3:
    df = f2.read_row_group(i).to_pandas()
    m = df[df["task"] == "protocol"]
    if len(m):
        pro_rows.append(m)
        found += 1
    i += max(1, f2.num_row_groups // 200)
if pro_rows:
    import pandas as pd
    allp = pd.concat(pro_rows, ignore_index=True)
    print()
    print(f"protocol 样本预览（{len(allp)} 条）:")
    print(allp[["sample_id", "protocol_label", "protocol_valid"]].head(3).to_string())
    r = allp.iloc[0]
    iq = np.array(r["iq"].tolist(), dtype=np.float32)
    print("首条 protocol iq:", iq.shape, "峰值幅度:", np.sqrt((iq ** 2).sum(1)).max())
else:
    print("part02 未找到 protocol 样本（可能集中在某段行组）")
