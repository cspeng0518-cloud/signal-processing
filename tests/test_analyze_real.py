"""analyze_real.py 测试用例（官方真实数据分析工具）

依赖 datasets/01~03 下的官方训练数据，若不存在则全部 skip。
"""

import os
import sys
import json
import glob
import unittest

import numpy as np
import pandas as pd

UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.insert(0, UTILS_DIR)

import analyze_real  # noqa: E402


def _real_data_available():
    return all(os.path.exists(p) for p in analyze_real.TRAIN_FILES)


@unittest.skipUnless(_real_data_available(), "官方真实数据未下载")
class TestScanAll:
    """全量扫描函数（快速模式：每文件 2 行组）"""

    @classmethod
    def setup_class(cls):
        cls.stats = analyze_real.scan_all(row_groups=2)

    def test_rows_scanned(self):
        """快速模式应读到 3 文件 × 2 行组 × 1024 行 = 6144 行"""
        assert self.stats["n_rows"] == 3 * 2 * 1024

    def test_task_counter(self):
        """task 只应有三个取值，且计数总数 = 总行数"""
        c = self.stats["task_counter"]
        assert set(c.keys()) <= {"modulation", "protocol", "parameter"}
        assert sum(c.values()) == self.stats["n_rows"]

    def test_modulation_labels_official_23(self):
        """modulation 标签必须 ⊆ 官方 schema 的 23 类"""
        official = analyze_real.MOD_LABELS if hasattr(analyze_real, "MOD_LABELS") else None
        if official is None:
            # 从官方 task_schema.json 读取
            schema_path = os.path.join(os.path.dirname(UTILS_DIR), "datasets", "01",
                                       "rf_signal_benchmark_train_v1.0", "task_schema.json")
            if os.path.exists(schema_path):
                with open(schema_path, encoding="utf-8") as f:
                    official = json.load(f)["classification_labels"]["modulation"]
        if official:
            assert set(self.stats["mod_counter"].keys()) <= set(official)

    def test_param_valid_exclusive(self):
        """parameter 样本中通信 3 参数与雷达 4 参数的有效位应互斥
        （组合模式只应出现 1110000 / 0001110 及其同类）"""
        cc = self.stats["valid_combo_counter"]
        for pattern in cc:
            comm = pattern[:3]   # symbol_rate/bandwidth/frequency_shift
            radar = pattern[3:]  # pulse_width/pri/time_delay/number_of_pulses
            # 任意一组有效时另一组应全无效
            if "1" in comm:
                assert radar == "0000", f"通信/雷达参数有效位未互斥: {pattern}"

    def test_iq_lengths(self):
        """iq 长度只应为 512 或 1024"""
        assert set(self.stats["iq_len_counter"].keys()) <= {512, 1024}


@unittest.skipUnless(_real_data_available(), "官方真实数据未下载")
class TestSampleFigures:
    """样本图生成"""

    def test_generate_sample_figures_smoke(self):
        """抽 1 类生成 1 张图，文件存在且 >10KB"""
        saved = analyze_real.generate_sample_figures(max_per_class=1)
        protocol_figs = [p for p in saved if "protocol" in p]
        assert len(saved) > 0
        for p in saved[:3]:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 10_000


@unittest.skipUnless(_real_data_available(), "官方真实数据未下载")
class TestIOValidation:
    """官方自带的 sample_io 评测样例一致性验证"""

    IO_DIR = os.path.join(os.path.dirname(UTILS_DIR), "datasets", "01",
                          "rf_signal_sample_io_v1.0")

    def test_io_input_matches_expected(self):
        """期望输出 jsonl 的 sample_id/顺序应与输入 parquet 一致"""
        parquet = os.path.join(self.IO_DIR, "rf_signal_sample_io_input_v1.0.parquet")
        expected = os.path.join(self.IO_DIR, "rf_signal_sample_io_expected_output_v1.0.jsonl")
        if not (os.path.exists(parquet) and os.path.exists(expected)):
            self.skipTest("sample_io 文件缺失")
        df = pd.read_parquet(parquet)
        with open(expected, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(df) == len(lines)
        assert list(df["sample_id"]) == [l["sample_id"] for l in lines]
        # task 也一致
        assert list(df["task"]) == [l["task"] for l in lines]

    def test_expected_matches_truth(self):
        """官方期望输出应与真值 json 一致（确认 truth 是标准答案）"""
        expected = os.path.join(self.IO_DIR, "rf_signal_sample_io_expected_output_v1.0.jsonl")
        truth = os.path.join(self.IO_DIR, "rf_signal_sample_io_truth_v1.0.json")
        if not (os.path.exists(expected) and os.path.exists(truth)):
            self.skipTest("sample_io 文件缺失")
        with open(truth, encoding="utf-8") as f:
            t = json.load(f)
        with open(expected, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for l in lines:
            if l["task"] in ("modulation", "protocol"):
                assert l["prediction"] == t[l["task"]][l["sample_id"]]
            else:
                for k, v in t[l["task"]][l["sample_id"]].items():
                    if k.endswith("_valid"):
                        continue
                    assert abs(l["prediction"][k] - v) < 1e-9


if __name__ == "__main__":
    unittest.main()
