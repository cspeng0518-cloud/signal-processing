"""visualize.py 测试用例

运行:
    pytest tests/test_visualize.py -v
"""

import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from visualize import (
    load_samples, plot_time_domain, plot_constellation,
    plot_spectrum, plot_spectrogram, plot_sample_grid, plot_batch, FS,
)

DATASET_DIR = os.path.join(ROOT, "datasets")


# ---------- fixtures ----------

def _make_iq(n=512):
    """确定性测试信号: LFM chirp + 高斯噪声"""
    rng = np.random.default_rng(0)
    t = np.arange(n) / FS
    x = np.exp(1j * np.pi * 2e9 * t**2) + 0.1 * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return np.stack([x.real, x.imag], axis=1).astype(np.float32)


# ---------- 数据加载 ----------

class TestLoadSamples:
    def test_load_classification(self):
        """加载调制分类样本: 含 sample_id/task/iq/label"""
        path = os.path.join(DATASET_DIR, "train_modulation.parquet")
        if not os.path.exists(path):
            import pytest
            pytest.skip("datasets 未生成，先运行 code/utils/data_gen.py")
        samples = load_samples(path, n=2)
        assert len(samples) == 2
        s = samples[0]
        assert s["task"] == "modulation"
        assert "sample_id" in s and "label" in s
        assert s["iq"].ndim == 2 and s["iq"].shape[1] == 2
        assert s["iq"].shape[0] in (512, 1024)

    def test_load_parameter(self):
        """加载参数估计样本: 含 params 字典"""
        path = os.path.join(DATASET_DIR, "train_parameter.parquet")
        if not os.path.exists(path):
            import pytest
            pytest.skip("datasets 未生成")
        samples = load_samples(path, n=1)
        s = samples[0]
        assert "params" in s
        assert "bandwidth" in s["params"]
        assert "pulse_width" in s["params"]

    def test_load_by_indices(self):
        """指定下标加载"""
        path = os.path.join(DATASET_DIR, "train_modulation.parquet")
        if not os.path.exists(path):
            import pytest
            pytest.skip("datasets 未生成")
        samples = load_samples(path, indices=[0, 5])
        assert samples[0]["sample_id"] != samples[1]["sample_id"]


# ---------- 四种单图 ----------

class TestPlots:
    def setup_method(self):
        self.iq = _make_iq()

    def _run(self, plot_fn):
        fig, ax = plt.subplots(figsize=(4, 3))
        plot_fn(ax, self.iq)
        plt.close(fig)  # 不落盘，只验证不抛异常

    def test_time_domain(self):
        self._run(plot_time_domain)
        assert True

    def test_constellation(self):
        self._run(plot_constellation)
        assert True

    def test_spectrum(self):
        self._run(plot_spectrum)
        assert True

    def test_spectrogram(self):
        self._run(plot_spectrogram)
        assert True


# ---------- 组合大图 ----------

class TestSampleGrid:
    def test_grid_saves_png(self):
        """组合图保存为非空 PNG"""
        sample = {"sample_id": "unit_test_000", "task": "test",
                  "iq": _make_iq(256)}
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "grid.png")
            plot_sample_grid(sample, p)
            assert os.path.exists(p)
            assert os.path.getsize(p) > 10_000  # PNG 应有实际内容

    def test_batch(self):
        """批量生成: 输出文件数 == 样本数"""
        samples = [
            {"sample_id": f"b_{i}", "task": "test", "iq": _make_iq(256)}
            for i in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_batch(samples, tmp)
            assert len(paths) == 2
            assert all(os.path.exists(p) and os.path.getsize(p) > 10_000 for p in paths)

    def test_grid_with_label_and_params(self):
        """带 label/params 的样本标题不报错"""
        sample = {
            "sample_id": "unit_test_001", "task": "parameter",
            "iq": _make_iq(256), "label": "radar",
            "params": {"bandwidth": 2.5e5, "pulse_width": 1e-5},
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "grid2.png")
            plot_sample_grid(sample, p)
            assert os.path.getsize(p) > 10_000


# ---------- 端到端（真实数据 -> PNG） ----------

class TestEndToEnd:
    def test_real_parquet_to_figure(self):
        """真实 Parquet -> 组合图全流程"""
        path = os.path.join(DATASET_DIR, "train_modulation.parquet")
        if not os.path.exists(path):
            import pytest
            pytest.skip("datasets 未生成")
        samples = load_samples(path, n=1)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "e2e.png")
            plot_sample_grid(samples[0], p)
            assert os.path.getsize(p) > 10_000
