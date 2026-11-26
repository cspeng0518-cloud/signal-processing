"""infer_file 行序/格式测试（假模型，不依赖权重）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from infer import infer_file               # noqa: E402
from utils.dataset import PARAM_NAMES      # noqa: E402


class FakeModel:
    def predict(self, iq, task):
        n = iq.size(0)
        if task == "parameter":
            out = torch.ones(n, 7) * 15.0
            out[:, 3:] = 0.0                       # 通信模式：雷达 4 参为 0
            return out
        return torch.zeros(n, dtype=torch.long)


def test_mixed_task_order_and_zero_invalid(tmp_path, monkeypatch):
    data = pd.DataFrame({"task": ["parameter", "modulation", "protocol"],
                         "sample_id": ["p", "m", "s"],
                         "iq": [np.zeros((64, 2), dtype=np.float32)] * 3})
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: data.copy())
    out = tmp_path / "pred.jsonl"
    infer_file(FakeModel(), "dummy.parquet", str(out), 64, torch.device("cpu"), batch_size=3,
               warmup=False)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["sample_id"] for x in rows] == ["p", "m", "s"]
    assert rows[0]["prediction"][PARAM_NAMES[0]] == 15.0
    assert all(rows[0]["prediction"][n] == 0.0 for n in PARAM_NAMES[3:])
    assert rows[1]["prediction"] == "OOK" and rows[2]["prediction"] == "WiFi"
