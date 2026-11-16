import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from infer import infer_file
from utils.dataset import PARAM_NAMES


class FakeModel:
    def __call__(self, iq, sr, task):
        n=iq.size(0)
        if task=="parameter":return torch.ones(n,7)*.5
        if task=="parameter_pattern":return torch.tensor([[10.,0.]]).expand(n,-1)
        if task=="protocol":return torch.eye(8)[:1].expand(n,-1)
        return torch.eye(23)[:1].expand(n,-1)


def test_mixed_task_order_and_zero_invalid(tmp_path, monkeypatch):
    data=pd.DataFrame({"task":["parameter","modulation","protocol"],
                       "sample_id":["p","m","s"],
                       "iq":[np.zeros((64,2),dtype=np.float32)]*3})
    monkeypatch.setattr(pd,"read_parquet",lambda *a,**kw:data.copy())
    out=tmp_path/"pred.jsonl"
    stats={"min":np.array([10]*7,dtype=np.float32),"max":np.array([20]*7,dtype=np.float32)}
    infer_file(FakeModel(),"dummy.parquet",str(out),64,stats,torch.device("cpu"),batch_size=3)
    import json
    rows=[json.loads(x) for x in out.read_text().splitlines()]
    assert [x["sample_id"] for x in rows]==["p","m","s"]
    assert rows[0]["prediction"][PARAM_NAMES[0]]==15.
    assert all(rows[0]["prediction"][name]==0. for name in PARAM_NAMES[3:])
