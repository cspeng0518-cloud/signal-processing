"""模型与训练链路回归测试（需要 PyTorch，CPU 即可，约 1 分钟）。

覆盖: MAE 预训练前向/反向、微调三任务损失、赛题红线、512/1024 补零不变性、
      MAE 权重 -> 微调加载（旧版“0 个张量被加载”缺陷的回归）、导出 -> infer 往返。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from models import build_model                                   # noqa: E402
from training.common import load_mae_pretrained                  # noqa: E402
from training.tasks import TASKS                                 # noqa: E402
from training.trainer import export_model, export_pretrained     # noqa: E402
from utils.param_codec import ParamCodec                         # noqa: E402

N = 128
CFG = {
    "d_model": 64, "n_layers": 3, "nhead": 4, "ffn_hidden": 128,
    "layer_types": ["full_attention", "mla", "mla"], "mla_kv_lora_rank": 16,
    "mla_q_lora_rank": 16, "neck_layers": 1, "neck_layer_types": ["conformer"],
    "conv_kernel": 5, "seq_len": N, "patch_size": 8, "length_quantum": 64,
    "stft_win": 16, "stem_width": 16, "freq_layers": 1, "mae_decoder_dim": 32,
    "mae_decoder_layers": 1, "dropout": 0.0, "drop_path": 0.0, "neck_drop_path": 0.0,
    "head_config": {"protocol": {"hidden": 32, "layers": 1},
                    "modulation": {"hidden": 32, "layers": 1},
                    "parameter": {"hidden": 32, "layers": 1}},
    "supcon_dim": 16,
}
TCFG = {"augment": True, "label_smoothing": {"modulation": 0.1},
        "supcon_weight": {"modulation": 0.1}}


def _iq(B, valid_len=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, 2, generator=g)
    x[:, valid_len:] = 0
    return x


def _param_batch(B, codec):
    pat = torch.arange(B) % 2
    v = torch.zeros(B, 7)
    v[:, :3] = (pat == 0).float()[:, None]
    v[:, 3:] = (pat == 1).float()[:, None]
    lo, hi = torch.tensor(codec.vmin).float(), torch.tensor(codec.vmax).float()
    return {"iq": _iq(B), "pattern": pat, "valid": v,
            "raw_params": (lo + (hi - lo) * torch.rand(B, 7)) * v}


def test_pretrain_forward_backward_and_mixed_lengths():
    torch.manual_seed(0)
    m = build_model(dict(CFG), with_decoder=True)
    x = torch.cat([_iq(2), _iq(2, valid_len=64, seed=1)])     # 1024/512 混批
    loss, parts = m.forward_pretrain(x)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert m.backbone.layers[0].attn.qkv.weight.grad is not None
    assert all(torch.isfinite(v) for v in parts.values())


def test_conformer_not_allowed_in_backbone():
    with pytest.raises(ValueError):
        build_model(dict(CFG, layer_types=["conformer"]))


def test_finetune_losses_shapes_and_limits():
    torch.manual_seed(0)
    codec = ParamCodec.default()
    m = build_model(dict(CFG, param_head_spec=codec.head_spec()))
    m.codec = codec
    assert m.mae_decoder is None
    assert m.shared_ratio() > 0.8
    for task, n_cls in (("protocol", 8), ("modulation", 23)):
        b = {"iq": _iq(4), "label": torch.randint(0, n_cls, (4,))}
        loss, _ = TASKS[task].loss(m, b, TCFG)
        loss.backward()
        assert m(_iq(3), task).shape == (3, n_cls)
    loss, _ = TASKS["parameter"].loss(m, _param_batch(4, codec), TCFG)
    loss.backward()
    assert torch.isfinite(loss)
    m.eval()
    pred = m.predict(_iq(4), "parameter")
    assert pred.shape == (4, 7)
    zero_comm = (pred[:, :3] == 0).all(1)          # 每行要么通信 3 参为 0，要么雷达 4 参为 0
    zero_radar = (pred[:, 3:] == 0).all(1)
    assert bool((zero_comm | zero_radar).all())


def test_padding_invariance():
    """512 点样本单独推理 与 和 1024 点样本混批推理，输出应一致（padding mask 正确）。"""
    torch.manual_seed(0)
    m = build_model(dict(CFG)).eval()
    short = _iq(1, valid_len=64, seed=3)
    with torch.no_grad():
        alone = m(short, "modulation")
        mixed = m(torch.cat([short, _iq(1, seed=4)]), "modulation")[:1]
    assert torch.allclose(alone, mixed, atol=1e-4), (alone - mixed).abs().max()


def test_mae_weights_actually_load(tmp_path):
    torch.manual_seed(0)
    pre = build_model(dict(CFG), with_decoder=True)
    export_pretrained(pre, tmp_path)
    ft = build_model(dict(CFG))
    n = load_mae_pretrained(ft, tmp_path / "mae_pretrained.pt")
    assert n > 0
    assert torch.equal(ft.backbone.layers[1].attn.q_a_proj.weight,
                       pre.backbone.layers[1].attn.q_a_proj.weight)


def test_export_and_infer_roundtrip(tmp_path, monkeypatch):
    import infer
    codec = ParamCodec.default()
    cfg_m = dict(CFG, param_head_spec=codec.head_spec())
    m = build_model(cfg_m)
    m.codec = codec
    export_model(m, {"model": cfg_m, "training": {"seq_len": N}}, codec, tmp_path)
    model, cfg = infer.load_model(str(tmp_path), torch.device("cpu"))
    df = pd.DataFrame({"task": ["parameter", "modulation", "protocol", "modulation"],
                       "sample_id": ["p", "m", "s", "m2"],
                       "iq": [np.random.randn(n, 2).astype(np.float32).tolist()
                              for n in (N, N, N // 2, N // 2)]})
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: df.copy())
    out = tmp_path / "pred.jsonl"
    infer.infer_file(model, "dummy.parquet", str(out), N, torch.device("cpu"), batch_size=2)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [r["sample_id"] for r in rows] == ["p", "m", "s", "m2"]
    assert set(rows[0]["prediction"]) == {"symbol_rate", "bandwidth", "frequency_shift",
                                          "pulse_width", "pri", "time_delay", "number_of_pulses"}
    assert isinstance(rows[1]["prediction"], str)
