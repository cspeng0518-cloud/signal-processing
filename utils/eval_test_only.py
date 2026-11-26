#!/usr/bin/env python3
"""对已训练权重做 test 复算（不训练、不改权重、不写任何日志/权重文件）。

默认读 weights/ 下的 config.json + model.safetensors + param_codec.json：

  python utils/eval_test_only.py            # 等价于 --style conformer --branch-dir . \
                                            #   --weights weights --data datasets

由 utils/compare_branches.py --eval-only 调用时同样传全套显式参数。
输出与 finetune --eval_test 相同格式的 [test] 行，供 compare 脚本解析。
"""
import argparse
import json
import sys
from pathlib import Path


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return torch.device("npu:0")
    except Exception:
        pass
    return torch.device("cpu")


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="conformer", choices=("main", "conformer"))
    ap.add_argument("--branch-dir", default=str(root))
    ap.add_argument("--weights", default=str(root / "weights"), help="含 config.json/model.safetensors 的目录")
    ap.add_argument("--data", default=str(root / "datasets"))
    ap.add_argument("--batch-size", type=int, default=0, help="0=取 config 里的 batch_size")
    ap.add_argument("--ref-scale", type=float, default=0.5)
    a = ap.parse_args()

    branch = Path(a.branch_dir).resolve()
    sys.path.insert(0, str(branch / "code"))
    import torch
    from safetensors.torch import load_file

    from training.tasks import TASK_ORDER, TASKS
    from utils.dataset import build_task_dataloaders

    weights = Path(a.weights).resolve()
    cfg = json.loads((weights / "config.json").read_text(encoding="utf-8"))
    tcfg = cfg["training"]
    tcfg["ref_scale"] = a.ref_scale          # 强制统一口径，覆盖旧 run 导出的值
    bs = a.batch_size or int(tcfg.get("batch_size", 256))
    device = pick_device()
    print(f"device={device} weights={weights} batch_size={bs}", flush=True)

    if a.style == "conformer":
        from models import build_model
        from utils.param_codec import ParamCodec
        from utils.train_utils import autocast_context

        codec_path = weights / "param_codec.json"
        codec = (ParamCodec.load(str(codec_path)) if codec_path.exists()
                 else ParamCodec.default())
        cfg["model"]["param_head_spec"] = codec.head_spec()
        cfg["model"]["param_codec"] = codec.spec
        model = build_model(cfg["model"]).to(device)
        model.codec = codec
        model.load_state_dict(load_file(str(weights / "model.safetensors")), strict=False)
        loaders = build_task_dataloaders(
            a.data, batch_size=bs, seq_len=tcfg.get("seq_len", 1024),
            splits=("test",), num_workers=tcfg.get("num_workers", 4),
            eval_batch_size=tcfg.get("eval_batch_size", 1024))
        with autocast_context(device, tcfg.get("amp", True)):
            res = {t: TASKS[t].evaluate(model, loaders["test"][t], device, tcfg)
                   for t in TASK_ORDER if t in loaders.get("test", {})}
    else:
        from models import EMFoundation

        model = EMFoundation(cfg["model"]).to(device)
        model.load_state_dict(load_file(str(weights / "model.safetensors")), strict=False)
        loaders, stats = build_task_dataloaders(
            a.data, batch_size=bs, seq_len=tcfg.get("seq_len", 1024),
            splits=("test",), num_workers=tcfg.get("num_workers", 4))
        res = {t: TASKS[t].evaluate(model, loaders["test"][t], device, stats, a.ref_scale)
               for t in TASK_ORDER if t in loaders.get("test", {})}

    if not res:
        print(f"[test] no test split under {a.data}", flush=True)
        sys.exit(2)
    for t, m in res.items():
        print(f"[test] {t}: "
              + str({k: round(float(v), 6) for k, v in m.items()}), flush=True)
    if len(res) == 3:
        score = (0.3 * res["protocol"]["acc"] + 0.3 * res["modulation"]["acc"]
                 + 0.4 * res["parameter"]["precision"])
        print(f"[test] proxy score={score:.5f}", flush=True)


if __name__ == "__main__":
    main()
