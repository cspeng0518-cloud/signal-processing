from .foundation import EMFoundation


def build_model(model_cfg, with_decoder=False):
    """统一构建入口：预训练 with_decoder=True；微调/推理 False（不构建 MAE 解码器）。"""
    return EMFoundation(model_cfg, with_decoder=with_decoder)


__all__ = ["EMFoundation", "build_model"]
