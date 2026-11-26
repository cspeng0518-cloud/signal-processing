"""官方评分指标实现

初赛评分规则：
  - 体制识别 30 分：准确率 A，同分比宏平均召回率
  - 调制识别 30 分：准确率 A，同分比宏平均召回率
  - 参数估计 40 分：精准度 P = 1/(1 + RMSE/参考尺度)
"""

import numpy as np


def accuracy(y_true, y_pred):
    """A = 预测正确样本数 / 全部样本数"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_recall(y_true, y_pred, n_classes=None):
    """宏平均召回率：各类召回率的算术平均（小类不能放弃）"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true) if n_classes is None else np.arange(n_classes)
    recalls = []
    for c in classes:
        m = y_true == c
        if m.sum() == 0:
            continue
        recalls.append((y_pred[m] == c).mean())
    return float(np.mean(recalls)) if recalls else 0.0


def rmse(y_true, y_pred, valid=None):
    """均方根误差。valid: [B,7] 有效位掩码（官方按有效性标记评分）"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        err = (y_pred - y_true) ** 2
        # 每个参数维度仅在有效样本上计算
        per_dim = []
        for j in range(err.shape[1]):
            m = valid[:, j]
            if m.sum() == 0:
                per_dim.append(0.0)
            else:
                per_dim.append(np.sqrt(err[m, j].mean()))
        return float(np.mean(per_dim))
    return float(np.sqrt(((y_pred - y_true) ** 2).mean()))


def precision_score(y_true, y_pred, ref_scale, valid=None):
    """官方精准度 P = 1 / (1 + RMSE / 固定参考尺度)"""
    r = rmse(y_true, y_pred, valid)
    return 1.0 / (1.0 + r / ref_scale)


def official_score(task, y_true, y_pred, weight, ref_scale=0.5, valid=None, n_classes=None):
    """任务得分 S = 分值 × 指标（task: protocol / modulation / parameter）"""
    if task in ("protocol", "system", "modulation"):  # system 为旧名兼容
        return weight * accuracy(y_true, y_pred)
    return weight * precision_score(y_true, y_pred, ref_scale, valid)
