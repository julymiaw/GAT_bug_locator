#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
排序优化损失函数

这个模块提供了专门为提高排序质量设计的损失函数，
旨在改进缺陷定位模型的MAP和MRR指标。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedRankMSELoss(nn.Module):
    """
    加权MSE损失函数 - 给予修复文件更高的权重

    这是最简单的排序优化损失函数，通过对正样本(修复文件)施加更高的权重，
    使模型更关注修复文件的排序位置。

    参数:
        fix_weight: 修复文件的权重倍数，默认为5.0
    """

    def __init__(self, fix_weight=5.0):
        super().__init__()
        self.fix_weight = fix_weight

    def forward(self, scores, targets):
        # 创建权重向量 - 修复文件获得更高权重
        weights = torch.ones_like(targets)
        weights[targets > 0] = self.fix_weight

        # 加权MSE损失
        loss = torch.mean(weights * (scores - targets) ** 2)
        return loss
