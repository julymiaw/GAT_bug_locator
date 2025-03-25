#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
权重计算函数模块

提供多种特征权重计算方法，用于缺陷定位算法中的特征重要性评估。
包括统计检验方法、树模型方法、降维方法等。
"""

import numpy as np
import pandas as pd
from typing import List, Callable, Tuple
from scipy.stats import kruskal, ttest_ind, levene
from sklearn.decomposition import FastICA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
)
from sklearn.feature_selection import chi2, mutual_info_classif, VarianceThreshold
from sklearn.utils import safe_mask

from metrics import calculate_metric_results


# region 核心权重计算函数


def _weights_normalize(weights: np.ndarray):
    """
    权重归一化函数

    参数：
        weights: 原始权重向量

    返回：
        weights: L1归一化后的权重向量（总和为1）

    说明：
        - 当权重总和>0时执行归一化
        - 处理全零权重时保持原值
    """
    weights_sum = weights.sum()
    if weights_sum > 0:
        weights /= weights_sum

    return weights


def weights_chi2(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    基于卡方检验的特征权重计算

    参数：
        df (pd.DataFrame): 包含特征和标签的数据
        columns (list): 特征列名列表

    返回：
        np.ndarray: 归一化的卡方统计量作为特征权重

    实现：
        使用sklearn.feature_selection.chi2计算各特征与目标变量的卡方统计量
    """
    weights = chi2(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def weights_mutual_info_classif(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """基于互信息的特征权重计算，适用于连续特征"""
    weights = mutual_info_classif(
        df[columns], df["used_in_fix"], discrete_features=False
    )
    weights = weights

    return _weights_normalize(weights)


def weights_FastICA(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """使用独立成分分析(ICA)的首个成分作为特征权重"""
    m = FastICA(n_components=1)
    m.fit(df[columns])
    weights = m.components_[0]

    return _weights_normalize(weights)


def weights_variance(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """基于特征方差的权重计算，负方差置零处理"""
    fs = VarianceThreshold()
    fs.fit(df[columns])
    weights = fs.variances_
    weights[weights < 0] = 0

    return _weights_normalize(weights)


def weights_const(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """恒定权重（0.5），用作基准方法"""
    return np.ones(df[columns].shape[1]) * 0.5


def weights_ExtraTreesClassifier(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """基于极端随机树分类器的特征重要性"""
    tree = ExtraTreesClassifier(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_GradientBoostingClassifier(
    df: pd.DataFrame, columns: List[str]
) -> np.ndarray:
    """梯度提升回归树特征重要性"""
    tree = GradientBoostingRegressor(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_AdaBoostClassifier(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """AdaBoost分类器特征重要性"""
    tree = AdaBoostClassifier(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_kruskal_classif(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    Kruskal-Wallis检验统计量作为权重
    适用于非正态分布的组间差异检验
    """
    weights = kruskal_classif(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def kruskal_classif(X, y):
    """
    执行Kruskal-Wallis H检验
    返回各特征的检验统计量绝对值和p值
    """
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = kruskal(*args)
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_ttest_ind_classif(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    独立样本t检验统计量作为权重
    假设方差不相等（Welch's t-test）
    """
    weights = ttest_ind_classif(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def ttest_ind_classif(X, y):
    """执行Welch's t-test，返回绝对t值和p值"""
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = ttest_ind(*args, equal_var=False)
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_levene_median(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    基于中位数Levene检验的权重计算
    用于检测组间方差差异
    """
    weights = levene_median(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def levene_median(X, y):
    """执行基于中位数的Levene方差齐性检验"""
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = levene(args[0], args[1], center="median")
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_mean_var(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    均值-方差比率权重：
    (修复样本的变异系数) / (非修复样本的变异系数)
    变异系数 = 标准差 / 均值
    """
    weights_var = np.var(df[df["used_in_fix"] == 1][columns], axis=0)
    weights_mean = np.mean(df[df["used_in_fix"] == 1][columns], axis=0)
    weights_var1 = np.var(df[df["used_in_fix"] == 0][columns], axis=0)
    weights_var1_mean = np.mean(df[df["used_in_fix"] == 0][columns], axis=0)

    return (weights_var / weights_mean) / (weights_var1 / weights_var1_mean)


def weights_maximum_absolute_deviation(
    df: pd.DataFrame, columns: List[str]
) -> np.ndarray:
    """
    最大绝对偏差权重：
    计算修复样本各特征值与其最大值的平均绝对偏差
    """
    weights_max = np.max(df[df["used_in_fix"] == 1][columns], axis=0)
    weights_mad = np.mean(
        np.abs(df[df["used_in_fix"] == 1][columns] - weights_max), axis=0
    )

    return weights_mad


# endregion


# region 权重评估辅助函数


def weights_on_df(
    method: Callable[[pd.DataFrame, List[str]], np.ndarray],
    df: pd.DataFrame,
    columns: List[str],
):
    """
    单权重方法计算包装函数

    参数：
        method: 权重计算方法
        df: 当前折训练数据
        columns: 特征列名列表

    返回：
        tuple: (方法名称, 权重向量)

    说明：
        - 用于并行计算任务包装
        - 调用具体权重计算方法并返回标准化结果
    """
    weights = method(df, columns)
    return method.__name__, weights


def eval_weights(
    m_name: str, weights: np.ndarray, df: pd.DataFrame, columns: List[str]
) -> Tuple[str, Tuple[np.ndarray, float]]:
    """
    权重评估函数（验证阶段）

    参数：
        m_name: 权重方法名称
        weights: 已计算的权重向量
        df: 验证集数据
        columns: 特征列名列表

    返回：
        tuple: (方法名称, (权重向量, MAP得分))

    流程：
        1. 计算验证集预测得分：X * weights
        2. 调用evaluate_fold计算MAP指标
    """
    Y = np.dot(df[columns], weights)
    return m_name, (weights, evaluate_fold(df, Y))


def evaluate_fold(df: pd.DataFrame, Y: np.ndarray) -> float:
    """
    评估预测结果的MAP指标

    参数：
        df: 待评估数据集（需包含used_in_fix列）
        Y: 预测得分向量

    返回：
        m_a_p: 平均精度均值（Mean Average Precision）

    流程：
        1. 构建结果数据框（含预测得分）
        2. 确定最小修复得分阈值（实际修复样本的最低得分）
        3. 生成候选集（预测得分≥阈值的样本）
        4. 调用calculate_metric_results计算指标
    """
    r = df[["used_in_fix"]].copy(deep=False)
    r["result"] = Y
    m_a_p = calculate_metric_results(r, metric_type="MAP")
    return m_a_p


# endregion


def get_weight_method_by_name(
    name: str,
) -> Callable[[pd.DataFrame, List[str]], np.ndarray]:
    """
    根据方法名称获取权重计算函数

    参数：
        name: 权重方法名称，可以是完整名称（如'weights_chi2'）或简写（如'chi2'）

    返回：
        Callable: 对应的权重计算函数

    异常：
        ValueError: 当指定名称的方法不存在时
    """
    # 获取所有权重方法
    all_methods = get_weights_methods()

    # 自动添加'weights_'前缀（如果没有）
    if not name.startswith("weights_"):
        full_name = f"weights_{name}"
    else:
        full_name = name

    # 查找匹配的方法
    for method in all_methods:
        if method.__name__ == full_name:
            return method

    # 如果没有找到，抛出异常
    available_methods = [m.__name__ for m in all_methods]
    short_names = [m.__name__.replace("weights_", "") for m in all_methods]
    raise ValueError(
        f"未找到名为 '{name}' 的权重计算方法。\n可用方法: {', '.join(available_methods)}\n或简写: {', '.join(short_names)}"
    )


def get_weights_methods():
    """
    获取所有权重计算方法

    返回：
        List[Callable]: 权重计算函数列表
    """
    return [
        weights_AdaBoostClassifier,
        weights_ExtraTreesClassifier,
        weights_GradientBoostingClassifier,
        weights_const,
        weights_variance,
        weights_chi2,
        weights_mutual_info_classif,
        weights_FastICA,
        weights_kruskal_classif,
        weights_ttest_ind_classif,
        weights_levene_median,
        weights_mean_var,
        weights_maximum_absolute_deviation,
    ]


if __name__ == "__main__":
    # 如果直接运行此脚本，显示所有可用的权重方法
    print("Available weight calculation methods:")
    for method in get_weights_methods():
        print(f"- {method.__name__}")
