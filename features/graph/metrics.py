#!/usr/bin/env python3
# -*- coding: utf-8-unix -*-

"""
评估信息检索指标计算工具库 (Accuracy@k, MAP, MRR)

本模块实现了常用的信息检索评估指标，用于衡量缺陷文件定位算法的效果:
- Accuracy@k: 在前k个结果中至少包含一个正确文件的bug比例
- MAP (Mean Average Precision): 平均精确率均值，考虑了排序质量
- MRR (Mean Reciprocal Rank): 平均倒数排名，衡量第一个正确结果的位置

使用方法:
    python metrics.py <pickled_dataframe_file>

数据格式要求:
    - 输入数据应为pandas DataFrames格式
    - 每行对应一个文件的评估指标
    - DataFrame应使用二级多重索引 (bug报告ID, 文件SHA)
    - 'result'列包含排名算法计算的得分
    - 'used_in_fix'列包含标记(0或1)表示该文件是否用于修复
"""

import numpy as np
import pandas as pd
import sys
from typing import Callable, Dict, List


def calculate_metrics(
    verification_df: pd.DataFrame, k_range=range(1, 21), metric_type=None
):
    """
    计算信息检索评估指标: Accuracy@k, MAP和MRR

    遍历每个bug报告，计算其检索结果的各项评估指标，并返回全局平均值。

    参数:
        verification_df: 包含预测结果的DataFrame
            - 多级索引: (bug_id, file_sha)
            - 至少包含'result'和'used_in_fix'两列
        k_range: 要计算Accuracy@k的k值范围，默认为1到20
        metric_type: 指定要计算的指标类型，可选值:
            - None: 计算所有指标
            - "ACCURACY": 只计算Accuracy@k
            - "MAP": 只计算MAP
            - "MRR": 只计算MRR

    返回:
        根据metric_type参数，返回不同的结果:
            - None: Tuple[Dict[int, float], float, float]，包含所有三个指标
            - "ACCURACY": Dict[int, float]，包含Accuracy@k的字典
            - "MAP": float，MAP指标值
            - "MRR": float，MRR指标值
    """
    # 收集每个bug报告的数据
    bug_reports_data = []

    # 按bug报告ID分组计算
    for bug_report, bug_report_files_dataframe in verification_df.groupby(
        level=0, sort=False
    ):
        # 找出所有修复文件中得分最低的阈值
        min_fix_result = bug_report_files_dataframe[
            bug_report_files_dataframe["used_in_fix"] == 1.0
        ]["result"].min()

        # 筛选出得分不低于修复文件最低分的文件
        bug_report_files_dataframe2 = bug_report_files_dataframe[
            bug_report_files_dataframe["result"] >= min_fix_result
        ]

        # 按得分降序排序
        sorted_df = bug_report_files_dataframe2.sort_values(
            ascending=False, by=["result"]
        )

        # 如果筛选后为空，则使用所有文件
        if sorted_df.shape[0] == 0:
            sorted_df = bug_report_files_dataframe.copy().sort_values(
                ascending=False, by=["result"]
            )

        # 为文件添加位置序号列
        tmp = sorted_df
        a = range(1, tmp.shape[0] + 1)
        tmp["position"] = pd.Series(a, index=tmp.index)

        # 获取所有修复文件的位置序号
        large_k_p = tmp[(tmp["used_in_fix"] == 1.0)]["position"].tolist()

        # 获取唯一的得分值并排序
        unique_results = sorted_df["result"].unique().tolist()
        unique_results.sort()

        # 存储当前bug报告的处理数据
        bug_data = {
            "sorted_df": sorted_df,
            "large_k_p": large_k_p,
            "unique_results": unique_results,
        }
        bug_reports_data.append(bug_data)

        # 释放内存
        del bug_report, bug_report_files_dataframe

    # 根据指定的指标类型计算结果
    if metric_type == "ACCURACY":
        return _calculate_accuracy_at_k(bug_reports_data, k_range)
    elif metric_type == "MAP":
        return _calculate_map(bug_reports_data)
    elif metric_type == "MRR":
        return _calculate_mrr(bug_reports_data)
    else:
        # 计算所有指标
        accuracy_at_k = _calculate_accuracy_at_k(bug_reports_data, k_range)
        mean_average_precision = _calculate_map(bug_reports_data)
        mean_reciprocal_rank = _calculate_mrr(bug_reports_data)
        return accuracy_at_k, mean_average_precision, mean_reciprocal_rank


def _calculate_accuracy_at_k(
    bug_reports_data: List[Dict], k_range=range(1, 21)
) -> Dict[int, float]:
    """
    计算Accuracy@k指标

    参数:
        bug_reports_data: 预处理后的bug报告数据列表
        k_range: 要计算Accuracy@k的k值范围

    返回:
        Dict[int, float]: 各k值的Accuracy@k
    """
    accuracy_at_k = dict.fromkeys(k_range, 0)  # 初始化各k值的accuracy计数器
    bug_report_number = len(bug_reports_data)  # bug报告总数

    for bug_data in bug_reports_data:
        sorted_df = bug_data["sorted_df"]
        unique_results = bug_data["unique_results"]

        # 计算各k值的Accuracy@k
        for k in k_range:
            # 如果k超出了结果数量，使用所有结果
            if k > len(unique_results):
                k_largest = unique_results
            else:
                k_largest = unique_results[-k:]

            # 获取得分不低于阈值的前k个文件
            largest_at_k = sorted_df[sorted_df["result"] >= min(k_largest)]
            # 计算前k个结果中是否含有正确文件
            real_fixes_at_k = largest_at_k["used_in_fix"][
                (largest_at_k["used_in_fix"] == 1.0)
            ].count()

            # 如果至少有1个正确文件，则accuracy+1
            if real_fixes_at_k >= 1:
                accuracy_at_k[k] += 1

    # 计算各k值的Accuracy@k百分比
    for k in k_range:
        accuracy_at_k[k] = (
            accuracy_at_k[k] / bug_report_number if bug_report_number > 0 else 0.0
        )

    return accuracy_at_k


def _calculate_map(bug_reports_data: List[Dict]) -> float:
    """
    计算MAP(Mean Average Precision)指标

    参数:
        bug_reports_data: 预处理后的bug报告数据列表

    返回:
        float: MAP指标值
    """
    average_precision_per_bug_report = []  # 存储每个bug报告的平均精确率

    for bug_data in bug_reports_data:
        sorted_df = bug_data["sorted_df"]
        large_k_p = bug_data["large_k_p"]
        unique_results = bug_data["unique_results"]

        # 计算每个修复文件位置的精确率
        precision_at_k = []
        for fk in large_k_p:
            k = int(fk)  # 当前修复文件的位置
            k_largest = unique_results[-k:]  # 前k个得分最高的阈值

            # 获取得分不低于阈值的文件
            largest_at_k = sorted_df[sorted_df["result"] >= min(k_largest)]
            # 计算前k个结果中正确文件数量
            real_fixes_at_k = (largest_at_k["used_in_fix"] == 1.0).sum()

            # 精确率 = 正确文件数 / k
            p = float(real_fixes_at_k) / float(k)
            precision_at_k.append(p)

        # 计算平均精确率 (Average Precision)
        # AP = 所有修复文件位置的精确率之和 / 修复文件数量
        if precision_at_k:  # 确保列表非空
            average_precision = pd.Series(precision_at_k).mean()
            average_precision_per_bug_report.append(average_precision)

    # MAP = 所有bug报告AP的均值
    return (
        pd.Series(average_precision_per_bug_report).mean()
        if average_precision_per_bug_report
        else 0.0
    )


def _calculate_mrr(bug_reports_data: List[Dict]) -> float:
    """
    计算MRR(Mean Reciprocal Rank)指标

    参数:
        bug_reports_data: 预处理后的bug报告数据列表

    返回:
        float: MRR指标值
    """
    reciprocal_ranks = []  # 存储每个bug报告的倒数排名

    for bug_data in bug_reports_data:
        sorted_df = bug_data["sorted_df"]

        # 计算倒数排名 (Reciprocal Rank)
        # 找到第一个正确文件的索引
        indexes_of_fixes = np.flatnonzero(sorted_df["used_in_fix"] == 1.0)
        if indexes_of_fixes.size == 0:
            # 如果没有找到正确文件，RR=0
            reciprocal_ranks.append(0.0)
        else:
            # 第一个正确文件的位置(从0开始索引，所以+1)
            first_index = indexes_of_fixes[0]
            # RR = 1/位置
            reciprocal_rank = 1.0 / (first_index + 1)
            reciprocal_ranks.append(reciprocal_rank)

    # MRR = 所有bug报告RR的均值
    return pd.Series(reciprocal_ranks).mean() if reciprocal_ranks else 0.0


def calculate_metric_results(df: pd.DataFrame, k_range=range(1, 21), metric_type=None):
    """
    计算并返回所有评估指标结果或指定的单个指标

    参数:
        df: 包含预测结果的DataFrame
        k_range: 要计算Accuracy@k的k值范围
        metric_type: 指定要计算的指标类型，可选值:
            - None: 计算所有指标
            - "ACCURACY": 只计算Accuracy@k
            - "MAP": 只计算MAP
            - "MRR": 只计算MRR

    返回:
        根据metric_type参数，返回不同的结果:
            - None: Tuple包含所有指标和k_range
            - "ACCURACY": Dict[int, float]，包含Accuracy@k的字典
            - "MAP": float，MAP指标值
            - "MRR": float，MRR指标值
    """
    result = calculate_metrics(df, k_range, metric_type)

    if metric_type is None:
        (
            all_data_accuracy_at_k,
            all_data_mean_average_precision,
            all_data_mean_reciprocal_rank,
        ) = result
        return (
            all_data_accuracy_at_k,
            all_data_mean_average_precision,
            all_data_mean_reciprocal_rank,
            k_range,
        )
    else:
        return result


# region 辅助函数


def fold_check(
    method: Callable[[pd.DataFrame, List[str]], np.ndarray],
    df: pd.DataFrame,
    columns: List[str],
    metric_type: str = "MAP",
):
    """
    单折权重评估函数（非交叉验证模式）

    参数：
        method (function): 权重计算方法
        df (pd.DataFrame): 完整训练数据
        columns (list): 特征列名列表
        metric_type (str): 评估指标类型，可选值:
            - "MAP": 平均精确率
            - "MRR": 平均倒数排名
    """
    weights = method(df, columns)
    Y = np.dot(df[columns], weights)
    return method.__name__, (weights, evaluate_fold(df, Y, metric_type=metric_type))


def eval_weights(
    m_name: str,
    weights: np.ndarray,
    df: pd.DataFrame,
    columns: List[str],
    metric_type: str = "MAP",
):
    """
    权重评估函数（验证阶段）

    参数：
        m_name: 权重方法名称
        weights: 已计算的权重向量
        df: 验证集数据
        columns: 特征列名列表
        metric_type: 评估指标类型，可选值:
            - "MAP": 平均精确率
            - "MRR": 平均倒数排名
    """
    Y = np.dot(df[columns], weights)
    return m_name, (weights, evaluate_fold(df, Y, metric_type=metric_type))


def evaluate_fold(df: pd.DataFrame, Y: np.ndarray, metric_type: str = "MAP"):
    """
    评估预测结果

    参数：
        df: 待评估数据集（需包含used_in_fix列）
        Y: 预测得分向量
        metric_type: 评估指标类型，可选值:
            - "MAP": 平均精确率
            - "MRR": 平均倒数排名

    返回：
        float: 评估指标值（MAP或MRR）

    """
    r = df[["used_in_fix"]].copy(deep=False)
    r["result"] = Y

    if metric_type is None:
        accuracy_at_k, map_score, mrr_score = calculate_metrics(r)
        return {"ACCURACY": accuracy_at_k, "MAP": map_score, "MRR": mrr_score}
    else:
        # 返回单一指标
        return calculate_metric_results(r, metric_type=metric_type)


# endregion


def print_metrics(
    all_data_accuracy_at_k: Dict[int, float],
    all_data_mean_average_precision: float,
    all_data_mean_reciprocal_rank: float,
    k_range: range,
) -> None:
    """
    打印所有评估指标结果

    参数:
        all_data_accuracy_at_k: 各k值的Accuracy@k
        all_data_mean_average_precision: MAP值
        all_data_mean_reciprocal_rank: MRR值
        k_range: 使用的k值范围
    """
    print("All data accuracy at k in k range", k_range)
    for k in k_range:
        print(f"Accuracy@{k}: {all_data_accuracy_at_k[k]:.4f}")
    print(
        f"All data mean average precision (MAP): {all_data_mean_average_precision:.4f}"
    )
    print(f"All data mean reciprocal rank (MRR): {all_data_mean_reciprocal_rank:.4f}")


def main() -> None:
    """
    主函数，从命令行读取数据文件，计算并打印评估指标
    """
    if len(sys.argv) != 2:
        print("用法: python metrics.py <pickled_dataframe_file>")
        sys.exit(1)

    # 读取DataFrame文件
    df_path = sys.argv[1]
    try:
        df = pd.read_pickle(df_path)
        print(f"成功加载数据文件: {df_path}")
        print(f"数据集大小: {df.shape}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        sys.exit(1)

    # 计算评估指标
    metrics_results = calculate_metric_results(df)
    # 打印结果
    print_metrics(*metrics_results)


if __name__ == "__main__":
    main()
