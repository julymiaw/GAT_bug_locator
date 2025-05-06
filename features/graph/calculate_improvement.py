#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算模型性能提升百分比，并生成适合README的Markdown表格
同时显示MAP和MRR优化的结果
"""

import json
import sys
import os
import glob


def load_metrics(folder_path):
    """从指定文件夹加载评估指标结果文件"""
    file_path = os.path.join(folder_path, "Adaptive_metrics_results.json")
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_improvement(baseline_metrics, improved_metrics):
    """计算性能提升百分比"""
    results = {}

    # 计算MAP和MRR的提升
    baseline_map = baseline_metrics["mean_average_precision"]
    improved_map = improved_metrics["mean_average_precision"]
    map_improvement = (improved_map - baseline_map) / baseline_map * 100

    baseline_mrr = baseline_metrics["mean_reciprocal_rank"]
    improved_mrr = improved_metrics["mean_reciprocal_rank"]
    mrr_improvement = (improved_mrr - baseline_mrr) / baseline_mrr * 100

    results["MAP"] = {
        "baseline": baseline_map,
        "improved": improved_map,
        "improvement_percent": map_improvement,
    }

    results["MRR"] = {
        "baseline": baseline_mrr,
        "improved": improved_mrr,
        "improvement_percent": mrr_improvement,
    }

    # 计算Accuracy@k的提升
    for k in range(1, 21):  # 收集所有Accuracy@1到Accuracy@20
        str_k = str(k)
        if (
            str_k in baseline_metrics["accuracy_at_k"]
            and str_k in improved_metrics["accuracy_at_k"]
        ):
            baseline_acc = baseline_metrics["accuracy_at_k"][str_k]
            improved_acc = improved_metrics["accuracy_at_k"][str_k]
            improvement = (improved_acc - baseline_acc) / baseline_acc * 100

            results[f"Accuracy@{k}"] = {
                "baseline": baseline_acc,
                "improved": improved_acc,
                "improvement_percent": improvement,
            }

    return results


def generate_markdown_tables(
    results,
    baseline_name="SGDRegressor",
    improved_name="GAT模型",
    metric_type: str = "MAP",
):
    """生成Markdown表格"""

    # 1. 基准模型结果表格
    baseline_table = (
        "| 属性        | 分数   |     | 属性        | 分数   |\n"
        "| ----------- | ------ | --- | ----------- | ------ |\n"
    )

    # 添加Accuracy@1到Accuracy@10在左边
    for k in range(1, 11):
        key = f"Accuracy@{k}"
        value = (
            f"{results[key]['baseline']:.4f}"
            if key in results and "baseline" in results[key]
            else "N/A"
        )

        # 添加Accuracy@11到Accuracy@20在右边
        right_k = k + 10
        right_key = f"Accuracy@{right_k}"
        right_value = (
            f"{results[right_key]['baseline']:.4f}"
            if right_key in results and "baseline" in results[right_key]
            else "N/A"
        )

        baseline_table += f"| {key}  | {value} |     | {right_key} | {right_value} |\n"

    # 添加MAP和MRR
    baseline_table += f"| MAP         | {results['MAP']['baseline']:.4f} |     | MRR         | {results['MRR']['baseline']:.4f} |\n"

    # 检查是否有改进数据，如果只有基准数据则提前返回
    if "improved" not in results.get("MAP", {}):
        return {"baseline": baseline_table}

    # 2. 改进模型结果表格
    improved_table = (
        "| 属性        | 分数   |     | 属性        | 分数   |\n"
        "| ----------- | ------ | --- | ----------- | ------ |\n"
    )

    # 添加Accuracy@1到Accuracy@10在左边
    for k in range(1, 11):
        key = f"Accuracy@{k}"
        value = (
            f"{results[key]['improved']:.4f}"
            if key in results and "improved" in results[key]
            else "N/A"
        )

        # 添加Accuracy@11到Accuracy@20在右边
        right_k = k + 10
        right_key = f"Accuracy@{right_k}"
        right_value = (
            f"{results[right_key]['improved']:.4f}"
            if right_key in results and "improved" in results[right_key]
            else "N/A"
        )

        improved_table += f"| {key}  | {value} |     | {right_key} | {right_value} |\n"

    # 添加MAP和MRR
    improved_table += f"| MAP         | {results['MAP']['improved']:.4f} |     | MRR         | {results['MRR']['improved']:.4f} |\n"

    # 3. 性能对比表格
    model_name = f"{improved_name} ({metric_type})" if metric_type else improved_name
    comparison_table = (
        f"| 模型         | MAP    | MRR    | Accuracy@1 | Accuracy@5 | Accuracy@10 | Accuracy@20 |\n"
        f"| :----------- | :----- | :----- | :--------- | :--------- | :---------- | :---------- |\n"
        f"| {baseline_name} | {results['MAP']['baseline']:.4f} | {results['MRR']['baseline']:.4f} | "
        f"{results['Accuracy@1']['baseline']:.4f}     | {results['Accuracy@5']['baseline']:.4f}     | "
        f"{results['Accuracy@10']['baseline']:.4f}      | {results['Accuracy@20']['baseline']:.4f}      |\n"
        f"| {model_name}      | {results['MAP']['improved']:.4f} | {results['MRR']['improved']:.4f} | "
        f"{results['Accuracy@1']['improved']:.4f}     | {results['Accuracy@5']['improved']:.4f}     | "
        f"{results['Accuracy@10']['improved']:.4f}      | {results['Accuracy@20']['improved']:.4f}      |\n"
        f"| 提升百分比   | {results['MAP']['improvement_percent']:.1f}%  | {results['MRR']['improvement_percent']:.1f}%  | "
        f"{results['Accuracy@1']['improvement_percent']:.1f}%      | {results['Accuracy@5']['improvement_percent']:.1f}%      | "
        f"{results['Accuracy@10']['improvement_percent']:.1f}%      | {results['Accuracy@20']['improvement_percent']:.1f}%      |\n"
    )

    return {
        "baseline": baseline_table,
        "improved": improved_table,
        "comparison": comparison_table,
    }


def find_gat_result_folder(dataset_prefix: str, metric_type: str) -> str:
    """查找指定数据集和指标类型的GAT结果文件夹"""
    # 直接使用固定命名格式
    folder_name = f"{dataset_prefix}_GAT_{metric_type}"

    if os.path.exists(folder_name):
        return folder_name

    # 如果找不到精确匹配的文件夹，尝试查找包含时间戳的文件夹
    pattern = f"{dataset_prefix}_GAT_{metric_type}_*"
    folders = glob.glob(pattern)

    if not folders:
        raise ValueError(f"错误: 找不到匹配 '{folder_name}' 或 '{pattern}' 的文件夹")

    # 返回第一个匹配的文件夹
    return folders[0]


def process_metric_type(dataset_prefix, metric_type, baseline_folder):
    """处理指定指标类型的结果"""
    try:
        # 查找GAT结果文件夹
        gat_folder = find_gat_result_folder(dataset_prefix, metric_type)

        # 加载指标
        baseline_metrics = load_metrics(baseline_folder)
        improved_metrics = load_metrics(gat_folder)

        # 计算提升
        improvement_results = calculate_improvement(baseline_metrics, improved_metrics)

        # 生成Markdown表格
        tables = generate_markdown_tables(
            improvement_results,
            baseline_name="SGDRegressor",
            improved_name="GAT模型",
            metric_type=metric_type,
        )

        # 准备标题和介绍
        title_improved = f"#### {metric_type}优化的GAT模型\n\n"
        title_comparison = f"#### 性能对比分析 - {metric_type}优化\n\n"

        # 返回结果
        return {
            "improved": f"{title_improved}{tables['improved']}",
            "comparison": f"{title_comparison}{tables['comparison']}",
        }
    except ValueError as e:
        print(f"警告: 处理{metric_type}结果时出错: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("用法: python calculate_improvement.py <dataset_prefix>")
        print("例如: python calculate_improvement.py aspectj")
        sys.exit(1)

    dataset_prefix = sys.argv[1]

    # 基准文件夹
    baseline_folder = f"{dataset_prefix}_SGD"

    try:
        # 先加载基准指标，检查是否可用
        baseline_metrics = load_metrics(baseline_folder)

        # 准备标题和基准模型部分
        title_dataset = f"### {dataset_prefix.capitalize()} 数据集\n"

        # 生成基准模型表格
        results_placeholder = {
            "MAP": {"baseline": baseline_metrics["mean_average_precision"]},
            "MRR": {"baseline": baseline_metrics["mean_reciprocal_rank"]},
        }

        # 添加Accuracy@k
        for k in range(1, 21):
            str_k = str(k)
            if str_k in baseline_metrics["accuracy_at_k"]:
                results_placeholder[f"Accuracy@{k}"] = {
                    "baseline": baseline_metrics["accuracy_at_k"][str_k]
                }

        baseline_table = generate_markdown_tables(results_placeholder)["baseline"]
        title_baseline = f"#### 基准模型 (SGDRegressor)\n\n"
        baseline_section = f"{title_baseline}{baseline_table}"

        # 输出数据集标题
        print(f"{title_dataset}")

        # 输出基准模型部分
        print(baseline_section)

        # 处理MAP优化结果
        map_results = process_metric_type(dataset_prefix, "MAP", baseline_folder)
        if map_results:
            print(map_results["improved"])
            print(map_results["comparison"])

        # 处理MRR优化结果
        mrr_results = process_metric_type(dataset_prefix, "MRR", baseline_folder)
        if mrr_results:
            print(mrr_results["improved"])
            print(mrr_results["comparison"])

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
