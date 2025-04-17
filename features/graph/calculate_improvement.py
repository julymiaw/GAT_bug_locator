import json
import sys
import os


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
    results, baseline_name="SGDRegressor", improved_name="GAT模型"
):
    """直接生成适合README.md格式的Markdown表格"""

    # 在列表中使用的表格需要缩进
    indent = "    "

    # 1. 基准模型结果表格
    baseline_table = (
        f"{indent}| 属性        | 分数   |     | 属性        | 分数   |\n"
        f"{indent}| ----------- | ------ | --- | ----------- | ------ |\n"
    )

    # 添加Accuracy@1到Accuracy@10在左边
    for k in range(1, 11):
        key = f"Accuracy@{k}"
        value = f"{results[key]['baseline']:.4f}" if key in results else "N/A"

        # 添加Accuracy@11到Accuracy@20在右边
        right_k = k + 10
        right_key = f"Accuracy@{right_k}"
        right_value = (
            f"{results[right_key]['baseline']:.4f}" if right_key in results else "N/A"
        )

        baseline_table += (
            f"{indent}| {key}  | {value} |     | {right_key} | {right_value} |\n"
        )

    # 添加MAP和MRR
    baseline_table += f"{indent}| MAP         | {results['MAP']['baseline']:.4f} |     | MRR         | {results['MRR']['baseline']:.4f} |\n"

    # 2. 改进模型结果表格
    improved_table = (
        f"{indent}| 属性        | 分数   |     | 属性        | 分数   |\n"
        f"{indent}| ----------- | ------ | --- | ----------- | ------ |\n"
    )

    # 添加Accuracy@1到Accuracy@10在左边
    for k in range(1, 11):
        key = f"Accuracy@{k}"
        value = f"{results[key]['improved']:.4f}" if key in results else "N/A"

        # 添加Accuracy@11到Accuracy@20在右边
        right_k = k + 10
        right_key = f"Accuracy@{right_k}"
        right_value = (
            f"{results[right_key]['improved']:.4f}" if right_key in results else "N/A"
        )

        improved_table += (
            f"{indent}| {key}  | {value} |     | {right_key} | {right_value} |\n"
        )

    # 添加MAP和MRR
    improved_table += f"{indent}| MAP         | {results['MAP']['improved']:.4f} |     | MRR         | {results['MRR']['improved']:.4f} |\n"

    # 3. 性能对比表格
    comparison_table = (
        f"{indent}| 模型         | MAP    | MRR    | Accuracy@1 | Accuracy@5 | Accuracy@10 | Accuracy@20 |\n"
        f"{indent}| :----------- | :----- | :----- | :--------- | :--------- | :---------- | :---------- |\n"
        f"{indent}| {baseline_name} | {results['MAP']['baseline']:.4f} | {results['MRR']['baseline']:.4f} | "
        f"{results['Accuracy@1']['baseline']:.4f}     | {results['Accuracy@5']['baseline']:.4f}     | "
        f"{results['Accuracy@10']['baseline']:.4f}      | {results['Accuracy@20']['baseline']:.4f}      |\n"
        f"{indent}| {improved_name}      | {results['MAP']['improved']:.4f} | {results['MRR']['improved']:.4f} | "
        f"{results['Accuracy@1']['improved']:.4f}     | {results['Accuracy@5']['improved']:.4f}     | "
        f"{results['Accuracy@10']['improved']:.4f}      | {results['Accuracy@20']['improved']:.4f}      |\n"
        f"{indent}| 提升百分比   | {results['MAP']['improvement_percent']:.1f}%  | {results['MRR']['improvement_percent']:.1f}%  | "
        f"{results['Accuracy@1']['improvement_percent']:.1f}%      | {results['Accuracy@5']['improvement_percent']:.1f}%      | "
        f"{results['Accuracy@10']['improvement_percent']:.1f}%      | {results['Accuracy@20']['improvement_percent']:.1f}%      |\n"
    )

    return {
        "baseline": baseline_table,
        "improved": improved_table,
        "comparison": comparison_table,
    }


def main():
    # 简化命令行参数，只需要提供两个模型文件夹
    if len(sys.argv) < 3:
        print(
            "用法: python calculate_improvement.py <baseline_folder> <improved_folder> [baseline_name] [improved_name]"
        )
        print(
            "例如: python calculate_improvement.py aspectj_SGD aspectj_GAT [SGDRegressor] [GAT模型]"
        )
        sys.exit(1)

    # 文件夹路径
    baseline_folder = sys.argv[1]
    improved_folder = sys.argv[2]

    # 可选的模型名称参数
    baseline_name = sys.argv[3] if len(sys.argv) > 3 else "SGDRegressor"
    improved_name = sys.argv[4] if len(sys.argv) > 4 else "GAT模型"

    try:
        # 加载指标
        baseline_metrics = load_metrics(baseline_folder)
        improved_metrics = load_metrics(improved_folder)

        # 计算提升
        improvement_results = calculate_improvement(baseline_metrics, improved_metrics)

        # 直接生成Markdown表格
        tables = generate_markdown_tables(
            improvement_results, baseline_name, improved_name
        )

        # 按README.md格式打印结果
        print(f"\n1. 原始模型({baseline_name})结果\n")
        print(tables["baseline"])

        print(f"\n2. {improved_name}结果\n")
        print(tables["improved"])

        print("\n3. 模型性能对比分析\n")
        print(tables["comparison"])

    except Exception as e:
        print(f"处理过程中出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
