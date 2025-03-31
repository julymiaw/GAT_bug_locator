import json
import pandas as pd


def load_metrics(file_path):
    """加载评估指标结果文件"""
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
    acc_results = {}
    for k in [1, 5, 10, 20]:
        if (
            str(k) in baseline_metrics["accuracy_at_k"]
            and str(k) in improved_metrics["accuracy_at_k"]
        ):
            baseline_acc = baseline_metrics["accuracy_at_k"][str(k)]
            improved_acc = improved_metrics["accuracy_at_k"][str(k)]
            improvement = (improved_acc - baseline_acc) / baseline_acc * 100

            acc_results[f"Accuracy@{k}"] = {
                "baseline": baseline_acc,
                "improved": improved_acc,
                "improvement_percent": improvement,
            }

    results.update(acc_results)
    return results


def format_result_table(results):
    """格式化结果为Markdown表格"""
    df = pd.DataFrame(
        {
            "模型": ["SGDRegressor", "GAT模型", "提升百分比"],
            "MAP": [
                f"{results['MAP']['baseline']:.4f}",
                f"{results['MAP']['improved']:.4f}",
                f"{results['MAP']['improvement_percent']:.1f}%",
            ],
            "MRR": [
                f"{results['MRR']['baseline']:.4f}",
                f"{results['MRR']['improved']:.4f}",
                f"{results['MRR']['improvement_percent']:.1f}%",
            ],
        }
    )

    # 添加Accuracy@k列
    for k in [1, 5, 10, 20]:
        key = f"Accuracy@{k}"
        if key in results:
            df[key] = [
                f"{results[key]['baseline']:.4f}",
                f"{results[key]['improved']:.4f}",
                f"{results[key]['improvement_percent']:.1f}%",
            ]

    return df.to_markdown(index=False)


def main():
    # 文件路径
    baseline_file = "aspectj_SGD/Adaptive_metrics_results.json"
    improved_file = "aspectj_GAT/Adaptive_metrics_results.json"

    # 加载指标
    baseline_metrics = load_metrics(baseline_file)
    improved_metrics = load_metrics(improved_file)

    # 计算提升
    improvement_results = calculate_improvement(baseline_metrics, improved_metrics)

    # 打印表格
    print("\n性能提升分析:")
    print(format_result_table(improvement_results))

    # 打印详细结果
    print("\n详细性能指标:")
    for metric, values in improvement_results.items():
        print(f"{metric}:")
        print(f"  基准模型: {values['baseline']:.4f}")
        print(f"  改进模型: {values['improved']:.4f}")
        print(f"  提升百分比: {values['improvement_percent']:.2f}%")
        print()


if __name__ == "__main__":
    main()
