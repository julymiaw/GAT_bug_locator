#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自适应GAT训练评估脚本

用途: 分析自适应训练过程中生成的各种日志文件，提供详细的性能分析和可视化
输入: 训练过程中生成的JSON日志文件
输出: 详细的分析结果、统计数据和可视化图表

作者: SRTP团队
日期: 2025年3月7日
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Any
import argparse
from train_utils import eprint

# Linux下设置
mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体为黑体
mpl.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题


class AdaptiveGATEvaluator:
    """自适应GAT训练评估器，用于分析训练日志并生成详细报告"""

    def __init__(self, log_dir: str):
        """
        初始化评估器

        参数:
            log_dir: 包含训练日志的目录路径
        """
        self.log_dir = log_dir
        self.output_dir = os.path.join(log_dir, "analysis_results")
        os.makedirs(self.output_dir, exist_ok=True)

        # 日志数据
        self.training_time = {}
        self.regression_log = {}
        self.prescoring_log = {}
        self.prediction_log = {}
        self.cv_regression_log = {}
        self.best_regression = {}
        self.best_prescoring = {}

        # 分析结果
        self.model_performance_df = None

    def load_logs(self):
        """加载所有日志文件"""
        print("正在加载日志文件...")

        # 训练时间日志
        try:
            with open(
                os.path.join(self.log_dir, "Adaptive_training_time.json"), "r"
            ) as f:
                self.training_time = json.load(f)
            print("✓ 成功加载训练时间日志")
        except Exception as e:
            print(f"✗ 无法加载训练时间日志: {e}")

        # 回归模型日志
        try:
            with open(
                os.path.join(self.log_dir, "Adaptive_regression_log.json"), "r"
            ) as f:
                self.regression_log = json.load(f)
            print("✓ 成功加载回归模型日志")
        except Exception as e:
            print(f"✗ 无法加载回归模型日志: {e}")

        # 预评分日志
        try:
            with open(
                os.path.join(self.log_dir, "Adaptive_prescoring_log.json"), "r"
            ) as f:
                self.prescoring_log = json.load(f)
            print("✓ 成功加载预评分日志")
        except Exception as e:
            print(f"✗ 无法加载预评分日志: {e}")

        # 预测日志
        try:
            with open(
                os.path.join(self.log_dir, "Adaptive_prediction_log.json"), "r"
            ) as f:
                self.prediction_log = json.load(f)
            print("✓ 成功加载预测日志")
        except Exception as e:
            print(f"✗ 无法加载预测日志: {e}")

        # 交叉验证日志
        try:
            with open(
                os.path.join(self.log_dir, "Adaptive_cv_regression_log.json"), "r"
            ) as f:
                self.cv_regression_log = json.load(f)
            print("✓ 成功加载交叉验证日志")
        except Exception as e:
            print(f"✗ 无法加载交叉验证日志: {e}")

        # 最佳回归模型日志
        try:
            with open(
                os.path.join(self.log_dir, "Adaptive_best_regression_log.json"), "r"
            ) as f:
                self.best_regression = json.load(f)
            print("✓ 成功加载最佳回归模型日志")
        except Exception as e:
            print(f"✗ 无法加载最佳回归模型日志: {e}")

        # 最佳预评分日志
        try:
            with open(
                os.path.join(self.log_dir, "Adaptive_best_prescoring_log.json"), "r"
            ) as f:
                self.best_prescoring = json.load(f)
            print("✓ 成功加载最佳预评分日志")
        except Exception as e:
            print(f"✗ 无法加载最佳预评分日志: {e}")

    def preprocess_data(self):
        """预处理和整合日志数据"""
        # 创建包含所有模型性能数据的DataFrame
        models_data = []

        for fold_num in self.regression_log.keys():
            for model_name, train_score in self.regression_log[fold_num].items():
                # 获取测试分数
                test_score = self.prediction_log.get(fold_num, {}).get(model_name, None)

                # 检查是否为最佳模型
                is_best = model_name == self.best_regression.get(fold_num, None)

                # 解析模型名称以提取参数
                parsed_params = self.parse_model_name(model_name)

                # 合并所有数据
                model_info = {
                    "fold": fold_num,
                    "model_name": model_name,
                    "train_score": train_score,
                    "test_score": test_score,
                    "is_best": is_best,
                    **parsed_params,
                }

                models_data.append(model_info)

        # 创建DataFrame
        self.model_performance_df = pd.DataFrame(models_data)
        print(f"✓ 成功处理 {len(self.model_performance_df)} 个模型配置的数据")

    def parse_model_name(self, model_name: str) -> Dict[str, Any]:
        """
        解析模型名称，提取参数信息

        参数:
            model_name: 模型名称字符串

        返回:
            包含参数信息的字典
        """
        params = {}
        parts = model_name.split("_")

        # 基本信息
        params["model_type"] = parts[0]
        params["loss_type"] = parts[1]
        params["hidden_dim"] = int(parts[2])
        params["penalty"] = parts[3]
        params["heads"] = parts[4]
        if parts[4] == "nohead":
            params["model_category"] = "baseline_mlp"

        # 解析其他超参数
        i = 5
        while i < len(parts):
            if parts[i].startswith("a"):
                params["alpha"] = float(parts[i].replace("a", ""))
            elif parts[i].startswith("dr"):
                params["dropout"] = float(parts[i].replace("dr", ""))
            elif parts[i].startswith("lr"):
                params["learning_rate"] = float(parts[i].replace("lr", ""))
            elif parts[i] == "selfloop":
                params["use_self_loops"] = True
            elif parts[i] == "shuf":
                params["shuffle"] = True
            i += 1

        # 设置默认值
        params.setdefault("use_self_loops", False)
        params.setdefault("shuffle", False)

        if "model_category" not in params:
            if params["use_self_loops"]:
                params["model_category"] = "gat_selfloop"  # GAT+自环模型
            else:
                params["model_category"] = "gat_realedge"  # GAT+真实边模型

        return params

    def analyze_overall_performance(self):
        """分析整体性能并生成摘要"""
        # 获取最佳模型
        best_models = self.model_performance_df[self.model_performance_df["is_best"]]

        # 平均性能分析
        avg_train = self.model_performance_df["train_score"].mean()
        avg_test = self.model_performance_df["test_score"].mean()

        # 生成摘要报告
        summary = {
            "总模型数": len(self.model_performance_df),
            "平均训练得分": avg_train,
            "平均测试得分": avg_test,
            "最佳模型": best_models["model_name"].tolist(),
            "最佳模型训练得分": best_models["train_score"].tolist(),
            "最佳模型测试得分": best_models["test_score"].tolist(),
        }

        # 保存摘要
        with open(os.path.join(self.output_dir, "performance_summary.json"), "w") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        # 打印摘要
        print("\n===== 性能摘要 =====")
        print(f"总模型数: {summary['总模型数']}")
        print(f"平均训练得分: {summary['平均训练得分']:.4f}")
        print(f"平均测试得分: {summary['平均测试得分']:.4f}")

        for i, model in enumerate(summary["最佳模型"]):
            print(f"\n最佳模型 {i+1}: {model}")
            print(f"  训练得分: {summary['最佳模型训练得分'][i]:.4f}")
            print(f"  测试得分: {summary['最佳模型测试得分'][i]:.4f}")

        return summary

    def analyze_parameter_impact(self):
        """分析不同参数对模型性能的影响"""
        # 要分析的参数列表
        params_to_analyze = [
            "loss_type",
            "hidden_dim",
            "penalty",
            "learning_rate",
            "alpha",
            "dropout",
        ]

        # 创建参数影响分析目录
        param_dir = os.path.join(self.output_dir, "parameter_impact")
        os.makedirs(param_dir, exist_ok=True)

        # 分析每个参数
        param_impact = {}

        for param in params_to_analyze:
            if param not in self.model_performance_df.columns:
                continue

            unique_values = self.model_performance_df[param].unique()
            if len(unique_values) <= 1:
                continue

            # 计算每个参数值的平均性能
            impact_data = []
            unique_values = self.model_performance_df[param].unique()

            for value in unique_values:
                subset = self.model_performance_df[
                    self.model_performance_df[param] == value
                ]

                impact_data.append(
                    {
                        "param_value": value,
                        "count": len(subset),
                        "mean_train": subset["train_score"].mean(),
                        "mean_test": subset["test_score"].mean(),
                        "best_count": subset["is_best"].sum(),
                    }
                )

            # 保存参数影响数据
            param_impact[param] = impact_data

            # 创建参数影响图表
            self.plot_parameter_impact(param, impact_data, param_dir)

        # 保存参数影响摘要
        with open(
            os.path.join(self.output_dir, "parameter_impact_summary.json"), "w"
        ) as f:
            json.dump(param_impact, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        return param_impact

    def plot_parameter_impact(
        self, param: str, impact_data: List[Dict], output_dir: str
    ):
        """
        绘制参数影响图表

        参数:
            param: 参数名称
            impact_data: 影响数据列表
            output_dir: 输出目录
        """
        # 转换为DataFrame
        impact_df = pd.DataFrame(impact_data)
        impact_df["param_label"] = impact_df["param_value"].astype(str)

        # 排序
        if param in ["hidden_dim", "learning_rate", "alpha", "dropout"]:
            impact_df = impact_df.sort_values(by="param_value")

        # 绘制过拟合对比图
        plt.figure(figsize=(12, 6))

        x = np.arange(len(impact_df))
        width = 0.35

        # 训练和测试分数
        plt.bar(x - width / 2, impact_df["mean_train"], width, label="训练得分")
        plt.bar(x + width / 2, impact_df["mean_test"], width, label="测试得分")

        # 添加标签
        plt.xlabel(param)
        plt.ylabel("平均得分(MAP)")
        plt.title(f"{param}参数对训练和测试性能的对比")
        plt.xticks(x, impact_df["param_label"])
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{param}_train_test_compare.png"))
        plt.close()

    def analyze_self_loops_effect(self):
        """分析自环模式对模型性能的影响"""
        # 检查是否有自环相关的数据
        if "use_self_loops" not in self.model_performance_df.columns:
            print("⚠ 数据中没有自环模式相关信息，跳过分析")
            return None

        # 获取使用自环和不使用自环的模型子集
        self_loop_models = self.model_performance_df[
            self.model_performance_df["use_self_loops"] == True
        ]
        real_edge_models = self.model_performance_df[
            self.model_performance_df["use_self_loops"] == False
        ]

        # 配对分析：查找具有相同配置但自环模式不同的模型对
        model_pairs = []

        for _, self_loop_model in self_loop_models.iterrows():
            # 查找相同配置但不使用自环的模型
            matching_models = real_edge_models[
                (real_edge_models["loss_type"] == self_loop_model["loss_type"])
                & (real_edge_models["hidden_dim"] == self_loop_model["hidden_dim"])
                & (real_edge_models["penalty"] == self_loop_model["penalty"])
                & (real_edge_models["alpha"] == self_loop_model["alpha"])
                & (real_edge_models["dropout"] == self_loop_model["dropout"])
                & (
                    real_edge_models["learning_rate"]
                    == self_loop_model["learning_rate"]
                )
                & (real_edge_models["heads"] == self_loop_model["heads"])
            ]

            if not matching_models.empty:
                real_edge_model = matching_models.iloc[0]

                model_pairs.append(
                    {
                        "self_loop_model": self_loop_model["model_name"],
                        "real_edge_model": real_edge_model["model_name"],
                        "self_loop_train": self_loop_model["train_score"],
                        "real_edge_train": real_edge_model["train_score"],
                        "self_loop_test": self_loop_model["test_score"],
                        "real_edge_test": real_edge_model["test_score"],
                        "test_improvement": self_loop_model["test_score"]
                        - real_edge_model["test_score"],
                    }
                )

        # 计算总体统计
        self_loop_stats = {
            "总体比较": {
                "平均训练得分 (自环)": self_loop_models["train_score"].mean(),
                "平均训练得分 (真实边)": real_edge_models["train_score"].mean(),
                "平均测试得分 (自环)": self_loop_models["test_score"].mean(),
                "平均测试得分 (真实边)": real_edge_models["test_score"].mean(),
            },
            "模型对比较": model_pairs,
        }

        # 保存自环分析结果
        with open(os.path.join(self.output_dir, "self_loops_analysis.json"), "w") as f:
            json.dump(
                self_loop_stats, f, indent=4, ensure_ascii=False, cls=NumpyEncoder
            )

        # 绘制自环效果图表
        self.plot_self_loops_effect(model_pairs)

        # 打印摘要
        print("\n===== 自环模式效果摘要 =====")
        print(f"自环模型数量: {len(self_loop_models)}")
        print(f"真实边模型数量: {len(real_edge_models)}")
        print(f"配对比较模型数: {len(model_pairs)}")
        print(
            f"自环模型平均测试得分: {self_loop_stats['总体比较']['平均测试得分 (自环)']:.4f}"
        )
        print(
            f"真实边模型平均测试得分: {self_loop_stats['总体比较']['平均测试得分 (真实边)']:.4f}"
        )

        return self_loop_stats

    def plot_self_loops_effect(self, model_pairs: List[Dict]):
        """
        绘制自环效果的图表

        参数:
            model_pairs: 包含配对模型信息的字典列表
        """
        if not model_pairs:
            return

        # 转换为DataFrame
        pairs_df = pd.DataFrame(model_pairs)

        # 自环与真实边模型的测试性能比较
        plt.figure(figsize=(10, 6))

        # 按性能提升排序
        sorted_pairs = pairs_df.sort_values(by="test_improvement", ascending=False)

        plt.bar(
            range(len(sorted_pairs)), sorted_pairs["self_loop_test"], label="自环模型"
        )
        plt.bar(
            range(len(sorted_pairs)),
            sorted_pairs["real_edge_test"],
            label="真实边模型",
            alpha=0.6,
        )

        plt.xlabel("模型对索引")
        plt.ylabel("测试得分(MAP)")
        plt.title("自环模型vs真实边模型测试性能对比")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "self_loop_vs_real_edge_test.png"))
        plt.close()

    def analyze_training_time(self):
        """分析训练时间"""
        if not self.training_time:
            eprint("⚠ 没有训练时间数据，跳过分析")
            return None

        # 计算平均每bug和每文件的训练时间
        time_stats = {}

        for fold, data in self.training_time.items():
            time_stats[fold] = {
                "total_time_seconds": data["time"],
                "total_time_minutes": data["time"] / 60,
                "total_time_hours": data["time"] / 3600,
                "bugs_count": data["bugs"],
                "files_count": data["files"],
                "seconds_per_bug": (
                    data["time"] / data["bugs"] if data["bugs"] > 0 else 0
                ),
                "seconds_per_file": (
                    data["time"] / data["files"] if data["files"] > 0 else 0
                ),
            }

        # 保存结果
        with open(
            os.path.join(self.output_dir, "training_time_analysis.json"), "w"
        ) as f:
            json.dump(time_stats, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        # 打印时间统计
        print("\n===== 训练时间分析 =====")
        for fold, stats in time_stats.items():
            print(f"折 {fold}:")
            print(
                f"  总训练时间: {stats['total_time_hours']:.2f}小时 ({stats['total_time_minutes']:.2f}分钟)"
            )
            print(f"  Bug数量: {stats['bugs_count']}")
            print(f"  文件数量: {stats['files_count']}")
            print(f"  平均每bug训练时间: {stats['seconds_per_bug']:.2f}秒")
            print(f"  平均每文件训练时间: {stats['seconds_per_file']:.2f}秒")

        return time_stats

    def generate_final_report(self):
        """生成最终综合报告"""

        # 获取最佳模型详情
        best_models = self.model_performance_df[self.model_performance_df["is_best"]]
        best_model_category = best_models["model_category"].iloc[0]

        # 按模型类别分组计算平均性能
        category_performance = (
            self.model_performance_df.groupby("model_category")["test_score"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

        # 准备报告内容
        report = {
            "总结": {
                "评估时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "评估目录": self.log_dir,
                "模型总数": len(self.model_performance_df),
                "最佳模型": best_models["model_name"].tolist(),
                "最佳模型测试得分": best_models["test_score"].tolist(),
                "最佳模型类型": {
                    "baseline_mlp": "基线MLP模型",
                    "gat_realedge": "GAT模型",
                    "gat_selfloop": "仅自环GAT模型",
                }.get(best_model_category, "未知类型"),
            },
            "模型类别性能": category_performance.to_dict(orient="records"),
            "主要发现": [
                (
                    (
                        "仅自环GAT模型优于GAT模型"
                        if category_performance[
                            category_performance["model_category"] == "gat_selfloop"
                        ]["mean"].iloc[0]
                        > category_performance[
                            category_performance["model_category"] == "gat_realedge"
                        ]["mean"].iloc[0]
                        else "GAT模型优于仅自环GAT模型"
                    )
                    if "gat_selfloop" in category_performance["model_category"].values
                    and "gat_realedge" in category_performance["model_category"].values
                    else "未比较自环与真实边"
                ),
                (
                    (
                        "MLP模型表现优于GAT模型"
                        if category_performance[
                            category_performance["model_category"] == "baseline_mlp"
                        ]["mean"].iloc[0]
                        > category_performance[
                            category_performance["model_category"] == "gat_realedge"
                        ]["mean"].iloc[0]
                        else "GAT模型表现优于MLP模型"
                    )
                    if "baseline_mlp" in category_performance["model_category"].values
                    and "gat_realedge" in category_performance["model_category"].values
                    else "未比较GAT与MLP"
                ),
            ],
            "推荐参数设置": {
                param: best_models[param].iloc[0]
                for param in [
                    "loss_type",
                    "hidden_dim",
                    "penalty",
                    "learning_rate",
                    "dropout",
                    "alpha",
                ]
                if param in best_models.columns
            },
        }

        # 保存最终报告
        with open(os.path.join(self.output_dir, "final_report.json"), "w") as f:
            json.dump(report, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        # 创建可读性更好的文本报告
        report_text = f"""
=========================================
自适应GAT训练评估报告
=========================================
评估时间: {report['总结']['评估时间']}
评估目录: {report['总结']['评估目录']}
模型总数: {report['总结']['模型总数']}

最佳模型: {report['总结']['最佳模型'][0]}
最佳模型类型: {report['总结']['最佳模型类型']}
最佳模型测试得分: {report['总结']['最佳模型测试得分'][0]:.4f}

主要发现:
{chr(10).join('- ' + finding for finding in report['主要发现'])}

推荐参数设置:
{chr(10).join('- ' + param + ': ' + str(value) for param, value in report['推荐参数设置'].items())}
=========================================
        """

        with open(
            os.path.join(self.output_dir, "final_report.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(report_text)

        print("\n" + report_text)

        return report

    def run_all_analyses(self):
        """运行所有分析"""
        # 加载日志数据
        self.load_logs()

        # 预处理数据
        self.preprocess_data()

        # 运行各种分析
        self.analyze_overall_performance()
        self.analyze_parameter_impact()
        self.analyze_self_loops_effect()
        self.analyze_training_time()

        # 生成最终报告
        self.generate_final_report()

        print(f"\n✓ 全部分析完成！结果保存在: {self.output_dir}")


def main():
    """主程序入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="分析自适应GAT训练日志并生成详细报告")
    parser.add_argument("log_dir", help="包含训练日志文件的目录路径")
    parser.add_argument(
        "--output", "-o", help="输出目录 (默认为log_dir/analysis_results)"
    )

    args = parser.parse_args()

    # 创建评估器并运行分析
    evaluator = AdaptiveGATEvaluator(args.log_dir)

    if args.output:
        evaluator.output_dir = args.output
        os.makedirs(evaluator.output_dir, exist_ok=True)

    evaluator.run_all_analyses()


class NumpyEncoder(json.JSONEncoder):
    """处理NumPy类型的JSON编码器"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):  # 添加对NumPy布尔类型的处理
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    main()
