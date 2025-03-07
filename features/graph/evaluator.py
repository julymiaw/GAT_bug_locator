#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自适应GAT训练评估脚本

用途: 分析自适应训练过程中生成的各种日志文件，提供详细的性能分析和可视化
输入: 训练过程中生成的JSON日志文件
输出: 详细的分析结果、统计数据和可视化图表

作者: [您的姓名]
日期: 2025年3月7日
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from typing import Dict, List, Any, Tuple
import argparse

# 设置中文字体，避免显示问题
# Windows下设置
# mpl.rc("font", family='Microsoft YaHei')

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
        print("正在处理数据...")

        # 创建包含所有模型性能数据的DataFrame
        models_data = []

        for fold_num in self.regression_log.keys():
            for model_name, train_score in self.regression_log[fold_num].items():
                # 获取测试分数
                test_score = self.prediction_log.get(fold_num, {}).get(model_name, None)

                # 获取交叉验证分数
                cv_scores = self.cv_regression_log.get(fold_num, {}).get(model_name, [])
                cv_mean = np.mean(cv_scores) if cv_scores else None
                cv_std = np.std(cv_scores) if cv_scores else None

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
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "is_best": is_best,
                    "overfitting": (
                        train_score - test_score if test_score is not None else None
                    ),
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

        # 解析头部信息
        i = 4
        heads = []
        while i < len(parts) and not (
            parts[i].startswith("a")
            or parts[i].startswith("dr")
            or parts[i].startswith("lr")
        ):
            if parts[i] == "nohead":
                params["is_gat"] = False
                params["model_category"] = "baseline_mlp"
                i += 1
                break
            heads.append(int(parts[i]))
            i += 1

        params["heads"] = heads
        params["is_gat"] = len(heads) > 0
        params["num_layers"] = len(heads)

        # 解析其他超参数
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

        # 分配更精确的模型类别
        if "model_category" not in params:
            if params["is_gat"]:
                if params["use_self_loops"]:
                    params["model_category"] = "gat_selfloop"  # GAT+自环模型
                else:
                    params["model_category"] = "gat_realedge"  # GAT+真实边模型
            else:
                params["model_category"] = "baseline_mlp"  # 基线MLP模型

        return params

    def analyze_overall_performance(self):
        """分析整体性能并生成摘要"""
        print("正在分析整体性能...")

        # 获取最佳模型
        best_models = self.model_performance_df[self.model_performance_df["is_best"]]

        # 平均性能分析
        avg_train = self.model_performance_df["train_score"].mean()
        avg_test = self.model_performance_df["test_score"].mean()
        avg_cv = self.model_performance_df["cv_mean"].mean()
        avg_overfit = self.model_performance_df["overfitting"].mean()

        # 生成摘要报告
        summary = {
            "总模型数": len(self.model_performance_df),
            "平均训练得分": avg_train,
            "平均测试得分": avg_test,
            "平均交叉验证得分": avg_cv,
            "平均过拟合程度": avg_overfit,
            "最佳模型": best_models["model_name"].tolist(),
            "最佳模型训练得分": best_models["train_score"].tolist(),
            "最佳模型测试得分": best_models["test_score"].tolist(),
            "最佳模型交叉验证得分": best_models["cv_mean"].tolist(),
        }

        # 保存摘要
        with open(os.path.join(self.output_dir, "performance_summary.json"), "w") as f:
            json.dump(summary, f, indent=4, cls=NumpyEncoder)

        # 打印摘要
        print("\n===== 性能摘要 =====")
        print(f"总模型数: {summary['总模型数']}")
        print(f"平均训练得分: {summary['平均训练得分']:.4f}")
        print(f"平均测试得分: {summary['平均测试得分']:.4f}")
        print(f"平均交叉验证得分: {summary['平均交叉验证得分']:.4f}")
        print(f"平均过拟合程度: {summary['平均过拟合程度']:.4f}")

        for i, model in enumerate(summary["最佳模型"]):
            print(f"\n最佳模型 {i+1}: {model}")
            print(f"  训练得分: {summary['最佳模型训练得分'][i]:.4f}")
            print(f"  测试得分: {summary['最佳模型测试得分'][i]:.4f}")
            print(f"  交叉验证得分: {summary['最佳模型交叉验证得分'][i]:.4f}")

        return summary

    def analyze_parameter_impact(self):
        """分析不同参数对模型性能的影响"""
        print("正在分析参数影响...")

        # 要分析的参数列表
        params_to_analyze = [
            "loss_type",
            "hidden_dim",
            "penalty",
            "learning_rate",
            "alpha",
            "dropout",
            "is_gat",
            "num_layers",
            "use_self_loops",
        ]

        # 创建参数影响分析目录
        param_dir = os.path.join(self.output_dir, "parameter_impact")
        os.makedirs(param_dir, exist_ok=True)

        # 分析每个参数
        param_impact = {}

        for param in params_to_analyze:
            if param not in self.model_performance_df.columns:
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
                        "mean_cv": subset["cv_mean"].mean(),
                        "mean_overfit": subset["overfitting"].mean(),
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
            json.dump(param_impact, f, indent=4, cls=NumpyEncoder)

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

        # 特殊处理某些参数的标签
        if param == "is_gat":
            impact_df["param_label"] = impact_df["param_value"].map(
                {True: "GAT", False: "MLP"}
            )
        elif param == "use_self_loops":
            impact_df["param_label"] = impact_df["param_value"].map(
                {True: "自环", False: "真实边"}
            )
        else:
            impact_df["param_label"] = impact_df["param_value"].astype(str)

        # 排序
        if param in ["hidden_dim", "learning_rate", "alpha", "dropout", "num_layers"]:
            impact_df = impact_df.sort_values(by="param_value")

        # 绘制测试性能柱状图
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x="param_label",
            y="mean_test",
            data=impact_df,
            hue="param_label",
            palette="viridis",
            legend=False,
        )

        # 添加值标签
        for i, row in enumerate(impact_df.itertuples()):
            ax.text(
                i,
                row.mean_test + 0.01,
                f"{row.mean_test:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.title(f"{param}参数对测试性能的影响")
        plt.xlabel(param)
        plt.ylabel("平均测试得分(MAP)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{param}_test_impact.png"))
        plt.close()

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
        print("正在分析自环模式的效果...")

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
                & (
                    real_edge_models["heads"].apply(lambda x: str(x))
                    == str(self_loop_model["heads"])
                )
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
                        "self_loop_cv": self_loop_model["cv_mean"],
                        "real_edge_cv": real_edge_model["cv_mean"],
                        "test_improvement": self_loop_model["test_score"]
                        - real_edge_model["test_score"],
                        "is_gat": self_loop_model["is_gat"],
                    }
                )

        # 计算总体统计
        self_loop_stats = {
            "总体比较": {
                "平均训练得分 (自环)": self_loop_models["train_score"].mean(),
                "平均训练得分 (真实边)": real_edge_models["train_score"].mean(),
                "平均测试得分 (自环)": self_loop_models["test_score"].mean(),
                "平均测试得分 (真实边)": real_edge_models["test_score"].mean(),
                "平均交叉验证得分 (自环)": self_loop_models["cv_mean"].mean(),
                "平均交叉验证得分 (真实边)": real_edge_models["cv_mean"].mean(),
                "测试性能平均提升": self_loop_models["test_score"].mean()
                - real_edge_models["test_score"].mean(),
            },
            "模型对比较": model_pairs,
        }

        # 保存自环分析结果
        with open(os.path.join(self.output_dir, "self_loops_analysis.json"), "w") as f:
            json.dump(self_loop_stats, f, indent=4, cls=NumpyEncoder)

        # 绘制自环效果图表
        self.plot_self_loops_effect(model_pairs)

        # 打印摘要
        print("\n===== 自环模式效果摘要 =====")
        print(f"自环模型数量: {len(self_loop_models)}")
        print(f"真实边模型数量: {len(real_edge_models)}")
        print(f"配对比较模型数: {len(model_pairs)}")
        print(
            f"平均测试性能提升: {self_loop_stats['总体比较']['测试性能平均提升']:.4f}"
        )
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

        # 创建自环分析目录
        self_loop_dir = os.path.join(self.output_dir, "self_loops_analysis")
        os.makedirs(self_loop_dir, exist_ok=True)

        # 转换为DataFrame
        pairs_df = pd.DataFrame(model_pairs)

        # 1. 自环与真实边模型的测试性能比较
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
        plt.savefig(os.path.join(self_loop_dir, "self_loop_vs_real_edge_test.png"))
        plt.close()

        # 2. 测试性能提升分布
        plt.figure(figsize=(10, 6))
        plt.hist(
            pairs_df["test_improvement"],
            bins=10,
            color="teal",
            edgecolor="black",
            alpha=0.7,
        )
        plt.axvline(x=0, color="red", linestyle="--", label="零提升线")
        plt.axvline(
            x=pairs_df["test_improvement"].mean(),
            color="green",
            linestyle="-",
            label=f"平均提升: {pairs_df['test_improvement'].mean():.4f}",
        )

        plt.xlabel("测试性能提升 (自环 - 真实边)")
        plt.ylabel("模型对数量")
        plt.title("自环模式带来的测试性能提升分布")
        plt.legend()
        plt.grid(axis="both", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self_loop_dir, "self_loop_improvement_distribution.png")
        )
        plt.close()

        # 3. GAT模型与MLP模型的自环效果比较
        if "is_gat" in pairs_df.columns and pairs_df["is_gat"].nunique() > 1:
            plt.figure(figsize=(12, 6))

            gat_pairs = pairs_df[pairs_df["is_gat"] == True]
            mlp_pairs = pairs_df[pairs_df["is_gat"] == False]

            data = [gat_pairs["test_improvement"], mlp_pairs["test_improvement"]]

            plt.boxplot(data, tick_labels=["GAT模型", "MLP模型"])
            plt.axhline(y=0, color="red", linestyle="--", label="零提升线")

            plt.xlabel("模型类型")
            plt.ylabel("自环模式带来的测试性能提升")
            plt.title("GAT与MLP模型的自环效果比较")
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self_loop_dir, "gat_vs_mlp_self_loop_effect.png"))
            plt.close()

    def analyze_cv_consistency(self):
        """分析交叉验证的一致性"""
        print("正在分析交叉验证一致性...")

        # 检查是否有交叉验证数据
        if "cv_mean" not in self.model_performance_df.columns:
            print("⚠ 数据中没有交叉验证信息，跳过分析")
            return None

        # 计算交叉验证与测试集性能的相关性
        cv_test_corr = (
            self.model_performance_df[["cv_mean", "test_score"]].corr().iloc[0, 1]
        )

        # 获取一些代表性模型进行可视化
        top_models = self.model_performance_df.sort_values(
            by="test_score", ascending=False
        ).head(5)
        worst_models = self.model_performance_df.sort_values(by="test_score").head(5)
        selected_models = pd.concat([top_models, worst_models])

        # 创建交叉验证分析目录
        cv_dir = os.path.join(self.output_dir, "cv_analysis")
        os.makedirs(cv_dir, exist_ok=True)

        # 绘制CV vs 测试集散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.model_performance_df["cv_mean"],
            self.model_performance_df["test_score"],
            alpha=0.7,
            label="所有模型",
        )

        # 标注特殊模型
        best_model = self.model_performance_df[
            self.model_performance_df["is_best"]
        ].iloc[0]
        plt.scatter(
            best_model["cv_mean"],
            best_model["test_score"],
            color="red",
            s=100,
            marker="*",
            label="最佳模型",
        )

        # 添加对角线参考
        max_val = max(
            self.model_performance_df["cv_mean"].max(),
            self.model_performance_df["test_score"].max(),
        )
        min_val = min(
            self.model_performance_df["cv_mean"].min(),
            self.model_performance_df["test_score"].min(),
        )
        plt.plot([min_val, max_val], [min_val, max_val], "k--", label="理想情况")

        plt.xlabel("交叉验证平均得分")
        plt.ylabel("测试集得分")
        plt.title("交叉验证与测试集性能相关性")
        plt.legend()
        plt.annotate(
            f"相关系数: {cv_test_corr:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(cv_dir, "cv_test_correlation.png"))
        plt.close()

        # 绘制训练/CV/测试性能比较
        plt.figure(figsize=(14, 8))

        # 获取可视化的模型
        models_to_plot = selected_models.sort_values("test_score")
        model_names = [f"模型{i+1}" for i in range(len(models_to_plot))]

        x = np.arange(len(models_to_plot))
        width = 0.25

        plt.bar(x - width, models_to_plot["train_score"], width, label="训练得分")
        plt.bar(x, models_to_plot["cv_mean"], width, label="交叉验证得分")
        plt.bar(x + width, models_to_plot["test_score"], width, label="测试得分")

        plt.xlabel("模型")
        plt.ylabel("得分(MAP)")
        plt.title("训练/交叉验证/测试性能对比")
        plt.xticks(x, model_names)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # 添加注释：哪些是表现最好的模型，哪些是表现最差的模型
        plt.annotate(
            "最佳模型",
            xy=(len(top_models) - 1 + 0.5, 0.05),
            xytext=(0, 30),
            textcoords="offset points",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.2),
            arrowprops=dict(arrowstyle="->"),
        )

        plt.annotate(
            "最差模型",
            xy=(0, 0.05),
            xytext=(0, 30),
            textcoords="offset points",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.2),
            arrowprops=dict(arrowstyle="->"),
        )

        plt.tight_layout()
        plt.savefig(os.path.join(cv_dir, "train_cv_test_comparison.png"))
        plt.close()

        # 返回分析结果
        return {
            "cv_test_correlation": cv_test_corr,
            "best_model_cv_test_diff": best_model["cv_mean"] - best_model["test_score"],
            "avg_cv_test_diff": self.model_performance_df["cv_mean"].mean()
            - self.model_performance_df["test_score"].mean(),
        }

    def analyze_prescoring_methods(self):
        """分析预评分方法的性能"""
        print("正在分析预评分方法...")

        if not self.prescoring_log:
            print("⚠ 没有预评分方法的日志数据，跳过分析")
            return None

        # 准备预评分方法数据
        prescoring_data = []

        for fold_num in self.prescoring_log.keys():
            for method_name, score in self.prescoring_log[fold_num].items():
                # 检查是否为最佳方法
                is_best = method_name == self.best_prescoring.get(fold_num, None)

                # 提取方法族和具体算法
                method_parts = method_name.split("_")
                method_family = method_parts[0] if len(method_parts) > 0 else "unknown"
                method_algo = (
                    "_".join(method_parts[1:]) if len(method_parts) > 1 else "unknown"
                )

                prescoring_data.append(
                    {
                        "fold": fold_num,
                        "method_name": method_name,
                        "method_family": method_family,
                        "method_algo": method_algo,
                        "score": score,
                        "is_best": is_best,
                    }
                )

        # 创建DataFrame
        prescoring_df = pd.DataFrame(prescoring_data)

        # 创建预评分方法分析目录
        prescoring_dir = os.path.join(self.output_dir, "prescoring_analysis")
        os.makedirs(prescoring_dir, exist_ok=True)

        # 按方法族分组计算平均性能
        family_performance = (
            prescoring_df.groupby("method_family")["score"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

        # 绘制方法族性能比较
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            x="method_family",
            y="mean",
            hue="method_family",
            data=family_performance,
            palette="viridis",
            legend=False,
        )

        # 添加值标签
        for i, row in enumerate(family_performance.itertuples()):
            ax.text(
                i,
                row.mean + 0.01,
                f"{row.mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.title("不同预评分方法族的平均性能")
        plt.xlabel("方法族")
        plt.ylabel("平均得分(MAP)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(prescoring_dir, "prescoring_family_comparison.png"))
        plt.close()

        # 绘制Top 10方法性能
        top_methods = prescoring_df.sort_values(by="score", ascending=False).head(10)

        plt.figure(figsize=(14, 6))
        ax = sns.barplot(
            x="method_name",
            y="score",
            hue="method_name",
            data=top_methods,
            palette="viridis",
            legend=False,
        )
        plt.xticks(rotation=45, ha="right")

        # 标注最佳方法
        for i, row in enumerate(top_methods.itertuples()):
            color = "red" if row.is_best else "black"
            weight = "bold" if row.is_best else "normal"
            ax.text(
                i,
                row.score + 0.01,
                f"{row.score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                weight=weight,
            )

        plt.title("Top 10预评分方法性能")
        plt.xlabel("方法名称")
        plt.ylabel("得分(MAP)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(prescoring_dir, "top10_prescoring_methods.png"))
        plt.close()

        # 保存结果
        prescoring_analysis = {
            "method_families": family_performance.to_dict(orient="records"),
            "top_methods": top_methods[["method_name", "score", "is_best"]].to_dict(
                orient="records"
            ),
            "best_method": prescoring_df[prescoring_df["is_best"]][
                ["method_name", "score"]
            ].to_dict(orient="records"),
        }

        with open(os.path.join(prescoring_dir, "prescoring_analysis.json"), "w") as f:
            json.dump(prescoring_analysis, f, indent=4, cls=NumpyEncoder)

        return prescoring_analysis

    def analyze_training_time(self):
        """分析训练时间"""
        print("正在分析训练时间...")

        if not self.training_time:
            print("⚠ 没有训练时间数据，跳过分析")
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

        # 创建训练时间分析目录
        time_dir = os.path.join(self.output_dir, "training_time")
        os.makedirs(time_dir, exist_ok=True)

        # 保存结果
        with open(os.path.join(time_dir, "training_time_analysis.json"), "w") as f:
            json.dump(time_stats, f, indent=4, cls=NumpyEncoder)

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
        print("正在生成最终报告...")

        # 获取最佳模型详情
        best_models = self.model_performance_df[self.model_performance_df["is_best"]]
        best_model_category = best_models["model_category"].iloc[0]

        # 按模型类别分组计算平均性能
        category_performance = (
            self.model_performance_df.groupby("model_category")["test_score"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

        # 判断创新是否成功
        innovation_successful = best_model_category != "baseline_mlp"

        # 准备报告内容
        report = {
            "总结": {
                "评估时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "评估目录": self.log_dir,
                "模型总数": len(self.model_performance_df),
                "最佳模型": best_models["model_name"].tolist(),
                "最佳模型测试得分": best_models["test_score"].tolist(),
                "最佳模型类型": {
                    "baseline_mlp": "基线MLP模型(无图结构)",
                    "gat_realedge": "GAT模型(真实边)",
                    "gat_selfloop": "GAT模型(自环)",
                }.get(best_model_category, "未知类型"),
                "创新是否成功": innovation_successful,
            },
            "模型类别性能": category_performance.to_dict(orient="records"),
            "主要发现": [
                (
                    "图结构创新成功，优于基线MLP模型"
                    if innovation_successful
                    else "图结构创新未能超越基线MLP模型"
                ),
                (
                    (
                        "自环模式GAT优于真实边GAT"
                        if category_performance[
                            category_performance["model_category"] == "gat_selfloop"
                        ]["mean"].iloc[0]
                        > category_performance[
                            category_performance["model_category"] == "gat_realedge"
                        ]["mean"].iloc[0]
                        else "真实边GAT优于自环模式GAT"
                    )
                    if "gat_selfloop" in category_performance["model_category"].values
                    and "gat_realedge" in category_performance["model_category"].values
                    else "未比较自环与真实边"
                ),
                (
                    "MLP模型表现优于GAT模型"
                    if self.model_performance_df[~self.model_performance_df["is_gat"]][
                        "test_score"
                    ].mean()
                    > self.model_performance_df[self.model_performance_df["is_gat"]][
                        "test_score"
                    ].mean()
                    else "GAT模型表现优于MLP模型"
                ),
                (
                    "交叉验证结果与测试集表现一致"
                    if (
                        "cv_mean" in self.model_performance_df.columns
                        and self.model_performance_df[["cv_mean", "test_score"]]
                        .corr()
                        .iloc[0, 1]
                        > 0.7
                    )
                    else "交叉验证结果与测试集表现相关性较低"
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
            json.dump(report, f, indent=4, cls=NumpyEncoder)

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
        print("\n开始运行全面分析...")

        # 加载日志数据
        self.load_logs()

        # 预处理数据
        self.preprocess_data()

        # 运行各种分析
        self.analyze_overall_performance()
        self.analyze_parameter_impact()
        self.analyze_self_loops_effect()
        self.analyze_cv_consistency()
        self.analyze_prescoring_methods()
        self.analyze_training_time()

        # 生成最终报告
        self.generate_final_report()

        print(f"\n✓ 全部分析完成！结果保存在: {self.output_dir}")


def main():
    """主程序入口"""
    # 解析命令行参数
    # parser = argparse.ArgumentParser(description="分析自适应GAT训练日志并生成详细报告")
    # parser.add_argument("log_dir", help="包含训练日志文件的目录路径")
    # parser.add_argument(
    #     "--output", "-o", help="输出目录 (默认为log_dir/analysis_results)"
    # )

    # args = parser.parse_args()、

    args = argparse.Namespace(
        log_dir="./aspectj_20250306230343/",
        output="./aspectj_20250306230343/analysis_results",
    )

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
