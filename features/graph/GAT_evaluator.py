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
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import argparse
from GATRegressor import ModelParameters
from experiment_tracker import ExperimentTracker

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
        self.experiment_tracker = None
        self.model_performance_df = None

        # 模型配对关系(自环模型 -> 真实边模型)
        self.paired_models = {}

    def load_logs(self):
        """加载所有日志文件"""
        print("正在加载日志文件...")

        try:
            self.experiment_tracker = ExperimentTracker()
            self.experiment_tracker.from_json(
                os.path.join(self.log_dir, "Adaptive_models_registry.json")
            )
            print("✓ 成功加载实验跟踪管理器模块")
        except Exception as e:
            print(f"✗ 无法加载实验跟踪管理器模块: {e}")
            return False

        return True

    def preprocess_data(self):
        """从实验跟踪管理器模块中提取并处理数据"""
        if not self.experiment_tracker:
            print("⚠ 实验跟踪管理器模块未加载，无法处理数据")
            return {}

        # 获取DataFrame
        df = self.experiment_tracker.to_dataframe()

        # 添加model_category字段用于分类
        def determine_category(row):
            model_id = row["model_id"]
            if model_id.startswith("weights_method"):
                return "baseline_weights"
            elif row.get("model_type") == "MLP":
                return "baseline_mlp"
            elif row.get("model_type") == "GAT":
                return (
                    "gat_selfloop"
                    if row.get("use_self_loops_only", False)
                    else "gat_realedge"
                )
            return "unknown"

        df["model_category"] = df.apply(determine_category, axis=1)

        # 构建模型配对关系
        self._build_model_pairs(df)

        # 保存数据框
        self.model_performance_df = df

        print(f"✓ 成功处理 {len(self.model_performance_df)} 个模型配置的数据")

        return True

    def _build_model_pairs(self, df):
        """找出自环模型和真实边模型的配对关系，建立单向映射（自环模型 -> 真实边模型）"""
        # 按fold分组
        for fold_num in df["fold_num"].unique():
            self.paired_models[fold_num] = {}

            # 获取当前fold的所有模型ID
            fold_models = df[df["fold_num"] == fold_num]
            selfloop_models = fold_models[
                fold_models["model_category"] == "gat_selfloop"
            ]
            realedge_models = fold_models[
                fold_models["model_category"] == "gat_realedge"
            ]

            # 检查每个自环模型，寻找匹配的真实边模型
            for _, selfloop_model in selfloop_models.iterrows():
                selfloop_id = selfloop_model["model_id"]

                # 寻找匹配的真实边模型
                for _, realedge_model in realedge_models.iterrows():
                    realedge_id = realedge_model["model_id"]

                    # 检查两个模型是否匹配
                    if self.experiment_tracker.are_models_matching(
                        selfloop_id, realedge_id
                    ):
                        # 建立从真实边模型到自环模型的映射
                        self.paired_models[fold_num][selfloop_id] = realedge_id
                        break

    def analyze_overall_performance(self):
        """分析整体性能并生成摘要"""
        score_types = ["MAP", "MRR"]

        overall_summary = {
            "总模型数": self.experiment_tracker.model_counter,
        }

        # 在DataFrame中添加'is_best_MAP'和'is_best_MRR'列用于标记最佳模型
        self.model_performance_df["is_best_MAP"] = False
        self.model_performance_df["is_best_MRR"] = False

        for score_type in score_types:
            # 确定对应的列名
            train_score_col = f"train_{score_type}_score"
            test_score_col = f"predict_{score_type}_score"
            is_best_col = f"is_best_{score_type}"

            # 动态计算最佳模型（每个fold中train_score最高的非权重模型）
            best_models = []

            for fold in self.model_performance_df["fold_num"].unique():
                fold_models = self.model_performance_df[
                    (self.model_performance_df["fold_num"] == fold)
                    & (
                        self.model_performance_df["model_category"].isin(
                            ["gat_selfloop", "gat_realedge"]
                        )
                    )
                ]

                if not fold_models.empty and train_score_col in fold_models.columns:
                    best_idx = fold_models[train_score_col].idxmax()
                    self.model_performance_df.loc[best_idx, is_best_col] = True
                    best_models.append(fold_models.loc[best_idx])

            # 生成摘要报告
            best_df = pd.DataFrame(best_models)

            # 生成摘要报告
            overall_summary[f"{score_type}最佳模型"] = {
                "fold": best_df["fold_num"].tolist(),
                "model_id": best_df["model_id"].tolist(),
                f"训练得分": best_df[train_score_col].tolist(),
                f"测试得分": best_df[test_score_col].tolist(),
            }

            # 打印摘要
            print(f"\n===== {score_type}性能摘要 =====")
            print(f"总模型数: {overall_summary['总模型数']}")

            for i, model in enumerate(
                overall_summary[f"{score_type}最佳模型"]["model_id"]
            ):
                fold = overall_summary[f"{score_type}最佳模型"]["fold"][i]
                train_score = overall_summary[f"{score_type}最佳模型"]["训练得分"][i]
                test_score = overall_summary[f"{score_type}最佳模型"]["测试得分"][i]

                print(f"\n折 {fold}:")
                print(f"  最佳模型: {model}")
                print(f"  训练{score_type}: {train_score:.4f}")
                print(f"  测试{score_type}: {test_score:.4f}")

        # 保存合并后的摘要到一个文件
        json_path = os.path.join(self.output_dir, "performance_summary.json")
        pd.Series(overall_summary).to_json(
            json_path, indent=4, orient="index", force_ascii=False
        )

        return overall_summary

    def analyze_parameter_impact(self):
        """按fold分组分析不同参数对模型性能的影响"""
        # 要分析的参数列表
        params_to_analyze = [
            "penalty",
            "dropout",
            "use_self_loops_only",
            "alpha",
            "lr",
        ]

        # 参数组合
        param_combinations = [
            ("heads", "hidden_dim"),
            ("min_score_ratio", "n_iter_no_change"),
        ]

        # 创建参数影响分析目录
        param_dir = os.path.join(self.output_dir, "parameter_impact")
        os.makedirs(param_dir, exist_ok=True)

        # 分析每个参数
        nn_models_df = self.model_performance_df[
            self.model_performance_df["model_category"] != "baseline_weights"
        ]

        # 获取所有折号
        folds = nn_models_df["fold_num"].unique()

        score_types = ["MAP", "MRR"]
        all_results = {"single_parameters": {}, "parameter_combinations": {}}

        for score_type in score_types:
            train_score_col = f"train_{score_type}_score"
            test_score_col = f"predict_{score_type}_score"

            # 确保这些列存在
            if (
                train_score_col not in nn_models_df.columns
                or test_score_col not in nn_models_df.columns
            ):
                print(f"⚠ 警告: {score_type}分数列不存在，跳过分析")
                continue

            # 1. 单参数分析（按fold分组）
            all_results["single_parameters"][score_type] = {}

            for param in params_to_analyze:
                if param not in nn_models_df.columns:
                    print(f"⚠ 警告：参数 {param} 不在数据集列中，跳过分析")
                    continue

                # 为每个fold创建单独的impact_data
                fold_impact_data = {}
                for fold in folds:
                    # 筛选当前fold的模型
                    fold_models = nn_models_df[nn_models_df["fold_num"] == fold]

                    unique_values = fold_models[param].unique()
                    if len(unique_values) <= 1:
                        continue

                    # 计算每个参数值的平均性能
                    impact_data = []

                    for value in unique_values:
                        if not pd.isna(value):
                            # 精确匹配非NaN值
                            subset = fold_models[fold_models[param] == value]

                            if not subset.empty:
                                impact_data.append(
                                    {
                                        "param_value": value,
                                        "count": len(subset),
                                        "mean_train": subset[train_score_col].mean(),
                                        "mean_test": subset[test_score_col].mean(),
                                        "fold": fold,  # 添加fold信息
                                    }
                                )

                    # 特殊处理NaN值
                    nan_subset = fold_models[pd.isna(fold_models[param])]
                    if not nan_subset.empty:
                        impact_data.append(
                            {
                                "param_value": np.nan,
                                "count": len(nan_subset),
                                "mean_train": nan_subset[train_score_col].mean(),
                                "mean_test": nan_subset[test_score_col].mean(),
                                "fold": fold,
                            }
                        )

                    # 保存影响数据
                    if impact_data:
                        fold_impact_data[fold] = impact_data

                # 合并所有fold的数据以保存总体结果
                all_fold_data = []
                for fold_data in fold_impact_data.values():
                    all_fold_data.extend(fold_data)

                if all_fold_data:
                    all_results["single_parameters"][score_type][param] = all_fold_data
                    # 绘制按fold分组的参数影响
                    self.plot_parameter_impact(
                        param, fold_impact_data, param_dir, score_type=score_type
                    )
                else:
                    print(f"⚠ 警告：参数 {param} 没有有效数据可分析")

            # 2. 参数组合分析（同样按fold分组）
            all_results["parameter_combinations"][score_type] = {}

            for param1, param2 in param_combinations:
                if (
                    param1 not in nn_models_df.columns
                    or param2 not in nn_models_df.columns
                ):
                    print(
                        f"⚠ 警告：参数组合 {param1}-{param2} 中有参数不在数据集列中，跳过分析"
                    )
                    continue

                # 按fold分组进行组合分析
                fold_combo_data = {}

                for fold in folds:
                    # 筛选当前fold的模型
                    fold_models = nn_models_df[nn_models_df["fold_num"] == fold]

                    # 获取每个参数的唯一值
                    unique_values1 = [
                        v for v in fold_models[param1].unique() if not pd.isna(v)
                    ]
                    unique_values2 = [
                        v for v in fold_models[param2].unique() if not pd.isna(v)
                    ]

                    if len(unique_values1) <= 1 and len(unique_values2) <= 1:
                        continue

                    # 为此组合创建数据
                    combo_data = []
                    for val1 in unique_values1:
                        for val2 in unique_values2:
                            # 找到具有此参数组合的模型
                            combo_subset = fold_models[
                                (fold_models[param1] == val1)
                                & (fold_models[param2] == val2)
                            ]

                            if not combo_subset.empty:
                                combo_data.append(
                                    {
                                        f"{param1}": val1,
                                        f"{param2}": val2,
                                        "combo_label": f"{val1}-{val2}",
                                        "count": len(combo_subset),
                                        "mean_train": combo_subset[
                                            train_score_col
                                        ].mean(),
                                        "mean_test": combo_subset[
                                            test_score_col
                                        ].mean(),
                                        "fold": fold,
                                    }
                                )

                    if combo_data:
                        fold_combo_data[fold] = combo_data

                # 合并所有fold的数据以保存总体结果
                all_fold_combo = []
                for fold_data in fold_combo_data.values():
                    all_fold_combo.extend(fold_data)

                if all_fold_combo:
                    combo_key = f"{param1}_{param2}"
                    all_results["parameter_combinations"][score_type][
                        combo_key
                    ] = all_fold_combo
                    # 绘制按fold分组的参数组合影响
                    self.plot_parameter_combination(
                        param1,
                        param2,
                        fold_combo_data,
                        param_dir,
                        score_type=score_type,
                    )
                else:
                    print(f"⚠ 警告：参数组合 {param1}-{param2} 没有有效数据可分析")

        # 保存参数影响摘要
        json_path = os.path.join(self.output_dir, "parameter_impact_summary.json")
        pd.Series(all_results).to_json(
            json_path, indent=4, orient="index", force_ascii=False
        )

        return all_results

    def plot_parameter_impact(
        self,
        param: str,
        fold_impact_data: Dict[int, List[Dict]],
        output_dir: str,
        score_type="MAP",
    ):
        """
        绘制按fold分组的参数影响图表

        参数:
            param: 参数名称
            fold_impact_data: 按fold分组的影响数据字典
            output_dir: 输出目录
            score_type: 评分类型 (MAP或MRR)
        """
        # 检查数据
        if not fold_impact_data:
            return

        # 创建多折对比子图
        folds = list(fold_impact_data.keys())
        fig, axs = plt.subplots(
            len(folds), 1, figsize=(12, 5 * len(folds)), sharex=True
        )

        # 如果只有一个fold，确保axs是列表
        if len(folds) == 1:
            axs = [axs]

        # 遍历每个fold绘制子图
        for i, fold in enumerate(folds):
            impact_df = pd.DataFrame(fold_impact_data[fold])
            impact_df["param_label"] = impact_df["param_value"].astype(str)
            impact_df = impact_df.sort_values(by="param_value")

            x = np.arange(len(impact_df))
            width = 0.35

            # 训练和测试分数
            axs[i].bar(
                x - width / 2,
                impact_df["mean_train"],
                width,
                label=f"训练{score_type}",
                color="skyblue",
            )
            axs[i].bar(
                x + width / 2,
                impact_df["mean_test"],
                width,
                label=f"测试{score_type}",
                color="salmon",
            )

            # 添加标签
            axs[i].set_ylabel(f"{score_type}得分")
            axs[i].set_title(f"折 {fold}: {param}参数对{score_type}性能的影响")
            axs[i].set_xticks(x)
            axs[i].set_xticklabels(impact_df["param_label"])
            axs[i].grid(axis="y", linestyle="--", alpha=0.7)
            axs[i].legend()

        # 共享的x轴标签
        fig.text(0.5, 0.04, param, ha="center", fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)

        # 保存图片
        plt.savefig(os.path.join(output_dir, f"{param}_{score_type}.png"))
        plt.close()

    def plot_parameter_combination(
        self,
        param1: str,
        param2: str,
        fold_combo_data: Dict[int, List[Dict]],
        output_dir: str,
        score_type="MAP",
    ):
        """
        绘制按fold分组的参数组合柱状图

        参数:
            param1: 第一个参数名称
            param2: 第二个参数名称
            fold_combo_data: 按fold分组的组合分析数据字典
            output_dir: 输出目录
            score_type: 评分类型 (MAP或MRR)
        """
        if not fold_combo_data:
            return

        # 创建多折对比子图
        folds = list(fold_combo_data.keys())
        fig, axs = plt.subplots(
            len(folds), 1, figsize=(15, 6 * len(folds)), sharex=True
        )

        # 如果只有一个fold，确保axs是列表
        if len(folds) == 1:
            axs = [axs]

        # 遍历每个fold绘制子图
        for i, fold in enumerate(folds):
            combo_df = pd.DataFrame(fold_combo_data[fold])
            combo_df["combined_label"] = combo_df.apply(
                lambda row: f"{row[param1]}-{row[param2]}", axis=1
            )

            # 按测试分数降序排序
            combo_df = combo_df.sort_values(by="mean_test", ascending=False)

            x = np.arange(len(combo_df))
            width = 0.35

            # 绘制训练和测试分数
            axs[i].bar(
                x - width / 2,
                combo_df["mean_train"],
                width,
                label=f"训练{score_type}",
                color="skyblue",
            )
            axs[i].bar(
                x + width / 2,
                combo_df["mean_test"],
                width,
                label=f"测试{score_type}",
                color="salmon",
            )

            # 添加标签
            axs[i].set_ylabel(f"{score_type}得分")
            axs[i].set_title(
                f"折 {fold}: {param1}和{param2}参数组合对{score_type}性能的影响"
            )
            axs[i].set_xticks(x)
            axs[i].set_xticklabels(combo_df["combined_label"], rotation=45, ha="right")
            axs[i].grid(axis="y", linestyle="--", alpha=0.7)
            axs[i].legend()

        # 共享的x轴标签
        fig.text(0.5, 0.02, f"{param1} - {param2} 组合", ha="center", fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        # 保存图片
        plt.savefig(
            os.path.join(
                output_dir, f"{param1}_{param2}_{score_type}_by_fold_bar_chart.png"
            )
        )
        plt.close()

    def analyze_self_loops_effect(self):
        """分析自环模式对模型性能的影响"""
        score_types = ["MAP", "MRR"]
        all_results = {}

        for score_type in score_types:
            train_score_col = f"train_{score_type}_score"
            test_score_col = f"predict_{score_type}_score"

            # 检查评分列是否存在
            if (
                train_score_col not in self.model_performance_df.columns
                or test_score_col not in self.model_performance_df.columns
            ):
                print(f"⚠ 警告: {score_type}分数列不存在，跳过自环分析")
                continue

            # 初始化结果结构
            self_loop_stats = {}

            # 按fold分析
            for fold in self.model_performance_df["fold_num"].unique():
                fold_models = self.model_performance_df[
                    self.model_performance_df["fold_num"] == fold
                ]
                fold_pairs = []
                paired_map = self.paired_models.get(fold, {})

                # 遍历该fold的所有自环模型配对
                for selfloop_id, realedge_id in paired_map.items():
                    # 获取自环模型数据
                    selfloop_model = fold_models[fold_models["model_id"] == selfloop_id]
                    if selfloop_model.empty:
                        continue

                    # 获取对应的真实边模型数据
                    realedge_model = fold_models[fold_models["model_id"] == realedge_id]
                    if realedge_model.empty:
                        continue

                    pair_info = {
                        "self_loop_model": selfloop_id,
                        "real_edge_model": realedge_id,
                        f"self_loop_train_{score_type}": selfloop_model[
                            train_score_col
                        ].iloc[0],
                        f"real_edge_train_{score_type}": realedge_model[
                            train_score_col
                        ].iloc[0],
                        f"self_loop_test_{score_type}": selfloop_model[
                            test_score_col
                        ].iloc[0],
                        f"real_edge_test_{score_type}": realedge_model[
                            test_score_col
                        ].iloc[0],
                        f"real_edge_improvement_{score_type}": (
                            realedge_model[test_score_col].iloc[0]
                            - selfloop_model[test_score_col].iloc[0]
                        ),
                    }
                    fold_pairs.append(pair_info)

                # 对结果按真实边模型对比自环模型的提升量降序排列
                fold_pairs.sort(
                    key=lambda x: x[f"real_edge_improvement_{score_type}"], reverse=True
                )
                self_loop_stats[fold] = fold_pairs

            all_results[score_type] = self_loop_stats

        json_path = os.path.join(self.output_dir, "self_loops_analysis.json")
        pd.Series(all_results).to_json(
            json_path, indent=4, orient="index", force_ascii=False
        )

        return all_results

    def analyze_training_time(self):
        """分析训练时间"""

        # 计算平均每bug和每文件的训练时间
        training_time = self.experiment_tracker.training_time_stats
        time_stats = {}

        for fold, data in training_time.items():
            time_stats[fold] = {
                "total_time_seconds": data["time"],
                "total_time_minutes": data["time"] / 60,
                "total_time_hours": data["time"] / 3600,
                "bugs_count": data["bugs"],
                "files_count": data["files"],
                "seconds_per_bug": data["seconds_per_bug"],
                "seconds_per_file": data["seconds_per_file"],
            }

        # 保存结果
        json_path = os.path.join(self.output_dir, "training_time_analysis.json")
        pd.Series(time_stats).to_json(
            json_path, indent=4, orient="index", force_ascii=False
        )

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

    def analyze_model_fitting(self):
        """分析模型的过拟合/欠拟合情况，以权重法为基准"""
        for score_type in ["MAP", "MRR"]:
            train_score_col = f"train_{score_type}_score"
            test_score_col = f"predict_{score_type}_score"
            fitting_status_col = f"fitting_status_{score_type}"

            # 获取每个fold中权重方法的分数
            weights_scores = {}
            for fold in self.model_performance_df["fold_num"].unique():
                fold_weights = self.model_performance_df[
                    (self.model_performance_df["fold_num"] == fold)
                    & (
                        self.model_performance_df["model_category"]
                        == "baseline_weights"
                    )
                ]

                train_score = fold_weights[train_score_col].iloc[0]
                test_score = fold_weights[test_score_col].iloc[0]
                weights_scores[fold] = {"train": train_score, "test": test_score}

            # 筛选神经网络模型
            nn_models_df = self.model_performance_df[
                self.model_performance_df["model_category"] != "baseline_weights"
            ]

            # 计算每个模型相对于权重方法的拟合情况
            fitting_status_map = {}

            for _, model in nn_models_df.iterrows():
                fold = model["fold_num"]

                # 计算与权重方法的差异
                train_diff = model[train_score_col] - weights_scores[fold]["train"]
                test_diff = model[test_score_col] - weights_scores[fold]["test"]

                # 定义拟合状态
                fitting_status = "正常拟合"  # 默认状态

                if train_diff < -0.01:  # 回归分数比权重法低1%以上
                    fitting_status = "欠拟合"
                elif (
                    train_diff > 0.01 and test_diff < -0.02
                ):  # 回归分数高但测试分数比权重法低2%以上
                    fitting_status = "过拟合"

                fitting_status_map[model["model_id"]] = fitting_status

            # 将拟合状态信息更新到主DataFrame中
            for model_id, status in fitting_status_map.items():
                self.model_performance_df.loc[
                    self.model_performance_df["model_id"] == model_id,
                    fitting_status_col,
                ] = status

    def _create_hover_text(self, row, include_params=True, include_scores=True):
        """
        为交互式图表创建悬停文本

        参数:
            row: DataFrame中的一行，包含模型信息
            include_params: 是否包含模型参数，默认为True
            include_scores: 是否包含分数信息，默认为True

        返回:
            str: 格式化的HTML悬停文本
        """
        # 模型类别名称映射
        category_names = {
            "baseline_mlp": "MLP模型",
            "gat_selfloop": "GAT自环模型",
            "gat_realedge": "GAT真实边模型",
            "baseline_weights": "权重法",
        }

        # 收集模型的基本信息
        params = [f"<b>{row['model_id']}</b>"]
        params.append(
            f"类别: {category_names.get(row['model_category'], row['model_category'])}"
        )
        params.append(f"折: {row['fold_num']}")

        is_best_col = f"is_best_{row['metric_type']}"
        if is_best_col in row and row[is_best_col]:
            params.append(f"<b style='color:gold'>★ 最佳模型 ★</b>")

        # 添加评分信息
        if include_scores:
            for score_type in ["MAP", "MRR"]:
                train_score_col = f"train_{score_type}_score"
                test_score_col = f"predict_{score_type}_score"
                params.append(f"训练{score_type}: {row[train_score_col]:.4f}")
                params.append(f"测试{score_type}: {row[test_score_col]:.4f}")

            # 添加拟合状态
            for score_type in ["MAP", "MRR"]:
                fitting_status_col = f"fitting_status_{score_type}"
                if fitting_status_col in row and not pd.isna(row[fitting_status_col]):
                    params.append(f"{score_type}拟合状态: {row[fitting_status_col]}")

        # 添加训练信息
        if row["model_category"] != "baseline_weights":
            params.append(f"停止原因: {row['stop_reason']}")
            params.append(f"最佳轮次: {row['best_epoch']}")
            params.append(f"总轮次: {row['final_epoch']}")

        # 添加模型参数
        if include_params:
            model_params = ModelParameters.get_all_params()
            params.append("<b>模型参数:</b>")
            for param in model_params:
                if param in row and not pd.isna(row[param]):
                    params.append(f"{param}: {row[param]}")

        return "<br>".join(params)

    def _create_interactive_filter_buttons(
        self, fig: go.Figure, folds, category_names, simple_mode=False
    ):
        """
        为交互式图表创建过滤按钮及设置相关布局，使三个筛选器形成"与"逻辑关系

        参数:
            fig: plotly图表对象
            folds: 折号列表
            category_names: 类别名称映射字典
            simple_mode: 布尔值，若为True，则仅显示fold按钮（用于模型对比图）

        返回:
            配置好按钮和布局的图表对象
        """
        model_nums = set()

        if not simple_mode:
            for i, trace in enumerate(fig.data):
                # 提取模型编号
                if trace.customdata:
                    model_num = trace.customdata[2]
                    model_nums.add(model_num)

        # 创建折按钮
        fold_buttons = []
        for fold in folds:
            fold_buttons.append(dict(method=None, label=f"折 {fold}", args=[{}, {}]))

        # 添加"显示全部折"按钮
        fold_buttons.append(dict(method=None, label="全部折", args=[{}, {}]))

        updatemenus = []

        # 在简化模式下，只添加折按钮
        if simple_mode:
            updatemenus = [
                # 只添加折按钮，并将其放置在中央
                dict(
                    type="dropdown",  # 下拉菜单
                    direction="down",
                    active=-1,  # 默认选中最后一项（全部）
                    x=0.5,
                    y=1.15,
                    xanchor="center",
                    yanchor="top",
                    buttons=fold_buttons,
                    name="foldMenu",
                    showactive=True,
                    bgcolor="lightgray",
                )
            ]
        else:
            # 创建模型类型按钮
            category_buttons = []
            for _, category_name in category_names.items():
                category_buttons.append(
                    dict(
                        method=None,  # 这里设置为None或空字符串
                        label=category_name,
                        args=[{}, {}],  # 空参数
                    )
                )

            # 添加"显示全部模型类型"按钮
            category_buttons.append(
                dict(method=None, label="全部模型类型", args=[{}, {}])  # 不执行任何操作
            )

            # 为每个唯一的模型编号创建一个按钮
            model_nums_sorted = sorted(model_nums)
            model_nums_sorted = [str(num) for num in model_nums_sorted if num != 999]

            model_buttons = []
            for model_num in model_nums_sorted:
                model_buttons.append(dict(method=None, label=model_num, args=[{}, {}]))

            # 添加"显示全部模型"按钮
            model_buttons.append(dict(method=None, label="全部", args=[{}, {}]))

            # 更新布局添加所有按钮组
            updatemenus = [
                # 第一行：模型类型按钮
                dict(
                    type="dropdown",  # 下拉菜单
                    direction="down",
                    active=-1,  # 默认选中最后一项（全部）
                    x=0.1,
                    y=1.15,
                    xanchor="left",
                    yanchor="top",
                    buttons=category_buttons,
                    name="categoryMenu",
                    showactive=True,
                    bgcolor="lightgray",
                ),
                # 第二行：折按钮
                dict(
                    type="dropdown",  # 下拉菜单
                    direction="down",
                    active=-1,  # 默认选中最后一项（全部）
                    x=0.3,
                    y=1.15,
                    xanchor="left",
                    yanchor="top",
                    buttons=fold_buttons,
                    name="foldMenu",
                    showactive=True,
                    bgcolor="lightgray",
                ),
                # 第三行：模型按钮
                dict(
                    type="dropdown",  # 下拉菜单
                    direction="down",
                    active=-1,  # 默认选中最后一项（全部）
                    x=0.5,
                    y=1.15,
                    xanchor="left",
                    yanchor="top",
                    buttons=model_buttons,
                    name="modelMenu",
                    showactive=True,
                    bgcolor="lightgray",
                ),
            ]

        fig.update_layout(updatemenus=updatemenus)

        # 添加下拉菜单标签
        if simple_mode:
            fig.add_annotation(
                x=0.3,
                y=1.19,
                xref="paper",
                yref="paper",
                text="折:",
                showarrow=False,
                font=dict(size=12),
                xanchor="left",
            )
        else:
            fig.add_annotation(
                x=0.1,
                y=1.19,
                xref="paper",
                yref="paper",
                text="模型类型:",
                showarrow=False,
                font=dict(size=12),
                xanchor="left",
            )

            fig.add_annotation(
                x=0.3,
                y=1.19,
                xref="paper",
                yref="paper",
                text="折:",
                showarrow=False,
                font=dict(size=12),
                xanchor="left",
            )

            fig.add_annotation(
                x=0.5,
                y=1.19,
                xref="paper",
                yref="paper",
                text="模型编号:",
                showarrow=False,
                font=dict(size=12),
                xanchor="left",
            )

        # 设置图表尺寸和布局，确保绘图区域不变窄
        fig.update_layout(
            autosize=False,  # 禁用自动调整大小
            width=1000,  # 设置固定宽度
            height=800,  # 设置固定高度
            margin=dict(t=120, l=80, r=80, b=80),  # 增加顶部边距，为下拉菜单腾出空间
            showlegend=False,  # 禁用图例
        )

        # 添加JavaScript用于"与"逻辑交互
        post_script = (
            """
        var gd = document.querySelectorAll('div.js-plotly-plot')[0];
        
        // 存储当前选择的状态
        var currentSelection = {
            category: "all",
            fold: "all",
            model: "all"
        };

        // 判断是否为简化模式
        var simpleMode = """
            + ("true" if simple_mode else "false")
            + """;
        console.log("简化模式:", simpleMode);
        
        // 从Plotly对象中提取曲线信息
        var traceInfo = extractTraceInfo(gd);
        
        // 提取按类别、折叠和模型编号分组的曲线索引
        var foldGroups = groupTracesByProperty(traceInfo, 'fold');
        var categoryGroups = simpleMode ? {} : groupTracesByProperty(traceInfo, 'category');
        var modelGroups = simpleMode ? {} : groupTracesByProperty(traceInfo, 'modelNum');
        
        console.log("分类组数:", Object.keys(categoryGroups).length);
        console.log("折叠组数:", Object.keys(foldGroups).length);
        console.log("模型组数:", Object.keys(modelGroups).length);

        // 监听菜单区域，使用事件委托处理点击
        document.addEventListener('click', function(e) {
            // 识别点击的是否为下拉菜单按钮区域
            var menuButton = e.target.closest('.updatemenu-item-rect, .updatemenu-item-text');
            if (menuButton) {
                console.log("检测到菜单按钮区域被点击");
                
                // 识别是哪个下拉菜单
                var filterType = identifyMenuType(menuButton, simpleMode);
                console.log("识别到菜单类型:", filterType);
                
                // 添加延迟以确保下拉菜单已经展开
                setTimeout(function() {
                    // 为所有菜单项添加点击监听
                    setupMenuItemListeners(filterType, currentSelection, simpleMode);
                }, 100);
            }
        }, true);

        // 从类名和位置识别菜单类型
        function identifyMenuType(menuButton, simpleMode) {
            if (simpleMode) {
                return "fold";  // 简化模式下只有fold菜单
            }

            // 尝试通过位置判断
            var rect = menuButton.getBoundingClientRect();
            var centerX = rect.left + rect.width / 2;
            
            // 根据菜单按钮的水平位置判断类型
            var plotlyRect = gd.getBoundingClientRect();
            var relativeX = (centerX - plotlyRect.left) / plotlyRect.width;
            
            if (relativeX < 0.3) return "category";
            else if (relativeX < 0.4) return "fold";
            else return "model";
        }
        
        // 为菜单项设置点击监听
        function setupMenuItemListeners(filterType, currentSelection, simpleMode) {
            var menuItems = document.querySelectorAll('.updatemenu-dropdown-button');
            console.log("找到菜单项:", menuItems.length, "个");
            
            menuItems.forEach(function(item) {
                // 避免重复添加监听器
                if (item.dataset.hasListener) return;
                
                item.dataset.hasListener = "true";
                item.addEventListener('click', function(e) {
                    var itemText = this.querySelector('.updatemenu-item-text').textContent.trim();
                    console.log("菜单项被点击:", itemText, "类型:", filterType);
                    
                    // 更新当前筛选状态
                    if (filterType === "fold") {
                        if (itemText === "全部折") {
                            currentSelection.fold = "all";
                        } else {
                            // 从"折 0"中提取数字
                            currentSelection.fold = itemText.replace("折 ", "");
                        }
                    } else if (!simpleMode && filterType === "category") {
                        currentSelection.category = itemText === "全部模型类型" ? "all" : itemText;
                    } else if (!simpleMode && filterType === "model") {
                        currentSelection.model = itemText === "全部" ? "all" : itemText;
                    }
                    
                    console.log("更新后的筛选状态:", currentSelection);
                    
                    // 应用组合筛选逻辑
                    setTimeout(function() {
                        applyCustomFilter(
                            gd, 
                            currentSelection, 
                            traceInfo, 
                            categoryGroups, 
                            foldGroups, 
                            modelGroups,
                            simpleMode,
                        );
                    }, 100);
                });
            });
        }
    
        // 从Plotly图表中提取曲线信息
        function extractTraceInfo(gd) {
            var traceInfo = [];

            for(var i = 0; i < gd.data.length; i++) {
                var trace = gd.data[i];
                var category = "";
                var fold = "";
                var modelNum = "";

                // 提取customdata
                if(trace.customdata) {
                    category = trace.customdata[0];
                    fold = trace.customdata[1];
                    modelNum = trace.customdata[2];
                }

                traceInfo.push({
                    index: i,
                    category: category,
                    fold: fold,
                    modelNum: modelNum
                });
            }

            return traceInfo;
        }

        // 按属性对曲线进行分组
        function groupTracesByProperty(traceInfo, property) {
            var groups = {};

            traceInfo.forEach(function(trace) {
                var value = trace[property];
                if (!groups[value]) {
                    groups[value] = [];
                }
                groups[value].push(trace.index);
            });

            return groups;
        }

        // 应用自定义筛选逻辑
        function applyCustomFilter(
            gd, 
            selection, 
            traceInfo, 
            categoryGroups, 
            foldGroups, 
            modelGroups,
            simpleMode,
        ) {
            console.log("应用筛选逻辑, 当前状态:", selection);

            console.log(categoryGroups);
            console.log(foldGroups);
            console.log(modelGroups);

            console.log(traceInfo);

            var newVisibility = [];
            var updateIndices = [];

            // 对每条曲线应用筛选逻辑
            for(var i = 0; i < traceInfo.length; i++) {
                var trace = traceInfo[i];
                var traceIdx = trace.index;
                updateIndices.push(traceIdx);

                // 默认继承原始可见性
                var isVisible = true;

                // 应用折筛选条件
                if (selection.fold !== "all" && 
                    foldGroups[selection.fold] && 
                    !foldGroups[selection.fold].includes(traceIdx)) {
                    isVisible = false;
                }

                // 在非简化模式下，应用其他筛选条件
                if (!simpleMode) {
                    // 应用类别筛选条件
                    if (isVisible && selection.category !== "all" && 
                        categoryGroups[selection.category] && 
                        !categoryGroups[selection.category].includes(traceIdx)) {
                        isVisible = false;
                    }
                    
                    // 应用模型编号筛选条件
                    if (isVisible && selection.model !== "all" && 
                        modelGroups[selection.model] && 
                        !modelGroups[selection.model].includes(traceIdx)) {
                        isVisible = false;
                    }
                }

                newVisibility.push(isVisible);
            }

            console.log("更新曲线可见性, 可见曲线数:", newVisibility.filter(Boolean).length);

            // 使用Plotly.restyle更新可见性
            Plotly.restyle(gd, {
                visible: newVisibility
            }, updateIndices);
        }
        """
        )

        return fig, post_script

    def plot_interactive_training_curves(self):
        """
        创建交互式训练曲线图表，展示评估分数和损失随训练轮次的变化
        """
        # 创建保存目录
        training_curves_dir = os.path.join(self.output_dir, "training_curves")
        os.makedirs(training_curves_dir, exist_ok=True)

        # 获取所有模型数据
        all_models = self.model_performance_df.copy()

        # 排除权重法模型，因为它们没有训练曲线
        nn_models = all_models[all_models["model_category"] != "baseline_weights"]

        # 检查是否有足够的数据
        if nn_models.empty:
            print("⚠ 警告: 没有足够的模型数据来绘制交互式训练曲线")
            return

        # 获取评估指标类型 (从模型参数中获取)
        metric_type = "MAP"  # 默认值
        for _, model_row in nn_models.iterrows():
            if "metric_type" in model_row and not pd.isna(model_row["metric_type"]):
                metric_type = model_row["metric_type"]
                break

        # 模型类别名称映射
        category_names = {
            "baseline_mlp": "MLP",
            "gat_selfloop": "GAT自环",
            "gat_realedge": "GAT真实边",
        }

        # 为每个fold分配不同颜色
        folds = nn_models["fold_num"].unique()
        colorscale = px.colors.qualitative.Set1
        fold_colors = {
            fold: colorscale[i % len(colorscale)] for i, fold in enumerate(folds)
        }

        # 为每种模型类别定义线型
        dash_styles = {
            "baseline_mlp": "solid",  # 实线
            "gat_selfloop": "dash",  # 虚线
            "gat_realedge": "dashdot",  # 点划线
        }

        # 创建图表
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                f"模型训练过程中的{metric_type}评估分数变化",
                "模型训练过程中的损失变化",
            ),
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        # 为每个模型绘制曲线
        for _, model_row in nn_models.iterrows():
            model_id = model_row["model_id"]
            fold = model_row["fold_num"]
            category = model_row["model_category"]

            # 获取训练摘要，检查是否包含训练曲线数据
            if model_id not in self.experiment_tracker.training_summaries:
                continue

            summary = self.experiment_tracker.training_summaries[model_id]

            if "all_scores" not in summary or not summary["all_scores"]:
                continue

            # 获取训练轮次、分数和损失
            epochs = list(range(1, len(summary["all_scores"]) + 1))
            scores = summary["all_scores"]
            losses = summary.get("all_losses", [])

            # 使用create_hover_text函数获取详细的模型参数文本
            base_hover_text = self._create_hover_text(
                model_row, include_params=True, include_scores=False
            )

            # 生成悬停文本
            hover_text = []
            for i, (epoch, score) in enumerate(zip(epochs, scores)):
                # 添加轮次特定信息
                epoch_info = f"<b>轮次: {epoch}</b><br>{metric_type}分数: {score:.6f}"
                if i < len(losses):
                    epoch_info += f"<br>损失: {losses[i]:.6f}"

                # 结合基础模型信息和轮次特定信息
                hover_text.append(f"{base_hover_text}<br><br>{epoch_info}")

            # 添加分数曲线
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=scores,
                    mode="lines+markers",
                    legendgroup=model_id,
                    marker=dict(size=4),  # 减小点的大小
                    line=dict(
                        color=fold_colors[fold],
                        dash=dash_styles[category],
                        width=1.5,  # 减小线的粗细
                    ),
                    text=hover_text,
                    hoverinfo="text",
                    customdata=[
                        category_names[category],
                        fold,
                        int(model_id.split("_")[-1]),
                    ],
                ),
                row=1,
                col=1,
            )

            # 如果有损失数据，添加损失曲线
            if losses:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=losses,
                        mode="lines+markers",
                        legendgroup=model_id,
                        marker=dict(size=4),  # 减小点的大小
                        line=dict(
                            color=fold_colors[fold],
                            dash=dash_styles[category],
                            width=1.5,  # 减小线的粗细
                        ),
                        text=hover_text,
                        hoverinfo="text",
                        showlegend=False,  # 不在图例中显示，已在分数图中显示
                        customdata=[
                            category_names[category],
                            fold,
                            int(model_id.split("_")[-1]),
                        ],
                    ),
                    row=2,
                    col=1,
                )

        # 应用通用的交互按钮布局
        fig, post_script = self._create_interactive_filter_buttons(
            fig, folds, category_names
        )

        # 设置x轴和y轴标签
        fig.update_xaxes(title_text="训练轮次", row=2, col=1)
        fig.update_yaxes(title_text=f"{metric_type}评估分数", row=1, col=1)
        fig.update_yaxes(title_text="损失值", row=2, col=1)

        fig_config = {
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"training_curves_{metric_type}",
                "height": 900,
                "width": 1200,
                "scale": 2,
            },
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }

        # 保存为HTML文件
        html_path = os.path.join(
            training_curves_dir, f"interactive_training_curves_{metric_type}.html"
        )
        fig.write_html(
            html_path,
            config=fig_config,
            include_plotlyjs="cdn",  # 使用CDN加载plotly.js以减小文件大小
            include_mathjax="cdn",
            post_script=post_script,
        )

        return html_path

    def plot_model_categories_comparison(self, score_type="MAP"):
        """
        绘制不同类别模型的对比散点图(MLP, GAT自环, GAT真实边)

        参数:
            score_type: 评分类型 (MAP或MRR)
        """
        # 创建保存目录
        model_comparison_dir = os.path.join(self.output_dir, "model_comparison")
        os.makedirs(model_comparison_dir, exist_ok=True)

        train_score_col = f"train_{score_type}_score"
        test_score_col = f"predict_{score_type}_score"
        fitting_status_col = f"fitting_status_{score_type}"

        # 准备数据
        all_models = self.model_performance_df.copy()

        # 创建散点图
        plt.figure(figsize=(12, 10))

        # 为每个fold分配不同颜色
        folds = all_models["fold_num"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(folds)))
        fold_color_map = {fold: colors[i] for i, fold in enumerate(folds)}

        # 为每种模型类别定义标记
        markers = {
            "baseline_mlp": "^",  # 三角形
            "gat_selfloop": "s",  # 方形
            "gat_realedge": "o",  # 圆形
        }

        # 模型类别名称映射
        category_names = {
            "baseline_mlp": "MLP模型",
            "gat_selfloop": "GAT自环模型",
            "gat_realedge": "GAT真实边模型",
        }

        # 拟合状态的边框颜色
        fitting_edge_colors = {
            "过拟合": "red",
            "欠拟合": "blue",
            "正常拟合": "green",
            "未知": None,
        }

        legend_elements = []

        # 绘制所有模型点
        for fold in folds:
            fold_models = all_models[all_models["fold_num"] == fold]

            # 按模型类别绘制
            for category, marker in markers.items():
                category_models = fold_models[fold_models["model_category"] == category]

                if not category_models.empty:
                    plt.scatter(
                        category_models[train_score_col],
                        category_models[test_score_col],
                        marker=marker,
                        s=80,
                        color=fold_color_map[fold],
                        alpha=0.7,
                        label="_nolegend_",
                    )

                    # 根据拟合状态添加边框
                    for _, model in category_models.iterrows():
                        if (
                            fitting_status_col in model.index
                            and not pd.isna(model[fitting_status_col])
                            and model[fitting_status_col] in fitting_edge_colors
                        ):
                            edge_color = fitting_edge_colors[model[fitting_status_col]]
                            if edge_color:  # 只为过拟合和欠拟合添加边框
                                plt.scatter(
                                    model[train_score_col],
                                    model[test_score_col],
                                    marker=marker,
                                    s=100,  # 稍大一点以便边框可见
                                    facecolors="none",
                                    edgecolors=edge_color,
                                    linewidths=1.5,
                                    label="_nolegend_",
                                )

                    # 只为每个类别的第一个fold添加图例
                    if fold == folds[0]:
                        legend_elements.append(
                            Line2D(
                                [0],
                                [0],
                                marker=marker,
                                color="k",
                                linestyle="none",
                                markerfacecolor="none",
                                markeredgewidth=1.5,
                                label=f"{category_names[category]}",
                                markersize=10,
                            )
                        )

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"折 {fold}",
                    markerfacecolor=fold_color_map[fold],
                    markersize=10,
                )
            )

        # 用虚线连接配对的模型
        for fold in folds:
            # 获取当前fold的配对关系
            paired_map = self.paired_models.get(fold, {})
            if not paired_map:
                continue

            fold_models = all_models[all_models["fold_num"] == fold]
            for selfloop_id, realedge_id in paired_map.items():
                selfloop_model = fold_models[fold_models["model_id"] == selfloop_id]
                realedge_model = fold_models[fold_models["model_id"] == realedge_id]

                if selfloop_model.empty or realedge_model.empty:
                    continue

                # 画连接线
                plt.plot(
                    [
                        selfloop_model[train_score_col].iloc[0],
                        realedge_model[train_score_col].iloc[0],
                    ],
                    [
                        selfloop_model[test_score_col].iloc[0],
                        realedge_model[test_score_col].iloc[0],
                    ],
                    "k--",  # 黑色虚线
                    alpha=0.5,
                    linewidth=0.7,
                )

        best_models = self.model_performance_df[
            self.model_performance_df[f"is_best_{score_type}"] == True
        ]

        # 添加拟合状态到图例
        legend_elements.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markeredgecolor="red",
                    markerfacecolor="none",
                    markersize=10,
                    label="过拟合",
                    markeredgewidth=1.5,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markeredgecolor="blue",
                    markerfacecolor="none",
                    markersize=10,
                    label="欠拟合",
                    markeredgewidth=1.5,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markeredgecolor="green",
                    markerfacecolor="none",
                    markersize=10,
                    label="正常拟合",
                    markeredgewidth=1.5,
                ),
            ]
        )

        # 添加最佳模型到图例
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markeredgecolor="gold",
                markerfacecolor="none",
                markersize=10,
                label="最佳模型",
                markeredgewidth=1.5,
            )
        )

        # 标记最佳模型
        for _, model in pd.DataFrame(best_models).iterrows():
            category = model["model_category"]
            marker = markers[category]
            plt.scatter(
                model[train_score_col],
                model[test_score_col],
                marker=marker,
                s=100,
                facecolors="none",
                edgecolors="gold",
                linewidths=2,
                label="_nolegend_",
            )

        plt.axis("equal")

        # 添加辅助元素
        plt.xlabel(f"回归阶段{score_type}得分", fontsize=12)
        plt.ylabel(f"测试阶段{score_type}得分", fontsize=12)
        plt.title(f"不同类别模型{score_type}性能对比", fontsize=14)

        plt.legend(handles=legend_elements, loc="best")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                model_comparison_dir, f"model_categories_comparison_{score_type}.png"
            )
        )
        plt.close()

    def plot_interactive_model_comparison(self, score_type="MAP"):
        """
        创建交互式散点图，鼠标悬停时显示模型详细信息

        参数:
            score_type: 评分类型 (MAP或MRR)
        """
        # 创建保存目录
        model_comparison_dir = os.path.join(self.output_dir, "model_comparison")
        os.makedirs(model_comparison_dir, exist_ok=True)

        train_score_col = f"train_{score_type}_score"
        test_score_col = f"predict_{score_type}_score"
        fitting_status_col = f"fitting_status_{score_type}"

        # 准备数据
        all_models = self.model_performance_df.copy()

        # 模型类别名称映射
        category_names = {
            "baseline_mlp": "MLP模型",
            "gat_selfloop": "GAT自环模型",
            "gat_realedge": "GAT真实边模型",
            "baseline_weights": "权重法",
        }

        # 拟合状态的边框颜色 - 与普通图保持一致
        fitting_edge_colors = {
            "过拟合": "red",
            "欠拟合": "blue",
            "正常拟合": "green",
            "未知": None,
        }

        # 为每个fold分配不同颜色
        folds = all_models["fold_num"].unique()
        colorscale = px.colors.qualitative.Set1
        fold_colors = {
            fold: colorscale[i % len(colorscale)] for i, fold in enumerate(folds)
        }

        # 创建图表
        fig = go.Figure()

        # 添加所有模型点
        for fold in folds:
            fold_models = all_models[all_models["fold_num"] == fold]

            # 按模型类别绘制
            for category in category_names:
                category_models = fold_models[fold_models["model_category"] == category]

                if not category_models.empty:
                    # 生成hover文本
                    hover_texts = category_models.apply(
                        self._create_hover_text, axis=1
                    ).tolist()

                    fig.add_trace(
                        go.Scatter(
                            x=category_models[train_score_col],
                            y=category_models[test_score_col],
                            mode="markers",
                            marker=dict(
                                size=12,
                                color=fold_colors[fold],
                                symbol={
                                    "baseline_mlp": "triangle-up",
                                    "gat_selfloop": "square",
                                    "gat_realedge": "circle",
                                    "baseline_weights": "x",
                                }.get(category, "circle"),
                                line=dict(width=1, color="DarkSlateGrey"),
                            ),
                            name=f"{category_names[category]} (折 {fold})",
                            customdata=[None, fold, None],
                            text=hover_texts,
                            hoverinfo="text",
                            showlegend=True,
                        )
                    )

                    # 根据拟合状态添加边框
                    for fitting_status, edge_color in fitting_edge_colors.items():
                        if edge_color:  # 只为有颜色的拟合状态添加边框
                            status_models = category_models[
                                category_models[fitting_status_col] == fitting_status
                            ]

                            if not status_models.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=status_models[train_score_col],
                                        y=status_models[test_score_col],
                                        mode="markers",
                                        marker=dict(
                                            size=16,  # 稍大一点以便边框可见
                                            color="rgba(0,0,0,0)",  # 透明填充
                                            symbol={
                                                "baseline_mlp": "triangle-up",
                                                "gat_selfloop": "square",
                                                "gat_realedge": "circle",
                                            }.get(category, "circle"),
                                            line=dict(width=2, color=edge_color),
                                        ),
                                        name=f"{fitting_status} (折 {fold})",
                                        customdata=[None, fold, None],
                                        hoverinfo="text",
                                        showlegend=False,
                                    )
                                )

        # 标记最佳模型
        best_models = self.model_performance_df[
            self.model_performance_df[f"is_best_{score_type}"] == True
        ]

        for i, model in pd.DataFrame(best_models).iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[model[train_score_col]],
                    y=[model[test_score_col]],
                    mode="markers",
                    marker=dict(
                        size=18,
                        color="rgba(0,0,0,0)",
                        symbol={
                            "baseline_mlp": "triangle-up",
                            "gat_selfloop": "square",
                            "gat_realedge": "circle",
                        }.get(model["model_category"], "circle"),
                        line=dict(width=2, color="gold"),
                    ),
                    name="最佳模型",
                    customdata=[None, model["fold_num"], None],
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # 添加连接配对模型的线
        for fold in folds:
            # 获取当前fold的配对关系（现在是单向的selfloop -> realedge）
            paired_map = self.paired_models.get(fold, {})
            if not paired_map:
                continue

            fold_models = all_models[all_models["fold_num"] == fold]

            for selfloop_id, realedge_id in paired_map.items():
                selfloop_model = fold_models[fold_models["model_id"] == selfloop_id]
                realedge_model = fold_models[fold_models["model_id"] == realedge_id]

                if selfloop_model.empty or realedge_model.empty:
                    continue

                # 画连接线
                fig.add_trace(
                    go.Scatter(
                        x=[
                            selfloop_model.iloc[0][train_score_col],
                            realedge_model.iloc[0][train_score_col],
                        ],
                        y=[
                            selfloop_model.iloc[0][test_score_col],
                            realedge_model.iloc[0][test_score_col],
                        ],
                        mode="lines",
                        line=dict(color="grey", width=1, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                        customdata=[None, fold, None],
                    )
                )

        # 应用通用的交互按钮布局
        fig, post_script = self._create_interactive_filter_buttons(
            fig, folds, category_names, simple_mode=True
        )

        # 设置图表布局
        fig.update_layout(
            title=dict(
                text=f"不同类别模型{score_type}性能对比（交互式）",
                y=0.9,  # 将标题下移，避开按钮区域
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(size=18),
            ),
            xaxis_title=f"回归阶段{score_type}得分",
            yaxis_title=f"测试阶段{score_type}得分",
            hovermode="closest",
        )

        # 设置相等的坐标轴比例
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        # 添加特殊配置
        fig_config = {
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"model_comparison_{score_type}",
                "height": 800,
                "width": 1000,
                "scale": 2,
            },
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }

        # 保存为HTML文件（可在浏览器中交互）
        html_path = os.path.join(
            model_comparison_dir, f"interactive_model_comparison_{score_type}.html"
        )

        fig.write_html(
            html_path,
            config=fig_config,
            include_plotlyjs="cdn",  # 使用CDN加载plotly.js以减小文件大小
            include_mathjax="cdn",
            post_script=post_script,
        )

        return html_path

    def generate_final_report(self):
        """生成最终综合报告"""
        score_types = ["MAP", "MRR"]
        final_report = {
            "评估时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "评估目录": self.log_dir,
            "模型总数": len(self.model_performance_df),
        }

        for score_type in score_types:
            train_score_col = f"train_{score_type}_score"
            test_score_col = f"predict_{score_type}_score"

            if (
                train_score_col not in self.model_performance_df.columns
                or test_score_col not in self.model_performance_df.columns
            ):
                print(f"⚠ 警告: {score_type}分数列不存在，跳过生成最终报告")
                continue

            best_models = self.model_performance_df[
                self.model_performance_df[f"is_best_{score_type}"] == True
            ]
            best_df = pd.DataFrame(best_models)
            if best_df.empty:
                print(f"⚠ 警告: 无法找到最佳模型，跳过生成{score_type}最终报告")
                continue

            # 获取最佳模型类别
            best_model_category = best_df["model_category"].iloc[0]

            # 按模型类别分组计算最佳性能
            fold_category_performance = (
                self.model_performance_df.groupby(["fold_num", "model_category"])[
                    test_score_col
                ]
                .max()
                .reset_index()
            )

            # 辅助函数：判断一个模型类别是否在所有fold中都优于另一个类别
            def is_better_in_all_folds(category1, category2):
                """判断category1是否在所有fold中都优于category2"""
                for fold in fold_category_performance["fold_num"].unique():
                    # 获取当前fold中两种类别的最高得分
                    cat1_score = fold_category_performance[
                        (fold_category_performance["fold_num"] == fold)
                        & (fold_category_performance["model_category"] == category1)
                    ][test_score_col]

                    cat2_score = fold_category_performance[
                        (fold_category_performance["fold_num"] == fold)
                        & (fold_category_performance["model_category"] == category2)
                    ][test_score_col]

                    # 如果任一类别在该fold中没有数据，或cat1不优于cat2，则返回False
                    if (
                        cat1_score.empty
                        or cat2_score.empty
                        or cat1_score.iloc[0] <= cat2_score.iloc[0]
                    ):
                        return False

                return True

            # 辅助函数：比较两个模型类别并返回结论
            def compare_categories(category, name):
                """比较两个模型类别，返回比较结论"""
                if category not in fold_category_performance["model_category"].values:
                    return f"未比较GAT模型与{name}"

                if is_better_in_all_folds("gat_realedge", category):
                    return f"GAT模型{score_type}表现优于{name}"
                elif is_better_in_all_folds(category, "gat_realedge"):
                    return f"{name}{score_type}表现优于GAT模型"
                else:
                    return f"GAT模型与{name}的{score_type}表现无法确定明显优劣"

            # 执行各种比较
            findings = [
                compare_categories("baseline_weights", "权重法"),
                compare_categories("gat_selfloop", "仅自环GAT模型"),
                compare_categories("baseline_mlp", "MLP模型"),
            ]

            # 整体性能统计
            category_performance = (
                self.model_performance_df.groupby("model_category")[test_score_col]
                .agg(["max", "mean"])
                .reset_index()
            )

            # 准备报告内容
            final_report[score_type] = {
                "最佳模型": best_df["model_id"].tolist(),
                "测试得分": best_df[test_score_col].tolist(),
                "最佳模型类型": {
                    "baseline_mlp": "基线MLP模型",
                    "baseline_weights": "权重线性组合模型",
                    "gat_realedge": "GAT模型",
                    "gat_selfloop": "仅自环GAT模型",
                }.get(best_model_category, "未知类型"),
                "模型类别性能": category_performance.to_dict(orient="records"),
                "主要发现": findings,
            }

        # 保存合并的报告到一个JSON文件
        json_path = os.path.join(self.output_dir, "final_report.json")
        pd.Series(final_report).to_json(
            json_path, indent=4, orient="index", force_ascii=False
        )

        return True

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
        self.analyze_model_fitting()

        # 生成最终报告
        self.generate_final_report()

        # 绘制图像
        self.plot_interactive_training_curves()
        for score_type in ["MAP", "MRR"]:
            self.plot_model_categories_comparison(score_type)
            self.plot_interactive_model_comparison(score_type)

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

    # args = argparse.Namespace(
    #     log_dir="tomcat_GAT_MAP",
    #     output=None,
    # )

    # 创建评估器并运行分析
    evaluator = AdaptiveGATEvaluator(args.log_dir)

    if args.output:
        evaluator.output_dir = args.output
        os.makedirs(evaluator.output_dir, exist_ok=True)

    evaluator.run_all_analyses()


if __name__ == "__main__":
    main()
