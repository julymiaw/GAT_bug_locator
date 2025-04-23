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
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import argparse
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
        """找出自环模型和真实边模型的配对关系"""
        # 按fold分组
        for fold_num in df["fold_num"].unique():
            self.paired_models[fold_num] = {}

            # 获取当前fold的所有模型ID
            fold_models = df[df["fold_num"] == fold_num]
            model_ids = fold_models["model_id"].tolist()

            # 检查每对模型
            for i, model_id1 in enumerate(model_ids):
                for model_id2 in model_ids[i + 1 :]:
                    # 使用实验跟踪器的方法判断是否匹配
                    if self.experiment_tracker.are_models_matching(
                        model_id1, model_id2
                    ):
                        # 建立双向映射
                        self.paired_models[fold_num][model_id1] = model_id2
                        self.paired_models[fold_num][model_id2] = model_id1

    def analyze_overall_performance(self):
        """分析整体性能并生成摘要"""
        score_types = ["MAP", "MRR"]

        overall_summary = {
            "总模型数": len(self.model_performance_df),
            "评估时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        for score_type in score_types:
            # 确定对应的列名
            train_score_col = f"train_{score_type}_score"
            test_score_col = f"predict_{score_type}_score"

            # 动态计算最佳模型（每个fold中train_score最高的非权重模型）
            best_models = []

            for fold in self.model_performance_df["fold_num"].unique():
                fold_models = self.model_performance_df[
                    (self.model_performance_df["fold_num"] == fold)
                    & (
                        self.model_performance_df["model_category"]
                        != "baseline_weights"
                    )
                    & (
                        self.model_performance_df["model_category"].isin(
                            ["gat_selfloop", "gat_realedge"]
                        )
                    )
                ]

                if not fold_models.empty and train_score_col in fold_models.columns:
                    best_idx = fold_models[train_score_col].idxmax()
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
        """分析不同参数对模型性能的影响"""
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

            # 1. 单参数分析
            all_results["single_parameters"][score_type] = {}
            for param in params_to_analyze:
                if param not in nn_models_df.columns:
                    print(f"⚠ 警告：参数 {param} 不在数据集列中，跳过分析")
                    continue

                unique_values = nn_models_df[param].unique()
                if len(unique_values) <= 1:
                    continue

                # 计算每个参数值的平均性能
                impact_data = []

                for value in unique_values:
                    if not pd.isna(value):
                        # 精确匹配非NaN值
                        subset = nn_models_df[nn_models_df[param] == value]

                        if not subset.empty:
                            impact_data.append(
                                {
                                    "param_value": value,
                                    "count": len(subset),
                                    "mean_train": subset[train_score_col].mean(),
                                    "mean_test": subset[test_score_col].mean(),
                                }
                            )

                # 特殊处理NaN值
                nan_subset = nn_models_df[pd.isna(nn_models_df[param])]
                if not nan_subset.empty:
                    impact_data.append(
                        {
                            "param_value": np.nan,
                            "count": len(nan_subset),
                            "mean_train": nan_subset[train_score_col].mean(),
                            "mean_test": nan_subset[test_score_col].mean(),
                        }
                    )

                # 保存并绘制参数影响数据
                if impact_data:
                    all_results["single_parameters"][score_type][param] = impact_data
                    self.plot_parameter_impact(
                        param, impact_data, param_dir, score_type=score_type
                    )
                else:
                    print(f"⚠ 警告：参数 {param} 没有有效数据可分析")

            # 2. 参数组合分析
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

                # 获取每个参数的唯一值
                unique_values1 = [
                    v for v in nn_models_df[param1].unique() if not pd.isna(v)
                ]
                unique_values2 = [
                    v for v in nn_models_df[param2].unique() if not pd.isna(v)
                ]

                if len(unique_values1) <= 1 and len(unique_values2) <= 1:
                    continue

                # 为此组合创建数据
                combo_data = []
                for val1 in unique_values1:
                    for val2 in unique_values2:
                        # 找到具有此参数组合的模型
                        combo_subset = nn_models_df[
                            (nn_models_df[param1] == val1)
                            & (nn_models_df[param2] == val2)
                        ]

                        if not combo_subset.empty:
                            combo_data.append(
                                {
                                    f"{param1}": val1,
                                    f"{param2}": val2,
                                    "combo_label": f"{val1}-{val2}",
                                    "count": len(combo_subset),
                                    "mean_train": combo_subset[train_score_col].mean(),
                                    "mean_test": combo_subset[test_score_col].mean(),
                                }
                            )

                if combo_data:
                    combo_key = f"{param1}_{param2}"
                    all_results["parameter_combinations"][score_type][
                        combo_key
                    ] = combo_data
                    self.plot_parameter_combination(
                        param1, param2, combo_data, param_dir, score_type=score_type
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
        self, param: str, impact_data: List[Dict], output_dir: str, score_type="MAP"
    ):
        """
        绘制参数影响图表

        参数:
            param: 参数名称
            impact_data: 影响数据列表
            output_dir: 输出目录
            score_type: 评分类型 (MAP或MRR)
        """
        # 转换为DataFrame
        impact_df = pd.DataFrame(impact_data)
        impact_df["param_label"] = impact_df["param_value"].astype(str)

        # 排序
        impact_df = impact_df.sort_values(by="param_value")

        # 绘制过拟合对比图
        plt.figure(figsize=(12, 6))

        x = np.arange(len(impact_df))
        width = 0.35

        # 训练和测试分数
        plt.bar(
            x - width / 2, impact_df["mean_train"], width, label=f"训练{score_type}"
        )
        plt.bar(x + width / 2, impact_df["mean_test"], width, label=f"测试{score_type}")

        # 添加标签
        plt.xlabel(param)
        plt.ylabel(f"{score_type}得分")
        plt.title(f"{param}参数对{score_type}训练和测试性能的对比")
        plt.xticks(x, impact_df["param_label"])
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{param}_{score_type}_train_test_compare.png")
        )
        plt.close()

    def plot_parameter_combination(
        self,
        param1: str,
        param2: str,
        combo_data: List[Dict],
        output_dir: str,
        score_type="MAP",
    ):
        """
        绘制参数组合的柱状图

        参数:
            param1: 第一个参数名称
            param2: 第二个参数名称
            combo_data: 组合分析数据列表
            output_dir: 输出目录
            score_type: 评分类型 (MAP或MRR)
        """
        if not combo_data:
            return

        # 将列表转换为DataFrame
        combo_df = pd.DataFrame(combo_data)

        # 创建组合标签
        combo_df["combined_label"] = combo_df.apply(
            lambda row: f"{row[param1]}-{row[param2]}", axis=1
        )

        # 按测试分数降序排序
        combo_df = combo_df.sort_values(by="mean_test", ascending=False)

        # 绘制柱状图
        plt.figure(figsize=(15, 8))

        x = np.arange(len(combo_df))
        width = 0.35

        # 训练和测试分数
        plt.bar(x - width / 2, combo_df["mean_train"], width, label=f"训练{score_type}")
        plt.bar(x + width / 2, combo_df["mean_test"], width, label=f"测试{score_type}")

        # 添加标签和图例
        plt.xlabel(f"{param1} - {param2} 组合")
        plt.ylabel(f"{score_type}得分")
        plt.title(f"{param1}和{param2}参数组合对{score_type}性能的影响")
        plt.xticks(x, combo_df["combined_label"], rotation=45, ha="right")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # 保存图片
        plt.savefig(
            os.path.join(output_dir, f"{param1}_{param2}_{score_type}_bar_chart.png")
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
                # 获取当前fold的模型
                fold_models = self.model_performance_df[
                    self.model_performance_df["fold_num"] == fold
                ]
                fold_self_loop = fold_models[
                    fold_models["model_category"] == "gat_selfloop"
                ]
                fold_real_edge = fold_models[
                    fold_models["model_category"] == "gat_realedge"
                ]

                # 使用预处理阶段建立的配对关系
                fold_pairs = []

                # 检查是否有配对信息
                for _, self_loop_model in fold_self_loop.iterrows():
                    model_id = self_loop_model["model_id"]

                    # 获取对应的真实边模型名称
                    if (
                        fold in self.paired_models
                        and model_id in self.paired_models[fold]
                    ):
                        real_edge_id = self.paired_models[fold][model_id]
                        real_edge_models = fold_real_edge[
                            fold_real_edge["model_id"] == real_edge_id
                        ]

                        if not real_edge_models.empty:
                            real_edge_model = real_edge_models.iloc[0]

                            # 只有当两个模型都有评分时才添加
                            if (
                                train_score_col in self_loop_model
                                and test_score_col in self_loop_model
                                and train_score_col in real_edge_model
                                and test_score_col in real_edge_model
                            ):

                                pair_info = {
                                    "self_loop_model": model_id,
                                    "real_edge_model": real_edge_id,
                                    f"self_loop_train_{score_type}": self_loop_model[
                                        train_score_col
                                    ],
                                    f"real_edge_train_{score_type}": real_edge_model[
                                        train_score_col
                                    ],
                                    f"self_loop_test_{score_type}": self_loop_model[
                                        test_score_col
                                    ],
                                    f"real_edge_test_{score_type}": real_edge_model[
                                        test_score_col
                                    ],
                                    f"real_edge_improvement_{score_type}": real_edge_model[
                                        test_score_col
                                    ]
                                    - self_loop_model[test_score_col],
                                }
                                fold_pairs.append(pair_info)

                # 对结果按真实边模型对比自环模型的提升量降序排列
                fold_pairs.sort(
                    key=lambda x: x[f"real_edge_improvement_{score_type}"], reverse=True
                )

                # 保存当前fold的统计信息
                self_loop_stats[fold] = fold_pairs

            # 绘制不同类别模型的对比散点图
            self.plot_model_categories_comparison(self_loop_stats, score_type)

            self.plot_interactive_model_comparison(self_loop_stats, score_type)

            all_results[score_type] = self_loop_stats

        json_path = os.path.join(self.output_dir, "self_loops_analysis.json")
        pd.Series(all_results).to_json(
            json_path, indent=4, orient="index", force_ascii=False
        )

        return all_results

    def plot_interactive_model_comparison(self, fold_comparisons, score_type="MAP"):
        """
        创建交互式散点图，鼠标悬停时显示模型详细信息

        参数:
            fold_comparisons: 按fold组织的比较数据字典
            score_type: 评分类型 (MAP或MRR)
        """
        if not fold_comparisons:
            return

        train_score_col = f"train_{score_type}_score"
        test_score_col = f"predict_{score_type}_score"

        # 准备数据
        all_models = self.model_performance_df.copy()

        # 模型类别名称映射
        category_names = {
            "baseline_mlp": "MLP模型",
            "gat_selfloop": "GAT自环模型",
            "gat_realedge": "GAT真实边模型",
            "baseline_weights": "权重法",
        }

        # 为每个fold分配不同颜色
        folds = all_models["fold_num"].unique()
        colorscale = px.colors.qualitative.Set1
        fold_colors = {
            fold: colorscale[i % len(colorscale)] for i, fold in enumerate(folds)
        }

        # 创建图表
        fig = go.Figure()

        # 生成自定义悬停信息
        def create_hover_text(row):
            # 收集模型的所有参数和结果
            params = [f"<b>{row['model_id']}</b>"]
            params.append(
                f"类别: {category_names.get(row['model_category'], row['model_category'])}"
            )

            # 添加基本信息
            params.append(f"折: {row['fold_num']}")
            params.append(f"训练{score_type}: {row[train_score_col]:.4f}")
            params.append(f"测试{score_type}: {row[test_score_col]:.4f}")

            # 添加训练信息
            if "training_stop_reason" in row:
                params.append(f"停止原因: {row['training_stop_reason']}")
            if "training_best_epoch" in row:
                params.append(f"最佳轮次: {row['training_best_epoch']}")
            if "training_final_epoch" in row:
                params.append(f"总轮次: {row['training_final_epoch']}")
            if "fitting_status_" + score_type in row:
                params.append(f"拟合状态: {row['fitting_status_' + score_type]}")

            # 添加模型参数
            model_params = [
                "heads",
                "hidden_dim",
                "dropout",
                "alpha",
                "lr",
                "penalty",
                "loss",
                "max_iter",
                "n_iter_no_change",
                "use_self_loops_only",
            ]

            params.append("<b>模型参数:</b>")
            for param in model_params:
                if param in row and not pd.isna(row[param]):
                    params.append(f"{param}: {row[param]}")

            return "<br>".join(params)

        # 添加所有模型点
        for fold in folds:
            fold_models = all_models[all_models["fold_num"] == fold]

            # 按模型类别绘制
            for category in category_names:
                category_models = fold_models[fold_models["model_category"] == category]

                if not category_models.empty:
                    # 生成hover文本
                    hover_texts = category_models.apply(
                        create_hover_text, axis=1
                    ).tolist()

                    # 添加散点
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
                            text=hover_texts,
                            hoverinfo="text",
                            showlegend=True,
                        )
                    )

        # 标记最佳模型
        best_models = []
        for fold in folds:
            fold_models = all_models[
                (all_models["fold_num"] == fold)
                & (all_models["model_category"] != "baseline_weights")
                & (all_models["model_category"].isin(["gat_selfloop", "gat_realedge"]))
            ]
            if not fold_models.empty and train_score_col in fold_models.columns:
                best_idx = fold_models[train_score_col].idxmax()
                best_models.append(all_models.loc[best_idx])

        for model in best_models:
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
                    hoverinfo="skip",
                    showlegend=True if model is best_models[0] else False,
                )
            )

        # 添加连接配对模型的线
        for fold, fold_data in fold_comparisons.items():
            for pair in fold_data:
                self_loop_model = all_models[
                    (all_models["model_id"] == pair["self_loop_model"])
                    & (all_models["fold_num"] == fold)
                ]
                real_edge_model = all_models[
                    (all_models["model_id"] == pair["real_edge_model"])
                    & (all_models["fold_num"] == fold)
                ]

                if not self_loop_model.empty and not real_edge_model.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=[
                                self_loop_model.iloc[0][train_score_col],
                                real_edge_model.iloc[0][train_score_col],
                            ],
                            y=[
                                self_loop_model.iloc[0][test_score_col],
                                real_edge_model.iloc[0][test_score_col],
                            ],
                            mode="lines",
                            line=dict(color="grey", width=1, dash="dash"),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        # 设置图表布局
        fig.update_layout(
            title=f"不同类别模型{score_type}性能对比（交互式）",
            xaxis_title=f"回归阶段{score_type}得分",
            yaxis_title=f"测试阶段{score_type}得分",
            hovermode="closest",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.5)",
            ),
            width=900,
            height=800,
        )

        # 设置相等的坐标轴比例
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        # 保存为HTML文件（可在浏览器中交互）
        html_path = os.path.join(
            self.output_dir, f"interactive_model_comparison_{score_type}.html"
        )
        fig.write_html(html_path)

        return html_path

    def plot_model_categories_comparison(self, fold_comparisons, score_type="MAP"):
        """
        绘制不同类别模型的对比散点图(MLP, GAT自环, GAT真实边)

        参数:
            fold_comparisons: 按fold组织的比较数据字典
            score_type: 评分类型 (MAP或MRR)
        """
        if not fold_comparisons:
            return

        train_score_col = f"train_{score_type}_score"
        test_score_col = f"predict_{score_type}_score"

        # 准备数据
        all_models = self.model_performance_df.copy()

        # 获取权重方法分数，用于判断模型的拟合状态
        weights_scores = {}
        for fold in all_models["fold_num"].unique():
            fold_weights = all_models[
                (all_models["fold_num"] == fold)
                & (all_models["model_category"] == "baseline_weights")
                & (all_models["model_category"].isin(["gat_selfloop", "gat_realedge"]))
            ]
            if not fold_weights.empty and train_score_col in fold_weights.columns:
                weights_scores[fold] = {
                    "train": fold_weights[train_score_col].iloc[0],
                    "test": fold_weights[test_score_col].iloc[0],
                }

        # 判断每个模型的拟合状态
        def determine_fitting_status(row):
            fold = row["fold_num"]
            if fold not in weights_scores:
                return "未知"

            train_diff = row[train_score_col] - weights_scores[fold]["train"]
            test_diff = row[test_score_col] - weights_scores[fold]["test"]

            if train_diff < -0.01:  # 回归分数比权重法低1%以上
                return "欠拟合"
            elif (
                train_diff > 0.01 and test_diff < -0.02
            ):  # 回归分数高但测试分数低2%以上
                return "过拟合"
            return "正常拟合"

        # 为非权重模型添加拟合状态
        nn_models = all_models["model_category"] != "baseline_weights"
        all_models.loc[nn_models, "fitting_status"] = all_models[nn_models].apply(
            determine_fitting_status, axis=1
        )

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
            "正常拟合": None,  # 正常拟合不添加边框
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
                            "fitting_status" in model
                            and model["fitting_status"] in fitting_edge_colors
                        ):
                            edge_color = fitting_edge_colors[model["fitting_status"]]
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
        for fold, fold_data in fold_comparisons.items():
            for pair in fold_data:
                self_loop_model = all_models[
                    (all_models["model_id"] == pair["self_loop_model"])
                    & (all_models["fold_num"] == fold)
                ]
                real_edge_model = all_models[
                    (all_models["model_id"] == pair["real_edge_model"])
                    & (all_models["fold_num"] == fold)
                ]

                if (
                    not self_loop_model.empty
                    and not real_edge_model.empty
                    and train_score_col in self_loop_model.columns
                    and test_score_col in self_loop_model.columns
                    and train_score_col in real_edge_model.columns
                    and test_score_col in real_edge_model.columns
                ):

                    plt.plot(
                        [
                            self_loop_model.iloc[0][train_score_col],
                            real_edge_model.iloc[0][train_score_col],
                        ],
                        [
                            self_loop_model.iloc[0][test_score_col],
                            real_edge_model.iloc[0][test_score_col],
                        ],
                        "k--",  # 黑色虚线
                        alpha=0.5,
                        linewidth=0.7,
                    )

        # 动态找出最佳模型并标记
        best_models = []
        for fold in folds:
            fold_models = all_models[
                (all_models["fold_num"] == fold)
                & (all_models["model_category"] != "baseline_weights")
            ]
            if not fold_models.empty and train_score_col in fold_models.columns:
                best_idx = fold_models[train_score_col].idxmax()
                best_models.append(all_models.loc[best_idx])

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
                self.output_dir, f"model_categories_comparison_{score_type}.png"
            )
        )
        plt.close()

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

    def analyze_epoch_impact(self):
        """分析不同参数对训练轮次(epoch)的影响"""
        # 筛选有epoch信息的神经网络模型
        nn_models_df = self.model_performance_df[
            (self.model_performance_df["model_category"] != "baseline_weights")
            & (self.model_performance_df["training_best_epoch"].notna())
        ]

        if nn_models_df.empty:
            print("⚠ 警告: 未找到包含训练轮次信息的模型")
            return {}

        score_types = ["MAP", "MRR"]
        all_results = {}

        for score_type in score_types:
            fitting_status_col = f"fitting_status_{score_type}"

            if fitting_status_col not in nn_models_df.columns:
                print(
                    f"⚠ 警告: 未找到{score_type}过拟合状态列({fitting_status_col})！请先确保已执行过'analyze_model_fitting'步骤。"
                )
                continue

            # 统计不同拟合状态下的epoch分布
            epoch_results = (
                nn_models_df.groupby(fitting_status_col)["training_best_epoch"]
                .agg(["count", "mean", "median", "min", "max"])
                .reset_index()
            )

            # 绘制不同拟合状态对应的epoch分布（示例：箱线图）
            plt.figure(figsize=(8, 6))

            # 为每个fold分配不同颜色
            folds = nn_models_df["fold_num"].unique()
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

            # 拟合状态映射到数值位置
            status_mapping = {
                "欠拟合": 0,
                "正常拟合": 1,
                "过拟合": 2,
                "未知": 1.5,  # 如有未知状态
            }

            # 图例元素列表
            legend_elements = []

            # 按拟合状态分组绘制，但不使用sns.stripplot
            for fold in folds:
                fold_models = nn_models_df[nn_models_df["fold_num"] == fold]

                # 按模型类别绘制
                for category, marker in markers.items():
                    category_models = fold_models[
                        fold_models["model_category"] == category
                    ]

                    # 绘制每个拟合状态
                    for status in category_models[fitting_status_col].unique():
                        status_models = category_models[
                            category_models[fitting_status_col] == status
                        ]

                        if not status_models.empty:
                            # 为每个数据点增加小的随机偏移，模拟jitter效果
                            jitter = np.random.uniform(-0.15, 0.15, len(status_models))
                            status_positions = (
                                np.zeros(len(status_models))
                                + status_mapping[status]
                                + jitter
                            )

                            plt.scatter(
                                status_positions,
                                status_models["training_best_epoch"],
                                marker=marker,
                                s=80,
                                color=fold_color_map[fold],
                                alpha=0.7,
                                label="_nolegend_",
                            )

                # 为每个fold添加图例
                if fold == folds[0]:  # 避免图例重复
                    for category, marker in markers.items():
                        legend_elements.append(
                            Line2D(
                                [0],
                                [0],
                                marker=marker,
                                color="k",
                                linestyle="none",
                                markerfacecolor="none",
                                markeredgewidth=1.5,
                                label=f"{category_names.get(category, category)}",
                                markersize=10,
                            )
                        )

                # 为每个fold添加图例
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

            plt.xticks([0, 1, 2], ["欠拟合", "正常拟合", "过拟合"])
            plt.xlabel("拟合状态", fontsize=12)
            plt.ylabel("最佳训练轮次(best_epoch)", fontsize=12)
            plt.title(f"{score_type}指标下不同拟合状态与训练轮次的关系", fontsize=14)
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # 添加图例
            plt.legend(handles=legend_elements, loc="best")
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    self.output_dir, f"epoch_overfitting_analysis_{score_type}.png"
                )
            )
            plt.close()

            all_results[score_type] = epoch_results.to_dict()

        # 保存分析结果
        json_path = os.path.join(self.output_dir, "epoch_impact_analysis.json")
        pd.Series(all_results).to_json(
            json_path, indent=4, orient="index", force_ascii=False
        )
        return all_results

    def _plot_epoch_parameter_impact(self, param, impact_data, output_dir):
        """绘制参数对epoch的影响图"""
        df = pd.DataFrame(impact_data)
        df["param_label"] = df["param_value"].astype(str)
        df = df.sort_values(by="param_value")

        plt.figure(figsize=(12, 6))
        plt.bar(df["param_label"], df["mean_epoch"], alpha=0.7)

        # 添加误差线，显示最小和最大值
        plt.errorbar(
            df["param_label"],
            df["mean_epoch"],
            yerr=[
                df["mean_epoch"] - df["min_epoch"],
                df["max_epoch"] - df["mean_epoch"],
            ],
            fmt="o",
            color="black",
            ecolor="gray",
            capsize=5,
        )

        plt.xlabel(param)
        plt.ylabel("平均训练轮次")
        plt.title(f"{param}参数对训练轮次的影响")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{param}_epoch_impact.png"))
        plt.close()

    def analyze_model_fitting(self):
        """分析模型的过拟合/欠拟合情况，以权重法为基准"""
        # 创建拟合分析目录
        fitting_dir = os.path.join(self.output_dir, "fitting_analysis")
        os.makedirs(fitting_dir, exist_ok=True)

        score_types = ["MAP", "MRR"]
        all_results = {}

        for score_type in score_types:
            train_score_col = f"train_{score_type}_score"
            test_score_col = f"predict_{score_type}_score"
            fitting_status_col = f"fitting_status_{score_type}"

            # 检查评分列是否存在
            if train_score_col not in self.model_performance_df.columns:
                print(f"⚠ 警告: {train_score_col}列不存在，跳过{score_type}分析")
                continue

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

                if not fold_weights.empty and train_score_col in fold_weights.columns:
                    train_score = fold_weights[train_score_col].iloc[0]
                    test_score = fold_weights[test_score_col].iloc[0]
                    weights_scores[fold] = {"train": train_score, "test": test_score}

            # 筛选神经网络模型
            nn_models_df = self.model_performance_df[
                self.model_performance_df["model_category"] != "baseline_weights"
            ]

            # 计算每个模型相对于权重方法的拟合情况
            fitting_data = []
            fitting_status_map = {}

            for _, model in nn_models_df.iterrows():
                fold = model["fold_num"]
                if fold not in weights_scores:
                    continue

                if train_score_col not in model or test_score_col not in model:
                    continue

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

                model_id = model["model_id"]
                fitting_status_map[model_id] = fitting_status

                # 收集数据
                fitting_data.append(
                    {
                        "model_id": model["model_id"],
                        "model_category": model["model_category"],
                        "fold_num": fold,
                        "train_score": model[train_score_col],
                        "test_score": model[test_score_col],
                        "weights_train": weights_scores[fold]["train"],
                        "weights_test": weights_scores[fold]["test"],
                        "train_diff": train_diff,
                        "test_diff": test_diff,
                        "training_best_epoch": model.get("training_best_epoch", np.nan),
                        "fitting_status": fitting_status,
                    }
                )

            # 将拟合状态信息更新到主DataFrame中
            for model_id, status in fitting_status_map.items():
                # 使用.loc确保正确更新DataFrame
                self.model_performance_df.loc[
                    self.model_performance_df["model_id"] == model_id,
                    fitting_status_col,
                ] = status

            # 转换为DataFrame
            fitting_df = pd.DataFrame(fitting_data)

            if fitting_df.empty:
                print(f"⚠ 警告: 没有足够的数据进行{score_type}拟合分析")
                continue

            category_fitting = pd.crosstab(
                fitting_df["model_category"], fitting_df["fitting_status"]
            )

            # 可视化模型类别与拟合状态的关系
            plt.figure(figsize=(10, 6))
            category_fitting.plot(kind="bar", stacked=True)
            plt.xlabel("模型类别")
            plt.ylabel("模型数量")
            plt.title(f"不同模型类别的拟合状态分布 ({score_type})")
            plt.legend(title="拟合状态")
            plt.tight_layout()
            plt.savefig(os.path.join(fitting_dir, f"category_fitting_{score_type}.png"))
            plt.close()

            # 保存该评分类型的分析结果
            all_results[score_type] = {
                "fitting_distribution": fitting_df["fitting_status"]
                .value_counts()
                .to_dict(),
                "category_fitting": category_fitting.to_dict(),
                "model_details": fitting_df.to_dict(orient="records"),
            }

        # 保存所有分析结果
        json_path = os.path.join(self.output_dir, "fitting_analysis.json")
        pd.Series(all_results).to_json(
            json_path, indent=4, orient="index", force_ascii=False
        )
        return all_results

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

            # 动态找出最佳模型
            best_models = []
            for fold in self.model_performance_df["fold_num"].unique():
                fold_models = self.model_performance_df[
                    (self.model_performance_df["fold_num"] == fold)
                    & (
                        self.model_performance_df["model_category"]
                        != "baseline_weights"
                    )
                ]
                if not fold_models.empty and train_score_col in fold_models.columns:
                    best_idx = fold_models[train_score_col].idxmax()
                    best_models.append(self.model_performance_df.loc[best_idx])

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
        self.analyze_epoch_impact()

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

    # args = argparse.Namespace(
    #     log_dir="tomcat_auto_20250320041613/",
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
