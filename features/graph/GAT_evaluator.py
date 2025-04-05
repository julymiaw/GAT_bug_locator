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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
        """预处理和整合日志数据，保持自环模型和真实边模型的配对关系"""
        # 创建包含所有模型性能数据的DataFrame
        models_data = []

        # 添加权重法的结果
        weights_method_results = {}
        weights_method_train_scores = {}

        # 获取权重法的预测分数
        for fold_num in self.prediction_log.keys():
            if "weights_method" in self.prediction_log[fold_num]:
                weights_method_results[fold_num] = self.prediction_log[fold_num][
                    "weights_method"
                ]

        # 获取权重法的训练分数
        for fold_num in self.best_prescoring.keys():
            best_method = self.best_prescoring[fold_num]
            if best_method in self.prescoring_log.get(fold_num, {}):
                weights_method_train_scores[fold_num] = self.prescoring_log[fold_num][
                    best_method
                ]

        # 记录过滤状态
        filtered_stats = {
            "unconverged_models": 0,  # 训练阶段未收敛好的模型数
            "underperforming_models": 0,  # 预测阶段表现不佳的模型数
            "filtered_due_to_paired_model": 0,  # 因为配对模型被过滤而一同过滤的模型数
        }

        # 先收集所有模型的数据（不做过滤）
        all_models_by_fold = {}

        for fold_num in self.regression_log.keys():
            # 添加权重法的结果
            weights_model = {
                "fold": fold_num,
                "model_name": "weights_method",
                "train_score": weights_method_train_scores.get(fold_num),
                "test_score": weights_method_results.get(fold_num),
                "is_best": False,
                "model_category": "baseline_weights",
            }

            models_data.append(weights_model)

            # 初始化当前fold的模型列表
            fold_models = []

            # 获取当前fold的权重方法基线得分
            fold_weight_train_score = weights_method_train_scores.get(fold_num)
            fold_weight_test_score = weights_method_results.get(fold_num)

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
                    "should_filter": False,  # 标记是否应该被过滤
                    "filter_reason": None,  # 过滤原因
                    **parsed_params,
                }

                # 检查过滤条件1：如果训练阶段得分低于最佳权重方法(未正确收敛)
                if (
                    fold_weight_train_score is not None
                    and train_score < fold_weight_train_score
                ):
                    # model_info["should_filter"] = True
                    model_info["filter_reason"] = "unconverged"
                    # filtered_stats["unconverged_models"] += 1

                # 检查过滤条件2：如果预测阶段得分低于权重法(表现不佳)
                if (
                    fold_weight_test_score is not None
                    and test_score is not None
                    and test_score < fold_weight_test_score
                ):
                    model_info["should_filter"] = True
                    model_info["filter_reason"] = "underperforming"
                    filtered_stats["underperforming_models"] += 1

                fold_models.append(model_info)

            all_models_by_fold[fold_num] = fold_models

        # 找出自环模型和真实边模型的配对关系
        paired_models = {}

        for fold_num, fold_models in all_models_by_fold.items():
            fold_paired = {}

            # 先按不同类别分组
            selfloop_models = [
                m for m in fold_models if m.get("model_category") == "gat_selfloop"
            ]
            realedge_models = [
                m for m in fold_models if m.get("model_category") == "gat_realedge"
            ]

            # 配对自环模型和真实边模型
            for selfloop_model in selfloop_models:
                for realedge_model in realedge_models:
                    # 检查是否具有相同的超参数配置
                    if (
                        selfloop_model["loss_type"] == realedge_model["loss_type"]
                        and selfloop_model["hidden_dim"] == realedge_model["hidden_dim"]
                        and selfloop_model["penalty"] == realedge_model["penalty"]
                        and selfloop_model["alpha"] == realedge_model["alpha"]
                        and selfloop_model["dropout"] == realedge_model["dropout"]
                        and selfloop_model["learning_rate"]
                        == realedge_model["learning_rate"]
                        and selfloop_model["heads"] == realedge_model["heads"]
                    ):

                        # 记录配对关系
                        fold_paired[selfloop_model["model_name"]] = realedge_model[
                            "model_name"
                        ]
                        fold_paired[realedge_model["model_name"]] = selfloop_model[
                            "model_name"
                        ]
                        break

            paired_models[fold_num] = fold_paired

        # 根据配对关系应用过滤规则
        for fold_num, fold_models in all_models_by_fold.items():
            fold_paired = paired_models[fold_num]

            # 检查每个模型的配对模型是否应该被过滤
            for model in fold_models:
                paired_model_name = fold_paired.get(model["model_name"])

                if paired_model_name and not model["should_filter"]:
                    # 找到配对的模型
                    paired_model = next(
                        (
                            m
                            for m in fold_models
                            if m["model_name"] == paired_model_name
                        ),
                        None,
                    )

                    # 如果配对模型应该被过滤，则这个模型也应该被过滤
                    if paired_model and paired_model["should_filter"]:
                        # model["should_filter"] = True
                        model["filter_reason"] = (
                            f"paired_with_{paired_model['filter_reason']}"
                        )
                        filtered_stats["filtered_due_to_paired_model"] += 1

            # 只保留不应该被过滤的模型
            valid_fold_models = [m for m in fold_models if not m["should_filter"]]

            # 将有效模型添加到最终数据列表中
            for model in valid_fold_models:
                # 移除临时过滤标记
                if "should_filter" in model:
                    del model["should_filter"]
                if "filter_reason" in model:
                    del model["filter_reason"]

                models_data.append(model)

        # 创建DataFrame
        self.model_performance_df = pd.DataFrame(models_data)

        # 记录配对关系以供后续分析
        self.paired_models = paired_models

        # 打印过滤统计信息
        total_filtered = (
            filtered_stats["unconverged_models"]
            + filtered_stats["underperforming_models"]
            + filtered_stats["filtered_due_to_paired_model"]
        )

        print(f"✓ 成功处理 {len(self.model_performance_df)} 个模型配置的数据")
        print(f"🔍 过滤掉 {total_filtered} 个模型:")
        print(f"  - {filtered_stats['unconverged_models']} 个模型训练阶段未正确收敛")
        print(
            f"  - {filtered_stats['underperforming_models']} 个模型预测表现不如权重法"
        )
        print(
            f"  - {filtered_stats['filtered_due_to_paired_model']} 个模型因配对模型被过滤而一同移除"
        )

        # 记录过滤结果
        self.filtered_stats = filtered_stats

        return filtered_stats

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
        # 获取标记为最佳的模型
        best_models = self.model_performance_df[self.model_performance_df["is_best"]]

        # 如果没有找到最佳模型(可能是被过滤掉了)，处理这种情况
        if best_models.empty:
            print(
                "⚠ 警告: 所有标记为'最佳'的模型都被过滤掉了，使用剩余模型中表现最好的作为替代"
            )

            # 按照fold分组，为每个fold选择一个新的最佳模型
            for fold in self.model_performance_df["fold"].unique():
                fold_models = self.model_performance_df[
                    self.model_performance_df["fold"] == fold
                ]

                # 先尝试从非权重模型中选择
                fold_nn_models = fold_models[
                    fold_models["model_category"] != "baseline_weights"
                ]

                if not fold_nn_models.empty:
                    # 找出训练得分最高的非权重模型并标记为最佳
                    best_idx = fold_nn_models["train_score"].idxmax()
                    self.model_performance_df.loc[best_idx, "is_best"] = True
                    print(
                        f"  折 {fold}: 标记 {self.model_performance_df.loc[best_idx, 'model_name']} 为新的最佳模型"
                    )
                else:
                    print(f"  折 {fold}: 没有找到非权重模型")

            # 重新获取更新后的最佳模型列表
            best_models = self.model_performance_df[
                self.model_performance_df["is_best"]
            ]

        # 生成摘要报告
        summary = {
            "总模型数": len(self.model_performance_df),
            "过滤掉的模型数": self.filtered_stats.get("unconverged_models", 0)
            + self.filtered_stats.get("underperforming_models", 0),
            "最佳模型fold": best_models["fold"].tolist(),
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

        for i, model in enumerate(summary["最佳模型"]):
            print(f"\n折 {summary['最佳模型fold'][i]}:")
            print(f"  最佳模型: {model}")
            print(f"  训练得分: {summary['最佳模型训练得分'][i]:.4f}")
            print(f"  测试得分: {summary['最佳模型测试得分'][i]:.4f}")

        return summary

    def analyze_parameter_impact(self):
        """分析不同参数对模型性能的影响"""
        # 要分析的参数列表
        params_to_analyze = [
            "heads",
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

        nn_models_df = self.model_performance_df[
            self.model_performance_df["model_category"] != "baseline_weights"
        ]

        for param in params_to_analyze:
            if param not in nn_models_df.columns:
                continue

            unique_values = nn_models_df[param].unique()
            if len(unique_values) <= 1:
                continue

            # 计算每个参数值的平均性能
            impact_data = []
            unique_values = nn_models_df[param].unique()

            for value in unique_values:
                subset = nn_models_df[nn_models_df[param] == value]

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
        """分析自环模式对模型性能的影响，直接使用预处理阶段建立的配对关系"""
        # 确保model_category字段存在
        if "model_category" not in self.model_performance_df.columns:
            print("⚠ 数据中没有model_category信息，无法绘制分类模型对比图")
            return None

        # 初始化结果结构
        self_loop_stats = {}

        # 按fold分析
        for fold in self.model_performance_df["fold"].unique():
            # 获取当前fold的模型
            fold_models = self.model_performance_df[
                self.model_performance_df["fold"] == fold
            ]
            fold_self_loop = fold_models[
                fold_models["model_category"] == "gat_selfloop"
            ]
            fold_real_edge = fold_models[
                fold_models["model_category"] == "gat_realedge"
            ]

            # 使用预处理阶段已建立的配对关系
            fold_pairs = []

            # 检查是否有配对信息
            for _, self_loop_model in fold_self_loop.iterrows():
                model_name = self_loop_model["model_name"]

                # 获取对应的真实边模型名称
                if model_name in self.paired_models[fold]:
                    real_edge_name = self.paired_models[fold][model_name]

                    # 查找对应的真实边模型
                    real_edge_models = fold_real_edge[
                        fold_real_edge["model_name"] == real_edge_name
                    ]

                    if not real_edge_models.empty:
                        real_edge_model = real_edge_models.iloc[0]

                        pair_info = {
                            "self_loop_model": model_name,
                            "real_edge_model": real_edge_name,
                            "self_loop_train": self_loop_model["train_score"],
                            "real_edge_train": real_edge_model["train_score"],
                            "self_loop_test": self_loop_model["test_score"],
                            "real_edge_test": real_edge_model["test_score"],
                            "real_edge_improvement": real_edge_model["test_score"]
                            - self_loop_model["test_score"],
                        }

                        fold_pairs.append(pair_info)

            # 对结果按真实边模型对比自环模型的提升量降序排列
            fold_pairs.sort(key=lambda x: x["real_edge_improvement"], reverse=True)

            # 保存当前fold的统计信息
            self_loop_stats[fold] = fold_pairs

        # 保存自环分析结果
        with open(os.path.join(self.output_dir, "self_loops_analysis.json"), "w") as f:
            json.dump(
                self_loop_stats, f, indent=4, ensure_ascii=False, cls=NumpyEncoder
            )

        # 绘制不同类别模型的对比散点图
        self.plot_model_categories_comparison(self_loop_stats)

        return self_loop_stats

    def plot_model_categories_comparison(self, fold_comparisons):
        """
        绘制不同类别模型的对比散点图(MLP, GAT自环, GAT真实边)

        参数:
            fold_comparisons: 按fold组织的比较数据字典
        """
        if not fold_comparisons:
            return

        # 准备数据
        all_models = self.model_performance_df.copy()

        # 创建散点图
        plt.figure(figsize=(12, 10))

        # 为每个fold分配不同颜色
        folds = all_models["fold"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(folds)))
        fold_color_map = {fold: colors[i] for i, fold in enumerate(folds)}

        legend_elements = []

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

        # 绘制所有模型点
        for fold in folds:
            fold_models = all_models[all_models["fold"] == fold]

            # 按模型类别绘制
            for category, marker in markers.items():
                category_models = fold_models[fold_models["model_category"] == category]

                if not category_models.empty:
                    plt.scatter(
                        category_models["train_score"],
                        category_models["test_score"],
                        marker=marker,
                        s=80,
                        color=fold_color_map[fold],
                        alpha=0.7,
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

        # 标记最佳模型
        best_models = all_models[all_models["is_best"]]
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="red",
                markerfacecolor="none",
                markersize=12,
                label="最佳模型",
                markeredgewidth=2,
            )
        )
        for _, model in best_models.iterrows():
            category = model["model_category"]
            marker = markers[category]
            plt.scatter(
                model["train_score"],
                model["test_score"],
                marker=marker,
                s=120,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                label="_nolegend_",
            )

        # 用虚线连接配对的模型 - 使用按fold组织的配对模型
        for fold, fold_data in fold_comparisons.items():
            for pair in fold_data:
                self_loop_model = all_models[
                    (all_models["model_name"] == pair["self_loop_model"])
                    & (all_models["fold"] == fold)
                ]
                real_edge_model = all_models[
                    (all_models["model_name"] == pair["real_edge_model"])
                    & (all_models["fold"] == fold)
                ]

                if not self_loop_model.empty and not real_edge_model.empty:
                    plt.plot(
                        [
                            self_loop_model.iloc[0]["train_score"],
                            real_edge_model.iloc[0]["train_score"],
                        ],
                        [
                            self_loop_model.iloc[0]["test_score"],
                            real_edge_model.iloc[0]["test_score"],
                        ],
                        "k--",  # 黑色虚线
                        alpha=0.5,
                        linewidth=0.7,
                    )

        plt.axis("equal")

        # 添加辅助元素
        plt.xlabel("回归阶段得分", fontsize=12)
        plt.ylabel("测试阶段得分", fontsize=12)
        plt.title("不同类别模型性能对比", fontsize=14)

        plt.legend(handles=legend_elements, loc="best")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_categories_comparison.png"))
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

        # 按模型类别分组计算最佳性能
        fold_category_performance = (
            self.model_performance_df.groupby(["fold", "model_category"])["test_score"]
            .max()
            .reset_index()
        )

        # 辅助函数：判断一个模型类别是否在所有fold中都优于另一个类别
        def is_better_in_all_folds(category1, category2):
            """判断category1是否在所有fold中都优于category2"""
            for fold in fold_category_performance["fold"].unique():
                # 获取当前fold中两种类别的最高得分
                cat1_score = fold_category_performance[
                    (fold_category_performance["fold"] == fold)
                    & (fold_category_performance["model_category"] == category1)
                ]["test_score"]

                cat2_score = fold_category_performance[
                    (fold_category_performance["fold"] == fold)
                    & (fold_category_performance["model_category"] == category2)
                ]["test_score"]

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
                return f"GAT模型表现优于{name}"
            elif is_better_in_all_folds(category, "gat_realedge"):
                return f"{name}表现优于GAT模型"
            else:
                return f"GAT模型与{name}表现无法确定明显优劣"

        # 执行各种比较
        findings = [
            compare_categories("baseline_weights", "权重法"),
            compare_categories("gat_selfloop", "仅自环GAT模型"),
            compare_categories("baseline_mlp", "MLP模型"),
        ]

        # 整体性能统计
        category_performance = (
            self.model_performance_df.groupby("model_category")["test_score"]
            .agg(["max", "mean"])
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
                    "baseline_weights": "权重线性组合模型",
                    "gat_realedge": "GAT模型",
                    "gat_selfloop": "仅自环GAT模型",
                }.get(best_model_category, "未知类型"),
            },
            "模型类别性能": category_performance.to_dict(orient="records"),
            "主要发现": findings,
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
