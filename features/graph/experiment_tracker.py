#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实验跟踪管理器模块

用途：提供一个中心化的数据结构来存储、管理和序列化ML实验的完整生命周期
特点：
    - 跟踪模型参数、训练过程和评估结果
    - R支持自动从模型实例提取参数
    - 记录训练时间和数据规模统计
    - 提供DataFrame和JSON格式的导出
    - 分离模型名称和参数管理，使参数扩展更灵活
    - 支持多折交叉验证结果聚合与比较

作者: SRTP团队
日期: 2025年4月7日
"""

import datetime
import json
import numpy as np
import pandas as pd

from GATRegressor import ModelParameters


class ExperimentTracker:
    """实验跟踪管理器，用于记录机器学习实验的完整周期"""

    def __init__(self):
        """初始化模型注册表"""
        self.model_params = {}  # 模型参数字典 {model_id: {param_name: param_value}}
        self.model_results = {}  # 模型结果字典 {model_id: {metric_name: metric_value}}
        self.training_summaries = {}  # {model_id: training_summary_dict}

        self.model_counter = 0  # 模型计数器
        self.fold_counters = {}  # 每个折的单独计数器

        self.training_time_stats = {}

        # 实验元数据
        self.metadata = {
            "total_models": 0,
            "timestamp": None,
            "experiment_name": "adaptive_gat_experiment",
        }

    def register_model(self, model, fold_num):
        """
        注册模型及其参数

        参数:
            model: 模型实例
            fold_num: 当前折号

        返回:
            model_id: 模型唯一标识符
        """
        # 如果这是新的折，初始化该折的计数器
        if fold_num not in self.fold_counters:
            self.fold_counters[fold_num] = 0

        # 生成唯一ID
        counter = self.fold_counters[fold_num]
        model_id = f"model_fold{fold_num}_{counter}"

        self.fold_counters[fold_num] += 1
        self.model_counter += 1

        # 自动从模型实例提取参数
        params = {}
        for param_name in ModelParameters.get_all_params():
            if hasattr(model, param_name):
                params[param_name] = getattr(model, param_name)

        # 添加特殊参数
        params["model_type"] = "MLP" if model.heads is None else "GAT"
        params["fold_num"] = fold_num

        # 将model_id添加到模型对象中，方便后续使用
        setattr(model, "model_id", model_id)

        self.model_params[model_id] = params
        self.model_results[model_id] = {}
        self.training_summaries[model_id] = {}

        return model_id

    def update_result(self, model_id, metric_name, metric_value):
        """
        更新模型结果

        参数:
            model_id: 模型唯一标识符
            metric_name: 指标名称
            metric_value: 指标值
        """
        if model_id not in self.model_results:
            raise ValueError(f"模型 {model_id} 未注册，不能更新结果！")

        self.model_results[model_id][metric_name] = metric_value

    def update_training_summary(self, model_id, training_summary):
        """
        更新模型的训练摘要信息

        参数:
            model_id: 模型唯一标识符
            training_summary: 包含训练终止信息的字典
        """
        if model_id not in self.training_summaries:
            raise ValueError(f"模型 {model_id} 未注册，不能更新训练摘要！")

        self.training_summaries[model_id] = training_summary

    def record_training_time(self, fold_num, time_seconds, bugs_count, files_count):
        """
        记录特定折的训练时间统计

        参数:
            fold_num: 折号
            time_seconds: 训练时间(秒)
            bugs_count: Bug数量
            files_count: 文件数量
        """
        self.training_time_stats[str(fold_num)] = {
            "time": time_seconds,
            "bugs": bugs_count,
            "files": files_count,
            "seconds_per_bug": time_seconds / bugs_count if bugs_count > 0 else 0,
            "seconds_per_file": time_seconds / files_count if files_count > 0 else 0,
        }

    def are_models_matching(self, model_id1, model_id2):
        """
        判断两个模型是否是参数匹配的配对模型（一个自环模型，一个真实边模型）

        参数:
            model_id1: 第一个模型的ID
            model_id2: 第二个模型的ID

        返回:
            bool: 如果模型参数匹配（除model_id和use_self_loops_only外），返回True，否则返回False
        """
        # 检查两个模型ID是否都存在
        if model_id1 not in self.model_params or model_id2 not in self.model_params:
            return False

        # 获取两个模型的参数
        params1 = self.model_params[model_id1]
        params2 = self.model_params[model_id2]

        # 确保两个模型都是GAT类型
        if params1.get("model_type") != "GAT" or params2.get("model_type") != "GAT":
            return False

        # 检查use_self_loops_only参数 - 一个必须是True，一个必须是False
        if not (
            (
                params1.get("use_self_loops_only") is True
                and params2.get("use_self_loops_only") is False
            )
            or (
                params1.get("use_self_loops_only") is False
                and params2.get("use_self_loops_only") is True
            )
        ):
            return False

        # 检查其他所有参数是否相同
        for param_name in set(params1.keys()).union(set(params2.keys())):
            # 跳过model_id和use_self_loops_only参数
            if param_name in ["model_id", "use_self_loops_only"]:
                continue

            # 如果任何参数不相同，返回False
            if param_name not in params1 or param_name not in params2:
                return False

            if params1[param_name] != params2[param_name]:
                return False

        # 所有参数匹配
        return True

    def to_dataframe(self):
        """
        将模型参数和结果转换为DataFrame

        返回:
            pd.DataFrame: 包含所有模型参数和结果的DataFrame
        """
        all_data = []

        for model_id in self.model_params:
            # 合并参数和结果
            model_data = {
                "model_id": model_id,
                **self.model_params[model_id],
                **self.model_results[model_id],
            }

            # 添加训练摘要字段
            if model_id in self.training_summaries:
                summary = self.training_summaries[model_id]

                if "stop_reason" in summary:
                    model_data["stop_reason"] = summary["stop_reason"]

                # 计算最佳分数和轮次
                if "all_scores" in summary and summary["all_scores"]:
                    best_score_idx = np.argmax(summary["all_scores"])
                    model_data["best_score"] = summary["all_scores"][best_score_idx]
                    model_data["best_epoch"] = best_score_idx + 1
                    model_data["final_epoch"] = len(summary["all_scores"])
                    model_data["final_score"] = summary["all_scores"][-1]

                # 添加权重分数和性能比率
                if "weight_score" in summary:
                    model_data["performance_ratio"] = (
                        model_data["final_score"] / summary["weight_score"]
                    )

            all_data.append(model_data)

        # 创建DataFrame
        if all_data:
            return pd.DataFrame(all_data)
        else:
            return pd.DataFrame()

    def to_json(self, file_path):
        """
        将模型参数和结果保存为JSON文件

        参数:
            file_path: JSON文件路径
        """

        # 更新元数据
        self.metadata["total_models"] = self.model_counter
        self.metadata["timestamp"] = datetime.datetime.now().isoformat()

        data = {
            "models": {},
            "training_time": self.training_time_stats,
            "metadata": self.metadata,
        }

        for model_id in self.model_params:
            data["models"][model_id] = {
                "params": self._convert_to_json_serializable(
                    self.model_params[model_id]
                ),
                "results": self._convert_to_json_serializable(
                    self.model_results.get(model_id, {})
                ),
                "training_summary": self._convert_to_json_serializable(
                    self.training_summaries.get(model_id, {})
                ),
            }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def from_json(self, file_path):
        """
        从JSON文件加载模型参数和结果

        参数:
            file_path: JSON文件路径
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        if "models" in data:
            for model_id, model_data in data["models"].items():
                self.model_params[model_id] = model_data.get("params", {})
                self.model_results[model_id] = model_data.get("results", {})
                self.training_summaries[model_id] = model_data.get(
                    "training_summary", {}
                )

            # 加载元数据
            if "metadata" in data:
                self.metadata = data["metadata"]
                if "total_models" in data["metadata"]:
                    self.model_counter = data["metadata"]["total_models"]

            # 加载训练时间统计
            if "training_time" in data:
                self.training_time_stats = data["training_time"]

    def _convert_to_json_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
