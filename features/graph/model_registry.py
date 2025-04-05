#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型参数与结果注册表模块

用途：提供一个中心化的数据结构来存储、管理和序列化模型参数及其评估结果
特点：
    - 支持自动从模型实例提取参数
    - 提供DataFrame和JSON格式的导出
    - 分离模型名称和参数管理，使参数扩展更灵活
    - 支持模型结果追踪，适用于多折交叉验证场景

作者: SRTP团队
日期: 2025年4月4日
"""

import json
import numpy as np
import pandas as pd


class ModelRegistry:
    """模型参数与结果注册表"""

    def __init__(self):
        """初始化模型注册表"""
        self.models = {}  # 模型实例字典 {model_id: model_instance}
        self.model_params = {}  # 模型参数字典 {model_id: {param_name: param_value}}
        self.model_results = {}  # 模型结果字典 {model_id: {metric_name: metric_value}}
        self.model_counter = 0  # 模型计数器，用于生成唯一ID

    def register_model(self, model, fold_num):
        """
        注册模型及其参数

        参数:
            model: 模型实例
            fold_num: 当前折号

        返回:
            model_id: 模型唯一标识符
        """
        # 生成唯一ID，如果提供了fold_num，则包含在ID中
        model_id = f"model_fold{fold_num}_{self.model_counter}"
        self.model_counter += 1

        # 存储模型实例
        self.models[model_id] = model

        # 自动从模型实例提取参数
        params = {
            "hidden_dim": getattr(model, "hidden_dim", None),
            "heads": getattr(model, "heads", None),
            "dropout": getattr(model, "dropout", None),
            "alpha": getattr(model, "alpha", None),
            "loss": getattr(model, "loss", None),
            "penalty": getattr(model, "penalty", None),
            "lr": getattr(model, "lr", None),
            "use_self_loops_only": getattr(model, "use_self_loops_only", None),
            "early_stop": getattr(model, "early_stop", None),
            "max_iter": getattr(model, "max_iter", None),
            "n_iter_no_change": getattr(model, "n_iter_no_change", None),
            "shuffle": getattr(model, "shuffle", None),
            "validation_fraction": getattr(model, "validation_fraction", None),
            "model_type": "MLP" if getattr(model, "heads", None) is None else "GAT",
            "fold_num": fold_num,
        }

        # 将model_id添加到模型对象中，方便后续使用
        setattr(model, "model_id", model_id)

        self.model_params[model_id] = params
        self.model_results[model_id] = {}

    def update_result(self, model_id, metric_name, metric_value):
        """
        更新模型结果

        参数:
            model_id: 模型唯一标识符
            metric_name: 指标名称
            metric_value: 指标值
        """
        if model_id not in self.model_results:
            self.model_results[model_id] = {}

        self.model_results[model_id][metric_name] = metric_value

    def update_training_info(self, model_id, best_epoch=None, best_val_score=None):
        """
        更新模型训练信息

        参数:
            model_id: 模型唯一标识符
            best_epoch: 最佳验证表现的轮次
            best_val_score: 最佳验证分数
        """
        if model_id not in self.model_results:
            self.model_results[model_id] = {}

        if best_epoch is not None:
            self.model_results[model_id]["best_epoch"] = best_epoch

        if best_val_score is not None:
            self.model_results[model_id]["best_validation_score"] = best_val_score

    def get_model(self, model_id):
        """获取模型实例"""
        return self.models.get(model_id)

    def get_params(self, model_id):
        """获取模型参数"""
        return self.model_params.get(model_id)

    def get_results(self, model_id):
        """获取模型结果"""
        return self.model_results.get(model_id)

    def get_best_model(self, metric="train_map_score", fold_num=None, model_type=None):
        """
        获取基于指定指标的最佳模型

        参数:
            metric: 评估指标名称
            fold_num: 如果提供，则仅在特定折中查找
            model_type: 如果提供，则仅查找特定类型的模型（'MLP'或'GAT'）

        返回:
            (model_id, model): 最佳模型的ID和模型实例
        """
        best_score = -float("inf")
        best_model_id = None

        for model_id in self.model_results:
            # 如果指定了fold_num，检查是否匹配
            if (
                fold_num is not None
                and self.model_params[model_id].get("fold_num") != fold_num
            ):
                continue

            # 如果指定了model_type，检查是否匹配
            if (
                model_type is not None
                and self.model_params[model_id].get("model_type") != model_type
            ):
                continue

            # 检查是否有指定的评估指标
            if metric in self.model_results[model_id]:
                score = self.model_results[model_id][metric]
                if score > best_score:
                    best_score = score
                    best_model_id = model_id

        if best_model_id:
            return best_model_id, self.models[best_model_id]
        return None, None

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
                **self.model_results.get(model_id, {}),
            }
            all_data.append(model_data)

        # 创建DataFrame
        if all_data:
            return pd.DataFrame(all_data)
        else:
            return pd.DataFrame()

    def filter_models(self, **kwargs):
        """
        根据参数筛选模型并返回DataFrame

        参数:
            **kwargs: 键值对形式的筛选条件

        返回:
            pd.DataFrame: 筛选后的模型DataFrame
        """
        df = self.to_dataframe()
        if df.empty:
            return df

        # 应用每个筛选条件
        for key, value in kwargs.items():
            if key in df.columns:
                df = df[df[key] == value]

        return df

    def to_json(self, file_path):
        """
        将模型参数和结果保存为JSON文件

        参数:
            file_path: JSON文件路径
        """
        data = {
            "models": {},
            "metadata": {
                "total_models": self.model_counter,
                "timestamp": pd.Timestamp.now().isoformat(),
            },
        }

        for model_id in self.model_params:
            data["models"][model_id] = {
                "params": self._convert_to_json_serializable(
                    self.model_params[model_id]
                ),
                "results": self._convert_to_json_serializable(
                    self.model_results.get(model_id, {})
                ),
            }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

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

            if "metadata" in data and "total_models" in data["metadata"]:
                self.model_counter = data["metadata"]["total_models"]

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

    def clear(self):
        """清空注册表"""
        self.models.clear()
        self.model_params.clear()
        self.model_results.clear()
        # 保留计数器值
