#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <feature_files_prefix>

Requires results of save_normalized_fold_dataframes.py
"""

import os
import json
import time
import argparse
from collections import defaultdict
from itertools import product
from timeit import default_timer

import gc
from typing import Dict, List
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from skopt import load

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.optim import AdamW
import torch.nn.functional as F
from torch_geometric.data import Data

from metrics import (
    calculate_metric_results,
    print_metrics,
    eval_weights,
    evaluate_fold,
    fold_check,
)
from train_utils import eprint
from ranking_losses import WeightedRankMSELoss
from experiment_tracker import ExperimentTracker

from weight_functions import (
    get_weights_methods,
    weights_on_df,
)

node_feature_columns = ["f" + str(i) for i in range(1, 20)]
edge_feature_columns = ["t" + str(i) for i in range(1, 13)]


class GATModule(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, heads, dropout):
        """
        GAT模型

        参数:
            node_dim: 节点特征维度
            edge_dim: 边特征维度
            hidden_dim: 隐藏层维度
            heads: 注意力头数，若为None则不使用GAT层
            dropout: Dropout率
        """
        super(GATModule, self).__init__()

        # 节点特征变换
        self.node_lin = nn.Linear(node_dim, hidden_dim)
        self.use_gat = heads is not None

        self.use_global_node = True

        # 边特征变换为权重
        if self.use_gat:
            # 边特征变换为权重
            self.edge_lin = nn.Linear(edge_dim, 1)

            # 多层GAT - 适应2跳距离的修复文件
            self.gat_layers = nn.ModuleList(
                [
                    GATConv(
                        hidden_dim,
                        hidden_dim,
                        heads=heads,
                        dropout=dropout,
                        add_self_loops=True,
                        fill_value=1.0,
                    ),
                    GATConv(
                        hidden_dim * heads,
                        hidden_dim,
                        heads=heads,
                        dropout=dropout,
                        add_self_loops=True,
                        fill_value=1.0,
                    ),
                ]
            )

            self.dropout = nn.Dropout(dropout)

            # 残差连接层
            self.residual_proj = nn.Linear(hidden_dim, hidden_dim * heads)

            # GAT流的输出层
            self.gat_out = nn.Linear(hidden_dim * heads, hidden_dim)
        else:
            # MLP流 - 仅处理节点特征
            self.mlp_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )

        # 全局信息聚合机制
        if self.use_global_node:
            self.global_attn = nn.Linear(hidden_dim, 1)
            self.global_proj = nn.Linear(hidden_dim, hidden_dim)

        # 预测层
        self.out_lin = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        # 节点特征变换
        x = self.node_lin(x)
        h = F.relu(x)

        if self.use_gat:
            # 边特征处理
            edge_weights = torch.sigmoid(self.edge_lin(edge_attr)).squeeze(-1) * 2

            # 第一层GAT处理
            h1 = self.gat_layers[0](h, edge_index, edge_attr=edge_weights)
            h1 = F.relu(h1)
            h1 = self.dropout(h1)

            # 第二层GAT处理(增加消息传递迭代次数)
            h2 = self.gat_layers[1](h1, edge_index, edge_attr=edge_weights)
            h2 = F.relu(h2)
            h2 = self.dropout(h2)

            # 残差连接(避免过平滑)
            h_res = self.residual_proj(h)
            x = h2 + h_res
            x = self.gat_out(x)
        else:
            # 无GAT时使用MLP
            x = self.mlp_stream(h)

        # 添加全局信息聚合(处理分散的连通分量)
        if self.use_global_node and x.size(0) > 1:
            # 计算节点注意力权重
            attn_weights = torch.softmax(self.global_attn(x), dim=0)
            # 全局图表示
            global_repr = torch.sum(x * attn_weights, dim=0, keepdim=True)
            # 全局信息处理
            global_info = self.global_proj(global_repr)
            # 将全局信息广播到所有节点
            x = x + 2.0 * global_info

        # 最终层
        x = self.out_lin(x)

        return x


class GATRegressor:
    """
    用于bug定位的GAT回归器，提供与scikit-learn兼容的接口
    """

    def __init__(
        self,
        node_feature_columns,
        dependency_feature_columns,
        hidden_dim=16,
        heads=None,
        dropout=0.3,
        alpha=0.0001,
        loss="MSE",
        penalty="l2",
        max_iter=500,
        tol=1e-4,
        shuffle=True,
        epsilon=0.1,
        random_state=42,
        lr=0.005,
        warm_start=False,
        n_iter_no_change=5,
        use_self_loops_only=False,
        early_stop=False,
        validation_fraction=0.2,
        metric_type="MRR",
    ):
        """
        初始化回归器

        参数:
            node_feature_columns: 用于训练的节点特征列名列表
            dependency_feature_columns: 用于训练的边特征列名列表
            hidden_dim: GAT隐藏层维度
            heads: 注意力头数，若为None则不使用GAT层
            dropout: Dropout率
            alpha: L2正则化系数
            loss: 损失函数类型，可选'MSE','Huber'或'WeightedMSE'
            penalty: 正则化类型，可选'l2'或None
            max_iter: 最大迭代次数（训练轮数）
            tol: 收敛容差，用于提前停止迭代
            shuffle: 是否在每轮训练后打乱数据
            epsilon: Huber损失中的epsilon参数
            random_state: 随机种子
            lr: AdamW优化器的学习率
            warm_start: 是否使用之前的解作为初始化
            n_iter_no_change: 用于提前停止的无改进迭代次数
            use_self_loops_only: 是否仅使用自环边
            early_stop: 是否启用早停机制
            validation_fraction: 用于早停的验证集比例
            metric_type: 评估指标类型，可选值为"MAP"或"MRR"
        """
        self.node_feature_columns = node_feature_columns
        self.dependency_feature_columns = dependency_feature_columns
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.epsilon = epsilon
        self.random_state = random_state
        self.lr = lr
        self.warm_start = warm_start
        self.n_iter_no_change = n_iter_no_change
        self.use_self_loops_only = use_self_loops_only
        self.early_stop = early_stop
        self.validation_fraction = validation_fraction
        self.metric_type = metric_type

        # 模型注册时进行初始化
        self.model_id: str = None

        # 初始化模型和累积器
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_epoch = 0

        # 设置随机种子
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def fit(self, node_features, edge_features, score):
        """
        训练模型

        参数:
            node_features: 包含节点特征的DataFrame
            edge_features: 包含边特征的DataFrame
            score: 目标分数，形状与node_features中的节点数匹配

        返回:
            self: 训练后的模型
        """
        # 复制节点特征DataFrame并添加分数
        node_features = node_features.copy(deep=False)
        node_features["score"] = score

        # 准备数据
        data_list = self._prepare_data(node_features, edge_features)

        # 如果启用了早停，需要分割训练集和验证集
        if self.early_stop and len(data_list) > 5:  # 确保有足够的数据用于分割
            # 按bug ID划分，保持同一bug的数据在同一集合中
            bug_ids = list(set(data.bug_id for data in data_list))
            np.random.shuffle(bug_ids)

            # 计算验证集大小
            val_size = max(1, int(len(bug_ids) * self.validation_fraction))
            val_bugs = bug_ids[:val_size]
            train_bugs = bug_ids[val_size:]

            # 分割数据
            train_data = [data for data in data_list if data.bug_id in train_bugs]
            val_data = [data for data in data_list if data.bug_id in val_bugs]

            # 如果验证集为空，使用一部分训练集作为验证集
            if not val_data and train_data:
                val_data = [train_data[-1]]
                train_data = train_data[:-1]
        else:
            train_data = data_list
            val_data = []

        # 如果不使用warm_start或第一次训练，则重新初始化模型
        if not self.warm_start or self.model is None:
            node_dim = len(self.node_feature_columns)
            edge_dim = len(self.dependency_feature_columns)

            self.model = GATModule(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=self.hidden_dim,
                heads=self.heads,
                dropout=self.dropout,
            ).to(self.device)

            self.iter_ = 0

        # 选择损失函数
        self.criterion = self._get_criterion()

        # 创建优化器
        if self.penalty == "l2":
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.alpha
            )
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

        # 加载数据到设备
        train_data = [data.to(self.device) for data in train_data]
        val_data = [data.to(self.device) for data in val_data] if val_data else []

        # 确保train_data不为空
        if not train_data:
            eprint("错误: 没有有效的训练数据")
            return self

        # 训练循环
        self.model.train()
        best_val_score = -float("inf")  # 用于早停的最佳验证分数
        best_train_loss = float("inf")  # 用于损失收敛的最佳训练损失
        no_improvement_count = 0
        best_weights = None

        for epoch in range(self.max_iter):
            # 如果需要，打乱数据
            if self.shuffle:
                np.random.shuffle(train_data)

            # 训练一轮
            total_loss = 0
            self.optimizer.zero_grad()

            for data in train_data:
                # 前向传播
                out = self.model(data.x, data.edge_index, data.edge_attr)
                # 计算损失
                loss = self.criterion(out, data.y)
                # 反向传播
                loss.backward()
                total_loss += loss.item()

            # 更新参数
            self.optimizer.step()

            avg_train_loss = total_loss / len(train_data)

            # 早停检查 -----------------------------------------------
            improved = False

            # 验证阶段 - 早停检查
            if self.early_stop and val_data:
                self.model.eval()
                val_score = self._evaluate_validation(val_data, self.metric_type)
                self.model.train()

                if val_score > best_val_score:
                    best_val_score = val_score
                    improved = True
                    self.best_validation_score = val_score
                    self.best_epoch = epoch
                    best_weights = {
                        name: param.clone().detach()
                        for name, param in self.model.state_dict().items()
                    }
            elif epoch == 0 or best_train_loss - avg_train_loss > self.tol:
                best_train_loss = avg_train_loss
                improved = True
                self.best_epoch = epoch
                best_weights = {
                    name: param.clone().detach()
                    for name, param in self.model.state_dict().items()
                }

            # 更新无改进计数
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # 检查是否应该提前停止
            if (
                self.n_iter_no_change is not None
                and no_improvement_count >= self.n_iter_no_change
            ):
                break

        # 恢复最佳权重
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return self

    def _prepare_data(self, node_features, edge_features):
        """
        准备PyG数据对象

        参数:
            node_features: 包含节点特征的DataFrame
            edge_features: 包含边特征的DataFrame

        返回:
            data_list: 每个bug对应的PyG Data对象列表
        """
        data_list: List[Data] = []

        for bug_id in node_features.index.get_level_values(0).unique():
            if bug_id not in node_features.index:
                eprint(f"警告: bug ID {bug_id} 在节点数据索引中不存在，跳过")
                continue
            # 获取该bug的节点特征
            bug_nodes = node_features.loc[bug_id]

            # 获取文件ID列表及其映射
            file_ids = bug_nodes.index.tolist()
            if len(file_ids) == 0:
                eprint(f"警告: bug ID {bug_id} 没有关联文件，跳过")
                continue
            file_to_idx = {file_id: idx for idx, file_id in enumerate(file_ids)}

            # 准备节点特征
            x = torch.tensor(
                bug_nodes[self.node_feature_columns].values, dtype=torch.float
            )

            # 准备目标值
            y = torch.tensor(bug_nodes["score"].values, dtype=torch.float).reshape(
                -1, 1
            )

            if self.heads is None:
                # MLP模式：不需要任何边数据
                edge_index = None
                edge_attr = None
            elif self.use_self_loops_only:
                # 使用空边集，让GATConv自己添加自环
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, len(self.dependency_feature_columns)))
            else:
                # 获取该bug的边信息
                bug_edges = edge_features.loc[bug_id]

                # 使用真实边数据
                source_indices = bug_edges["source"].map(file_to_idx).values
                target_indices = bug_edges["target"].map(file_to_idx).values

                edge_index = torch.tensor(
                    np.vstack([source_indices, target_indices]), dtype=torch.long
                )
                edge_attr = torch.tensor(
                    bug_edges[self.dependency_feature_columns].values,
                    dtype=torch.float,
                )
            # 创建数据对象
            data = Data(
                x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, bug_id=bug_id
            )
            data_list.append(data)

        return data_list

    def _get_criterion(self):
        """根据loss参数选择损失函数"""
        if self.loss == "MSE":
            return nn.MSELoss()
        elif self.loss == "Huber":
            return nn.HuberLoss(delta=self.epsilon)
        elif self.loss == "WeightedMSE":
            return WeightedRankMSELoss(fix_weight=5.0)
        else:
            return nn.MSELoss()  # 默认使用MSE

    def _evaluate_validation(self, val_data):
        """
        评估验证集性能，返回评估指标

        参数:
            val_data: 验证集数据

        返回:
            float: 平均性能得分(越高越好)
        """
        with torch.no_grad():
            # 为每个bug创建一个临时DataFrame，最后合并
            temp_dfs = []

            for data in val_data:
                # 获取预测值
                out = self.model(data.x, data.edge_index, data.edge_attr)

                # 获取真实标签和预测值
                y_true = data.y.cpu().numpy().flatten()
                y_pred = out.cpu().numpy().flatten()

                # 只有存在真实修复文件(y_true>0)的数据才有评估意义
                if np.any(y_true > 0):
                    n_files = len(y_true)
                    bug_id = data.bug_id.item()  # 获取当前bug的ID

                    # 创建包含结果和标签的DataFrame
                    temp_df = pd.DataFrame(
                        {"result": y_pred, "used_in_fix": (y_true > 0).astype(float)},
                        index=pd.Index([bug_id] * n_files, name="bug_id"),
                    )

                    temp_dfs.append(temp_df)

            # 如果没有找到有修复文件的bug，返回0
            if not temp_dfs:
                return 0.0

            all_bugs_df = pd.concat(temp_dfs)
            score = calculate_metric_results(all_bugs_df, metric_type=self.metric_type)

            return score

    def predict(self, node_features, edge_features):
        """
        使用训练好的模型进行预测

        参数:
            node_features: 包含节点特征的DataFrame
            edge_features: 包含边特征的DataFrame

        返回:
            predictions: 预测分数的numpy数组
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用fit方法")

        # 复制节点特征DataFrame并添加虚拟分数
        node_features = node_features.copy(deep=False)
        node_features["score"] = 0  # 添加虚拟分数

        # 准备数据
        data_list = self._prepare_data(node_features, edge_features)

        # 预测
        self.model.eval()

        # 创建一个与节点数相同大小的数组
        total_nodes = sum(len(data.x) for data in data_list)
        predictions = np.zeros(total_nodes)

        idx = 0
        with torch.no_grad():
            for data in data_list:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr)
                pred = out.numpy().flatten()
                predictions[idx : idx + len(pred)] = pred
                idx += len(pred)

        return predictions


class Adaptive_Process(object):
    """
    自适应缺陷定位算法核心类，通过动态选择特征权重计算、回归模型和特征筛选策略优化结果。
    实现多阶段交叉验证和并行计算以提升效率，支持模型持久化和性能日志记录。

    主要流程：
        1. 权重计算阶段：通过多种统计/机器学习方法评估特征重要性
        2. 模型选择阶段：遍历（评分方法 × 回归模型）组合寻找最优配置
        3. 预测阶段：根据训练结果选择线性权重组合或回归模型进行预测

    设计特性：
        - 通过enforce_relearning控制是否复用历史模型
        - 在权重计算阶段使用交叉验证
        - 使用joblib并行化权重计算和模型评估过程
    """

    def __init__(self, metric_type="MRR"):
        """
        初始化方法，配置所有可用算法组件

        参数：
            metric_type: 使用的评估指标类型，可选值:
                "MAP": 平均精确率（Mean Average Precision）
                "MRR": 平均倒数排名（Mean Reciprocal Rank）
        """
        # region 算法组件配置
        # 特征权重计算方法列表（统计检验/树模型/降维方法等）
        self.weights_methods = get_weights_methods()

        # 指定评估指标类型
        self.metric_type = metric_type
        if metric_type not in ["MAP", "MRR"]:
            eprint(f"警告: 未知的评估指标类型 '{metric_type}'，使用 'MRR' 模式")
            self.metric_type = "MRR"

        # 回归模型集合
        self.reg_models: List[GATRegressor] = []
        self.reg_models.extend(get_skmodels(self.metric_type))

        # 评分方法（标准评分）
        self.score_method = normal_score
        # endregion

        # region 运行时状态存储
        self.weights = None  # 当前最佳特征权重向量（np.ndarray）
        self.weights_score = 0  # 当前最佳权重方法得分
        self.reg_model = None  # 当前选择的回归模型对象
        self.all_reg_models = {}  # 所有回归模型对象
        self.reg_model_score = 0  # 当前最佳回归模型得分
        # endregion

        # region 配置参数
        self.name = "Adaptive"  # 算法标识名
        self.use_prescoring_always = False  # 是否始终使用预评分权重
        self.use_reg_model_always = True  # 是否强制使用回归模型
        self.use_prescoring_cross_validation = True  # 权重计算阶段交叉验证开关
        self.use_training_cross_validation = True
        self.cross_validation_fold_number = 5  # 交叉验证折数
        # endregion

        self.experiment_tracker = ExperimentTracker()

    def train(self, df: pd.DataFrame, dependency_df: pd.DataFrame, fold_num):
        """
        训练入口函数

        参数：
            df (pd.DataFrame): 当前折的训练数据
            dependency_df (Data): 当前折的依赖数据
            fold_num (int): 当前折号

        逻辑：
            - 首折或强制学习模式下执行完整adapt_process
            - 后续折可复用已有配置（当enforce_relearning=False时）
            - 记录训练时间和数据规模

        返回：
            当前配置的回归模型对象
        """
        before_training = default_timer()
        columns = node_feature_columns.copy()

        self.adapt_process(df, columns, dependency_df, fold_num)

        after_training = default_timer()
        total_training = after_training - before_training

        self.experiment_tracker.record_training_time(
            fold_num,
            total_training,
            df.index.get_level_values(0).unique().shape[0],  # bugs count
            df.index.get_level_values(1).unique().shape[0],  # files count
        )

        return self.reg_model

    def adapt_process(
        self,
        df: pd.DataFrame,
        columns: List[str],
        dependency_df: pd.DataFrame,
        fold_num=0,
    ):
        """
        自适应训练核心流程

        参数：
            df (pd.DataFrame): 完整训练数据
            columns (list): 特征列名列表
            dependency_df (pd.DataFrame): 依赖数据
            fold_num (int): 当前折号

        阶段：
            1. 权重计算阶段：调用compute_weights选择最佳权重方法
            2. 模型选择阶段：遍历所有可能的（评分×模型×筛选）组合
               - 使用交叉验证评估每个组合的性能
               - 选择最佳组合配置

        更新：
            self.reg_model: 最佳回归模型实例
        """
        eprint("=============== Weights Select")
        results = self.compute_weights(df, columns)

        # 选择最佳权重方法
        w_maks: float = 0
        w_method: str = None
        w_weights: np.ndarray = None
        w_map_score: float = 0
        w_mrr_score: float = 0

        for k, (weights, scores) in results.items():
            current_score = scores[self.metric_type]
            # 记录最佳方法
            if current_score > w_maks:
                w_maks = current_score
                w_method = k
                w_weights = weights
                w_map_score = scores["MAP"]
                w_mrr_score = scores["MRR"]

        self.weights = w_weights
        self.weights_score = w_maks
        eprint(f"Best weights method: {w_method} {self.metric_type}: {w_maks}")
        eprint(f"  MAP: {w_map_score:.4f}, MRR: {w_mrr_score:.4f}")

        weights_model_id = f"weights_method_fold{fold_num}"
        if weights_model_id not in self.experiment_tracker.model_params:
            # 注册权重方法
            self.experiment_tracker.model_params[weights_model_id] = {
                "method": w_method,  # 使用刚刚确定的最佳方法名称
                "fold_num": fold_num,
                "model_type": "weights",
            }
            self.experiment_tracker.model_results[weights_model_id] = {}

        # 更新权重方法的训练得分
        self.experiment_tracker.update_result(
            weights_model_id, f"train_MAP_score", w_map_score
        )
        self.experiment_tracker.update_result(
            weights_model_id, f"train_MRR_score", w_mrr_score
        )

        eprint("===============")

        eprint("=============== Regression model select")

        # 为当前折创建所有模型的副本并注册
        fold_reg_models = []
        for base_model in self.reg_models:
            # 克隆模型，以避免模型实例在多个折之间共享
            fold_model = clone_regressor(base_model)
            self.experiment_tracker.register_model(fold_model, fold_num)
            fold_reg_models.append(fold_model)

        results: List[tuple[GATRegressor, float]] = Parallel(n_jobs=8)(
            delayed(self._train)(
                df,
                columns,
                dependency_df,
                w_weights,
                reg_model,
            )
            for reg_model in fold_reg_models
        )

        self.reg_model_score = 0

        for reg_model, val_scores in results:
            model_id = reg_model.model_id

            self.experiment_tracker.update_training_info(
                model_id,
                best_epoch=getattr(reg_model, "best_epoch", None),
                best_val_score=getattr(reg_model, "best_validation_score", None),
            )

            # 从val_scores中提取MAP和MRR得分
            map_score = val_scores["MAP"]
            mrr_score = val_scores["MRR"]

            # 更新模型的训练得分 - 同时记录MAP和MRR
            self.experiment_tracker.update_result(
                model_id, f"train_MAP_score", map_score
            )
            self.experiment_tracker.update_result(
                model_id, f"train_MRR_score", mrr_score
            )

            # 存储模型实例和评估结果
            self.all_reg_models[model_id] = reg_model

            # 记录最佳模型
            current_score = val_scores[self.metric_type]
            if current_score > self.reg_model_score and reg_model.heads is not None:
                self.reg_model = reg_model
                self.reg_model_id = model_id
                self.reg_model_score = current_score

        eprint(
            f"Best Regression Model: {self.reg_model_id} {self.metric_type}: {self.reg_model_score}"
        )
        eprint("===============")

    def compute_weights(self, df: pd.DataFrame, columns: List[str]):
        """
        特征权重计算阶段（并行交叉验证）

        参数：
            df: 训练数据集，包含特征和used_in_fix标签
            columns: 特征列名列表
        """
        if self.use_prescoring_cross_validation:
            # 使用 k 折交叉验证计算权重
            kfold = KFold(
                n_splits=self.cross_validation_fold_number,
                random_state=None,
            )
            # 保存每种方法在每折的结果，包括权重向量和 MAP 得分
            partial_result_dict = defaultdict(list)
            for train_index, test_index in kfold.split(df):
                kdf = df.iloc[train_index]
                weights = Parallel(n_jobs=-1)(
                    delayed(weights_on_df)(m, kdf, columns)
                    for m in self.weights_methods
                )
                kdf_test = df.iloc[test_index]
                weights_results = Parallel(n_jobs=-1)(
                    delayed(eval_weights)(m, w, kdf_test, columns, None)
                    for m, w in weights
                )
                weights_results_dict = dict(weights_results)
                for m_name in weights_results_dict:
                    partial_result_dict[m_name].append(weights_results_dict[m_name])
            results = {}
            for m_name in partial_result_dict:
                values = partial_result_dict[m_name]
                weights_list: List[np.ndarray] = []
                eval_map_list = []
                eval_mrr_list = []

                for value in values:
                    weights_list.append(value[0])
                    scores = value[1]
                    eval_map_list.append(scores["MAP"])
                    eval_mrr_list.append(scores["MRR"])

                weights_avg: np.ndarray = np.mean(weights_list, axis=0)

                # 创建平均分数字典
                avg_scores = {
                    "MAP": np.mean(eval_map_list),
                    "MRR": np.mean(eval_mrr_list),
                }
                results[m_name] = (weights_avg, avg_scores)
        else:
            weights_results = Parallel(n_jobs=-1)(
                delayed(fold_check)(m, df, columns, None) for m in self.weights_methods
            )

            results = dict(weights_results)

        return results

    def _train(
        self,
        df: pd.DataFrame,
        columns: List[str],
        dependency_df: pd.DataFrame,
        weights: np.ndarray,
        reg_model: GATRegressor,
    ):
        """
        单配置评估内部方法，支持交叉验证

        参数：
            df (pd.DataFrame): 训练数据
            columns (list): 特征列
            dependency_df (pd.DataFrame): 依赖数据
            weights (np.ndarray): 当前权重向量
            reg_model: 回归模型实例

        流程：
            1. 计算特征得分并修正（增加修复样本权重）
            2. 应用特征筛选获取训练子集
            3. 使用交叉验证训练回归模型并评估

        返回：
            tuple: (模型, {"MAP": map_score, "MRR": mrr_score})
        """
        score = self.score_method(df, columns, weights)
        fix_score = score + df["used_in_fix"] * np.max(score)

        reg_model.fit(df, dependency_df, fix_score)

        # 根据配置决定是否使用交叉验证
        if self.use_training_cross_validation:
            # 按bug ID进行划分，确保同一bug的文件在同一折
            bug_ids = df.index.get_level_values(0).unique()
            kf = KFold(
                n_splits=self.cross_validation_fold_number,
                shuffle=True,
                random_state=42,
            )

            cv_map_scores = []
            cv_mrr_scores = []

            # 对每一折执行训练和验证
            for train_idx, val_idx in kf.split(bug_ids):
                # 获取训练和验证数据的bug IDs
                train_bugs = bug_ids[train_idx]
                val_bugs = bug_ids[val_idx]

                # 基于bug ID筛选数据
                train_mask = df.index.get_level_values(0).isin(train_bugs)
                val_mask = df.index.get_level_values(0).isin(val_bugs)

                # 准备训练和验证数据
                train_df = df[train_mask]
                val_df = df[val_mask]
                train_dep_df = dependency_df[
                    dependency_df.index.get_level_values(0).isin(train_bugs)
                ]
                val_dep_df = dependency_df[
                    dependency_df.index.get_level_values(0).isin(val_bugs)
                ]

                # 准备训练得分
                train_fix_score = fix_score[train_mask]

                # 训练模型
                fold_model = clone_regressor(reg_model)
                fold_model.fit(train_df, train_dep_df, train_fix_score)

                # 在验证集上评估
                val_pred = fold_model.predict(val_df, val_dep_df)

                scores = evaluate_fold(val_df, val_pred, None)

                cv_map_scores.append(scores["MAP"])
                cv_mrr_scores.append(scores["MRR"])

            # 创建包含所有评分的字典
            val_scores = {"MAP": np.mean(cv_map_scores), "MRR": np.mean(cv_mrr_scores)}
        else:
            predict_score = reg_model.predict(df, dependency_df)
            val_scores = evaluate_fold(df, predict_score, None)

        return reg_model, val_scores

    def predict(
        self,
        clf: GATRegressor,
        df: pd.DataFrame,
        df_dependency: pd.DataFrame,
        fold_num,
    ):
        """
        预测函数

        参数：
            clf: 训练好的回归模型
            df (pd.DataFrame): 测试数据
            df_dependency (pd.DataFrame）: 依赖数据
            fold_num: 当前折号

        决策逻辑：
            - 当回归模型得分高于权重方法得分时，使用回归预测
            - 否则使用特征权重线性组合得分

        返回：
            pd.DataFrame: 包含预测结果（result列）及原始字段的副本
        """

        X = df[node_feature_columns].values

        # Check if weights method gives better results on training
        if not self.use_prescoring_always and (
            self.reg_model_score >= self.weights_score or self.use_reg_model_always
        ):
            result = clf.predict(df, df_dependency)
        else:
            eprint(
                "GAT Regression model score is lower than weights method score, using weights method for prediction."
            )
            result = np.dot(X, self.weights)

        weights_model_id, (_, w_scores) = eval_weights(
            f"weights_method_fold{fold_num}",
            self.weights,
            df,
            node_feature_columns,
            None,
        )

        self.experiment_tracker.update_result(
            weights_model_id, f"predict_MAP_score", w_scores["MAP"]
        )
        self.experiment_tracker.update_result(
            weights_model_id, f"predict_MRR_score", w_scores["MRR"]
        )

        for model_id, model in self.all_reg_models.items():
            model_result = model.predict(df, df_dependency)

            m_scores = evaluate_fold(df, model_result, None)

            # 更新模型注册表中的测试结果
            self.experiment_tracker.update_result(
                model_id, f"predict_MAP_score", m_scores["MAP"]
            )
            self.experiment_tracker.update_result(
                model_id, f"predict_MRR_score", m_scores["MRR"]
            )

        r = df[["used_in_fix"]].copy(deep=False)
        r["result"] = result

        return r

    # ---------------------- 辅助函数 ----------------------


def get_skmodels(metric_type="MRR"):
    """
    根据指定的模型类型创建模型列表

    参数：
        metric_type: 评估指标类型，默认为"MRR"

    返回：
        GATRegressor模型列表
    """
    # 数据集大小不同，最优的超参数可能不同
    alpha_values = [1e-3, 1e-4]
    loss = ["WeightedMSE"]
    lr_list = [1e-3]
    penalty = ["l2", None]
    dropout_rates = [0.2]
    use_self_loops_modes = [False, True]
    early_stop_configs = [
        (False, 5),
    ]
    head_dim_configs = [
        (None, 32),  # MLP模式 - 中等维度
        (None, 64),  # MLP模式 - 大维度
        (1, 32),  # GAT单头 - 中等维度
        (1, 64),  # GAT单头 - 大维度
        (2, 32),  # GAT双头 - 中等维度
        (4, 16),  # GAT四头 - 小维度（控制总体参数量）
    ]

    models = [
        GATRegressor(
            node_feature_columns.copy(),
            edge_feature_columns.copy(),
            hidden_dim=hd,
            heads=h,
            use_self_loops_only=loop,
            loss=ls,
            lr=lr,
            penalty=p,
            dropout=dr,
            alpha=a,
            early_stop=es,
            n_iter_no_change=nic,
            metric_type=metric_type,
        )
        for (h, hd), ls, lr, p, dr, a, loop, (es, nic) in product(
            head_dim_configs,
            loss,
            lr_list,
            penalty,
            dropout_rates,
            alpha_values,
            use_self_loops_modes,
            early_stop_configs,
        )
        if not (h is None and loop is True)  # MLP模式下不使用自环
        and not (p is None and a != alpha_values[0])  # 无正则化时仅使用一个alpha值
    ]

    # 输出模型数量信息
    eprint(f"创建了 {len(models)} 个模型")

    return models


def process(
    ptemplate: Adaptive_Process,
    fold_number: int,
    fold_testing: Dict[int, pd.DataFrame],
    fold_training: Dict[int, pd.DataFrame],
    fold_dependency_testing: Dict[int, pd.DataFrame],
    fold_dependency_training: Dict[int, pd.DataFrame],
):
    """
    主处理函数

    参数：
        ptemplate (Adaptive_Process): 自适应算法实例
        fold_number: 折数
        fold_testing: 测试数据
        fold_training: 训练数据
        fold_dependency_testing: 测试依赖数据
        fold_dependency_training: 训练依赖数据
    """
    results_list = []

    for i in range(fold_number):
        r = _process(
            ptemplate,
            fold_training[i],
            fold_dependency_training[i],
            fold_testing[i + 1],
            fold_dependency_testing[i + 1],
            i,
        )
        if r is None:
            del ptemplate
            gc.collect()
            return None

        results_list.append(r)

    all_results_df = pd.concat(results_list)
    all_results_df.reset_index(level=1, drop=True, inplace=True)

    return {
        "name": ptemplate.name,
        "results": calculate_metric_results(all_results_df),
    }


def _process(
    ptemplate: Adaptive_Process,
    fold_training: pd.DataFrame,
    fold_dependency_training: pd.DataFrame,
    fold_testing: pd.DataFrame,
    fold_dependency_testing: pd.DataFrame,
    fold_num: int = 0,
):
    clf = ptemplate.train(fold_training, fold_dependency_training, fold_num)
    result = ptemplate.predict(clf, fold_testing, fold_dependency_testing, fold_num)
    return result


# 辅助函数：克隆回归器
def clone_regressor(reg_model):
    """创建回归器的深拷贝，保留所有参数但重置状态"""
    return GATRegressor(
        node_feature_columns=reg_model.node_feature_columns,
        dependency_feature_columns=reg_model.dependency_feature_columns,
        hidden_dim=reg_model.hidden_dim,
        heads=reg_model.heads,
        dropout=reg_model.dropout,
        alpha=reg_model.alpha,
        loss=reg_model.loss,
        penalty=reg_model.penalty,
        max_iter=reg_model.max_iter,
        tol=reg_model.tol,
        shuffle=reg_model.shuffle,
        epsilon=reg_model.epsilon,
        random_state=reg_model.random_state,
        lr=reg_model.lr,
        warm_start=reg_model.warm_start,
        n_iter_no_change=reg_model.n_iter_no_change,
        use_self_loops_only=reg_model.use_self_loops_only,
        early_stop=reg_model.early_stop,
        validation_fraction=reg_model.validation_fraction,
        metric_type=reg_model.metric_type,
    )


def normal_score(
    df: pd.DataFrame, columns: List[str], weights: np.ndarray
) -> np.ndarray:
    """
    计算特征线性组合得分（基本评分函数）

    参数：
        df: 包含特征的数据框
        columns: 要使用的特征列名列表
        weights: 特征权重向量，维度应与columns长度一致

    返回：
        np.ndarray: 每个样本的得分向量，通过特征与权重的点积计算

    说明：
        这是最基本的评分方法，通过特征向量与权重向量的点积计算样本得分。
        计算公式: score = X * weights，其中X为样本的特征矩阵
    """
    # 计算特征矩阵与权重向量的点积
    score = np.dot(df[columns], weights)
    return score


def main():
    parser = argparse.ArgumentParser(description="Train Adaptive Process")
    parser.add_argument("file_prefix", type=str, help="Feature files prefix")
    parser.add_argument("--max", action="store_true", help="Include feature 37")
    parser.add_argument("--mean", action="store_true", help="Include feature 38")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["MAP", "MRR"],
        default="MRR",
        help="评估指标类型 (MAP 或 MRR，默认: MRR)",
    )
    args = parser.parse_args()

    # 根据参数决定是否添加特征37和38
    if args.max:
        node_feature_columns.append("f37")
    if args.mean:
        node_feature_columns.append("f38")

    file_prefix = args.file_prefix

    (
        fold_number,
        fold_testing,
        fold_training,
        fold_dependency_testing,
        fold_dependency_training,
    ) = load(f"../joblib_memmap_{file_prefix}_graph/data_memmap", mmap_mode="r")

    ptemplate = Adaptive_Process(metric_type=args.metric)

    result = process(
        ptemplate,
        fold_number,
        fold_testing,
        fold_training,
        fold_dependency_testing,
        fold_dependency_training,
    )

    results_timestamp = time.strftime("%Y%m%d%H%M%S")
    result_dir = f"{file_prefix}_{results_timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # 保存模型注册表数据
    ptemplate.experiment_tracker.to_json(
        os.path.join(result_dir, f"{ptemplate.name}_models_registry.json")
    )

    # 保存模型训练结果指标
    with open(
        os.path.join(result_dir, f"{ptemplate.name}_metrics_results.json"), "w"
    ) as metrics_file:
        # 将结果转换为可JSON序列化的格式
        metrics_data = {
            "accuracy_at_k": {str(k): v for k, v in result["results"][0].items()},
            "mean_average_precision": float(result["results"][1]),
            "mean_reciprocal_rank": float(result["results"][2]),
        }
        json.dump(metrics_data, metrics_file, indent=4)

    print("======Results======")
    print_metrics(*result["results"])


if __name__ == "__main__":
    main()
