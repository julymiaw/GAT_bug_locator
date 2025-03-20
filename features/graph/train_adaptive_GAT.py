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
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from skopt import load

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data

from metrics import calculate_metric_results, print_metrics
from train_utils import eprint
from ranking_losses import WeightedRankMSELoss

from weight_functions import get_weights_methods

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
        l1_ratio=0.15,
        max_iter=500,
        tol=1e-4,
        shuffle=True,
        epsilon=0.1,
        random_state=42,
        lr=0.005,
        warm_start=False,
        n_iter_no_change=5,
        use_self_loops_only=False,
    ):
        """
        初始化回归器

        参数:
            node_feature_columns: 用于训练的节点特征列名列表
            dependency_feature_columns: 用于训练的边特征列名列表
            hidden_dim: GAT隐藏层维度
            heads: 注意力头数，若为None则不使用GAT层
            dropout: Dropout率
            alpha: 正则化系数，类似于SGDRegressor中的alpha
            loss: 损失函数类型，可选'MSE'或'Huber'
            penalty: 正则化类型，可选'l2'、'l1'、'elasticnet'或None
            l1_ratio: elasticnet混合参数，仅当penalty='elasticnet'时使用
            max_iter: 最大迭代次数（训练轮数）
            tol: 收敛容差，用于提前停止迭代
            shuffle: 是否在每轮训练后打乱数据
            epsilon: Huber损失中的epsilon参数
            random_state: 随机种子
            lr: Adam优化器的学习率
            warm_start: 是否使用之前的解作为初始化
            n_iter_no_change: 用于提前停止的无改进迭代次数
        """
        self.node_feature_columns = node_feature_columns
        self.dependency_feature_columns = dependency_feature_columns
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

        # SGDRegressor风格的参数
        self.alpha = alpha
        self.loss = loss
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.epsilon = epsilon
        self.random_state = random_state
        self.lr = lr
        self.warm_start = warm_start
        self.n_iter_no_change = n_iter_no_change

        # 初始化模型和累积器
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 设置随机种子
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # 仅使用自环
        self.use_self_loops_only = use_self_loops_only

    def __str__(self):
        """返回模型的字符串表示，用于映射和调试"""
        # 正则化类型
        penalty_str = self.penalty if self.penalty is not None else "none"

        # 注意力头数
        if self.heads is None:
            heads_str = "nohead"  # 特殊标识符表示无GAT层的MLP模型
        else:
            heads_str = str(self.heads)

        base_str = f"GATRegressor_{self.loss}_{self.hidden_dim}_{penalty_str}_{heads_str}_a{self.alpha}_dr{self.dropout}_lr{self.lr}"

        if self.use_self_loops_only:
            base_str += "_selfloop"

        if self.shuffle:
            base_str += "_shuf"

        return base_str

    def __repr__(self):
        """返回模型的详细字符串表示，用于调试"""
        return self.__str__()

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

    def _get_reg_loss(self, model):
        """计算正则化损失"""
        reg_loss = 0
        if self.penalty is None:
            return 0

        for name, param in model.named_parameters():
            if "weight" in name:  # 只对权重进行正则化，不包括偏置
                if self.penalty == "l2":
                    reg_loss += torch.sum(param**2)
                elif self.penalty == "l1":
                    reg_loss += torch.sum(torch.abs(param))
                elif self.penalty == "elasticnet":
                    l1 = torch.sum(torch.abs(param))
                    l2 = torch.sum(param**2)
                    reg_loss += self.l1_ratio * l1 + (1 - self.l1_ratio) * l2

        return self.alpha * reg_loss

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

        # 直接添加分数
        node_features["score"] = score

        # 准备数据
        data_list = self._prepare_data(node_features, edge_features)

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
            self.optimizer = Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.alpha
            )
        else:
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        # 训练循环
        self.model.train()
        best_loss = float("inf")
        no_improvement_count = 0

        # 全部数据一次性加载到内存
        all_data = [data.to(self.device) for data in data_list]

        # 确保all_data不为空
        if len(all_data) == 0:
            eprint("错误: 没有有效的训练数据")
            return self

        # 提前停止条件的监控变量
        best_weights = None

        for epoch in range(self.max_iter):
            # 如果需要，打乱数据
            if self.shuffle:
                np.random.shuffle(all_data)

            # 训练一轮
            total_loss = 0
            self.optimizer.zero_grad()

            for data in all_data:
                # 前向传播
                out = self.model(data.x, data.edge_index, data.edge_attr)

                # 计算损失
                loss = self.criterion(out, data.y)

                # 仅当非L2正则化或无正则化时添加手动正则化
                if self.penalty != "l2" and self.penalty is not None:
                    loss += self._get_reg_loss(self.model)

                # 反向传播
                loss.backward()
                total_loss += loss.item()

            # 更新参数
            self.optimizer.step()

            # 计算平均损失
            avg_loss = total_loss / len(all_data)

            # 检查收敛性
            if best_loss - avg_loss > self.tol:
                best_loss = avg_loss
                no_improvement_count = 0
                # 保存最佳模型
                best_weights = {
                    name: param.clone().detach()
                    for name, param in self.model.state_dict().items()
                }
            else:
                no_improvement_count += 1

            # 提前停止
            if (
                self.n_iter_no_change is not None
                and no_improvement_count >= self.n_iter_no_change
            ):
                # eprint(
                #     f"{str(self)}: Converged after {epoch+1} epochs. Loss: {best_loss}"
                # )
                break

        # 恢复最佳权重
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return self

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

    def __init__(self, model_type="auto"):
        """
        初始化方法，配置所有可用算法组件

        参数：
            model_type: 选择使用的模型类型
                "auto": 自动选择性能最佳的模型（默认）
                "mlp": 只使用MLP模型（heads=None）
                "gat": 只使用GAT模型（heads=数字，use_self_loops_only=False）
                "gat_degenerative": 只使用GAT退化模型（heads=数字，use_self_loops_only=True）
        """
        # region 算法组件配置
        # 特征权重计算方法列表（统计检验/树模型/降维方法等）
        self.weights_methods = get_weights_methods()

        # 指定模型类型
        self.model_type = model_type
        if model_type not in ["auto", "mlp", "gat", "gat_degenerative"]:
            eprint(f"警告: 未知的模型类型 '{model_type}'，使用 'auto' 模式")
            self.model_type = "auto"

        # 回归模型集合
        self.reg_models: List[GATRegressor] = []
        self.reg_models.extend(get_skmodels(self.model_type))

        # 评分方法（标准评分）
        self.score_method: Callable[
            [pd.DataFrame, List[str], np.ndarray], np.ndarray
        ] = normal_score
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

        # region 性能日志
        self.training_time_stats = (
            {}
        )  # {折号: {"time": 时间, "bugs": 缺陷数, "files": 文件数}}
        self.prescoring_log = {}  # {fold_num: {方法名: 评估得分}}
        self.best_prescoring_log = {}  # {fold_num: 最佳权重方法名}
        self.regression_log = {}  # 回归模型评估：{折号: {模型名: 评估得分}}
        self.cv_regression_log = {}  # {fold_num: {模型名: [各折CV得分]}}
        self.prediction_log = {}  # 预测结果日志：{折号: {模型名: 评估得分}}
        self.best_regression_log = {}  # {fold_num: 最佳模型名}
        # endregion

        # region 配置映射表（内部使用）
        self.weights_methods_map = {m.__name__: m for m in self.weights_methods}
        # endregion

    def compute_weights(self, df: pd.DataFrame, columns: List[str]):
        """
        特征权重计算阶段（并行交叉验证）

        参数：
            df: 训练数据集，包含特征和used_in_fix标签
            columns: 特征列名列表

        流程：
            1. 使用KFold拆分数据为预设折数
            2. 对每折训练集并行计算所有权重方法的权重向量
            3. 在验证集评估各权重方法的预测效果（MAP指标）
            4. 聚合各方法在所有折的平均表现

        结果存储：
            self.weights更新为最佳方法的平均权重向量
            self.prescoring_log记录所有方法评估结果
            self.best_prescoring_log记录最佳方法信息
        """
        if self.use_prescoring_cross_validation:
            # 使用 k 折交叉验证计算权重
            kfold = KFold(
                n_splits=self.cross_validation_fold_number,
                random_state=None,
                shuffle=False,
            )
            # 保存每种方法在每折的结果，包括权重向量和 MAP 得分
            partial_result_dict: Dict[str, List[Tuple[np.ndarray, float]]] = (
                defaultdict(list)
            )
            for train_index, test_index in kfold.split(df):
                kdf = df.iloc[train_index]
                weights: List[Tuple[str, np.ndarray]] = Parallel(n_jobs=-1)(
                    delayed(weights_on_df)(m, kdf, columns)
                    for m in self.weights_methods
                )
                kdf_test = df.iloc[test_index]
                weights_results: List[Tuple[str, Tuple[np.ndarray, float]]] = Parallel(
                    n_jobs=-1
                )(delayed(eval_weights)(m, w, kdf_test, columns) for m, w in weights)
                weights_results_dict: Dict[str, Tuple[np.ndarray, float]] = dict(
                    weights_results
                )
                for m_name in weights_results_dict:
                    partial_result_dict[m_name].append(weights_results_dict[m_name])
            results: Dict[str, Tuple[np.ndarray, float]] = {}
            for m_name in partial_result_dict:
                values: List[Tuple[np.ndarray, float]] = partial_result_dict[m_name]
                weights_list: List[np.ndarray] = []
                eval_list: List[float] = []
                for value in values:
                    weights_list.append(value[0])
                    eval_list.append(value[1])
                weights_avg: np.ndarray = np.mean(weights_list, axis=0)
                eval_avg: float = np.mean(eval_list)
                results[m_name] = (weights_avg, eval_avg)
            self.weights = results
        else:
            results: List[Tuple[str, Tuple[np.ndarray, float]]] = Parallel(n_jobs=-1)(
                delayed(fold_check)(m, df, columns) for m in self.weights_methods
            )
            self.weights = dict(results)

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

        阶段：
            1. 权重计算阶段：调用compute_weights选择最佳权重方法
            2. 模型选择阶段：遍历所有可能的（评分×模型×筛选）组合
               - 使用交叉验证评估每个组合的性能
               - 选择最佳组合配置

        更新：
            self.reg_model: 最佳回归模型实例
        """
        eprint("=============== Weights Select")
        self.compute_weights(df, columns)

        # 选择最佳权重方法
        w_maks: float = 0
        w_method: str = None
        w_weights: np.ndarray = None

        if fold_num not in self.prescoring_log:
            self.prescoring_log[fold_num] = {}

        for k, v in self.weights.items():
            # 存储每种方法的评估结果
            self.prescoring_log[fold_num][k] = v[1]
            # 记录最佳方法
            if v[1] > w_maks:
                w_maks = v[1]
                w_method = k
                w_weights = v[0]

        self.weights = w_weights
        self.weights_score = w_maks
        eprint(f"Best weights method: {w_method} MAP: {w_maks}")

        self.best_prescoring_log[fold_num] = w_method
        eprint("===============")

        eprint("=============== Regression model select")

        results: List[tuple[GATRegressor, str, float]] = Parallel(n_jobs=8)(
            delayed(self._train)(
                df,
                columns,
                dependency_df,
                w_weights,
                reg_model,
            )
            for reg_model in self.reg_models
        )

        res_max = 0
        self.regression_log[fold_num] = {}
        self.cv_regression_log[fold_num] = {}

        for res in results:
            current_reg_model, current_score, cv_results = res
            current_model_name = str(current_reg_model)

            self.all_reg_models[current_model_name] = current_reg_model
            self.regression_log[fold_num][current_model_name] = current_score
            if current_score > res_max:
                res_max = current_score
                self.reg_model = current_reg_model
                self.reg_model_name = current_model_name
            # 如果有CV结果，合并到主日志中
            if cv_results:
                for model_name, scores in cv_results.items():
                    self.cv_regression_log[fold_num][model_name] = scores

        self.reg_model_score = res_max
        current_reg_model = self.reg_model
        self.best_regression_log[fold_num] = self.reg_model_name

        eprint(
            f"Best Regression Model: {self.reg_model_name} MAP: {self.reg_model_score}"
        )
        eprint("===============")

    def train(self, df: pd.DataFrame, dependency_df: pd.DataFrame, fold_num):
        """
        训练入口函数

        参数：
            df (pd.DataFrame): 当前折的训练数据
            dependency_df (Data): 当前折的依赖数据

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
        self.training_time_stats[fold_num] = {
            "time": total_training,
            "bugs": df.index.get_level_values(0).unique().shape[0],
            "files": df.index.get_level_values(1).unique().shape[0],
        }

        return self.reg_model

    def predict(
        self,
        clf: GATRegressor,
        df: pd.DataFrame,
        df_dependency: pd.DataFrame,
        fold_num=0,
    ):
        """
        预测函数

        参数：
            clf: 训练好的回归模型（实际可能未使用）
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

        self.prediction_log[fold_num] = {}

        for model_name, model in self.all_reg_models.items():
            try:
                model_result = model.predict(df, df_dependency)

                # 计算当前模型在测试集上的MAP分数
                r_temp = df[["used_in_fix"]].copy(deep=False)
                r_temp["result"] = model_result
                _, map_score, _, _ = calculate_metric_results(r_temp)

                # 记录到日志
                self.prediction_log[fold_num][model_name] = map_score
            except Exception as e:
                eprint(f"Error predicting with model {model_name}: {str(e)}")
                self.prediction_log[fold_num][model_name] = None

        r = df[["used_in_fix"]].copy(deep=False)
        r["result"] = result

        return r

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
            tuple: (模型, 筛选方法名, 评分方法名, 平均MAP得分)
        """
        score = self.score_method(df, columns, weights)
        fix_score = score + df["used_in_fix"] * np.max(score)

        # 获取模型名称用于日志记录
        model_name = str(reg_model)

        cv_results = None

        # 根据配置决定是否使用交叉验证
        if self.use_training_cross_validation:
            # 按bug ID进行划分，确保同一bug的文件在同一折
            bug_ids = df.index.get_level_values(0).unique()
            kf = KFold(
                n_splits=self.cross_validation_fold_number,
                shuffle=True,
                random_state=42,
            )

            cv_scores = []

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
                val_result = val_df[["used_in_fix"]].copy(deep=False)
                val_result["result"] = val_pred
                _, val_map, _, _ = calculate_metric_results(val_result)

                cv_scores.append(val_map)

            # 计算平均交叉验证得分
            cv_mean_score = np.mean(cv_scores)
            eprint(
                f"{model_name} 的交叉验证得分: {cv_mean_score:.4f} (std: {np.std(cv_scores):.4f})"
            )

            cv_results = {model_name: [float(score) for score in cv_scores]}

            # 在全部数据上重新训练模型，用于后续预测
            reg_model.fit(df, dependency_df, fix_score)

            # 返回使用交叉验证平均得分
            return (reg_model, cv_mean_score, cv_results)
        else:
            # 原有逻辑：直接在整个训练集训练并评估
            reg_model.fit(df, dependency_df, fix_score)
            predict_score = reg_model.predict(df, dependency_df)
            return (reg_model, evaluate_fold(df, predict_score), None)

    # ---------------------- 辅助函数 ----------------------


def get_skmodels(model_type="auto"):
    """
    根据指定的模型类型创建模型列表

    参数：
        model_type: 模型类型
            "auto": 返回所有类型的模型（默认）
            "mlp": 只返回MLP模型（heads=None）
            "gat": 只返回GAT模型（heads=数字，use_self_loops_only=False）
            "gat_degenerative": 只返回GAT退化模型（heads=数字，use_self_loops_only=True）

    返回：
        GATRegressor模型列表
    """
    hidden_dim = [16]
    alpha_values = [0.0001]
    loss = ["MSE", "WeightedMSE"]
    lr_list = [0.005]
    penalty = ["l2"]
    dropout_rates = [0.3]

    # 根据模型类型配置heads和use_self_loops_modes参数
    if model_type == "mlp":
        # 只使用MLP模型
        heads = [None]
        use_self_loops_modes = [False]  # 这个参数对MLP没有影响
    elif model_type == "gat":
        # 只使用GAT模型
        heads = [2, 4]
        use_self_loops_modes = [False]
    elif model_type == "gat_degenerative":
        # 只使用GAT退化模型
        heads = [2, 4]
        use_self_loops_modes = [True]
    else:  # "auto"或其他值
        # 使用所有类型的模型
        heads = [None, 2, 4]
        use_self_loops_modes = [False, True]

    models = [
        GATRegressor(
            node_feature_columns.copy(),
            edge_feature_columns.copy(),
            hidden_dim=hd,
            heads=h,
            shuffle=False,
            use_self_loops_only=loop,
            loss=ls,
            lr=lr,
            penalty=p,
            dropout=dr,
            alpha=a,
        )
        for hd, h, ls, lr, p, dr, a, loop in product(
            hidden_dim,
            heads,
            loss,
            lr_list,
            penalty,
            dropout_rates,
            alpha_values,
            use_self_loops_modes,
        )
        if not (h is None and loop is True)
    ]

    # 输出模型数量信息
    eprint(f"创建了 {len(models)} 个模型，当前类型: {model_type}")

    return models


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


def process(
    ptemplate: Adaptive_Process,
    fold_number: int,
    fold_testing: Dict[int, pd.DataFrame],
    fold_training: Dict[int, pd.DataFrame],
    fold_dependency_testing: Dict[int, pd.DataFrame],
    fold_dependency_training: Dict[int, pd.DataFrame],
    file_prefix: str,
    mode="train",
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
        file_prefix: 文件前缀
        mode: 训练时，保存日志。预测时，不保存日志
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

    if mode == "predict":
        return {
            "name": ptemplate.name,
            "results": calculate_metric_results(all_results_df),
        }

    results_timestamp = time.strftime("%Y%m%d%H%M%S")
    result_dir = f"{file_prefix}_{ptemplate.model_type}_{results_timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # 保存训练时间统计
    with open(
        os.path.join(result_dir, f"{ptemplate.name}_training_time.json"), "w"
    ) as time_file:
        json.dump(ptemplate.training_time_stats, time_file, indent=4)

    # 保存权重方法评估日志
    with open(
        os.path.join(result_dir, f"{ptemplate.name}_prescoring_log.json"), "w"
    ) as file:
        json.dump(ptemplate.prescoring_log, file, indent=4)

    # 保存回归模型评估日志
    with open(
        os.path.join(result_dir, f"{ptemplate.name}_regression_log.json"), "w"
    ) as file:
        json.dump(ptemplate.regression_log, file, indent=4)

    # 保存预测结果日志
    with open(
        os.path.join(result_dir, f"{ptemplate.name}_prediction_log.json"), "w"
    ) as file:
        json.dump(ptemplate.prediction_log, file, indent=4)

    # 保存最佳权重方法日志
    with open(
        os.path.join(result_dir, f"{ptemplate.name}_best_prescoring_log.json"), "w"
    ) as file:
        json.dump(ptemplate.best_prescoring_log, file, indent=4)

    # 保存最佳模型日志
    with open(
        os.path.join(result_dir, f"{ptemplate.name}_best_regression_log.json"), "w"
    ) as file:
        json.dump(ptemplate.best_regression_log, file, indent=4)

    with open(
        os.path.join(result_dir, f"{ptemplate.name}_cv_regression_log.json"), "w"
    ) as file:
        json.dump(ptemplate.cv_regression_log, file, indent=4)

    return {
        "name": ptemplate.name,
        "results": calculate_metric_results(all_results_df),
    }


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
        l1_ratio=reg_model.l1_ratio,
        max_iter=reg_model.max_iter,
        tol=reg_model.tol,
        shuffle=reg_model.shuffle,
        epsilon=reg_model.epsilon,
        random_state=reg_model.random_state,
        lr=reg_model.lr,
        warm_start=reg_model.warm_start,
        n_iter_no_change=reg_model.n_iter_no_change,
        use_self_loops_only=reg_model.use_self_loops_only,
    )


def weights_on_df(
    method: Callable[[pd.DataFrame, List[str]], np.ndarray],
    df: pd.DataFrame,
    columns: List[str],
):
    """
    单权重方法计算包装函数

    参数：
        method: 权重计算方法
        df: 当前折训练数据
        columns: 特征列名列表

    返回：
        tuple: (方法名称, 权重向量)

    说明：
        - 用于并行计算任务包装
        - 调用具体权重计算方法并返回标准化结果
    """
    weights = method(df, columns)
    return method.__name__, weights


def eval_weights(
    m_name: str, weights: np.ndarray, df: pd.DataFrame, columns: List[str]
) -> Tuple[str, Tuple[np.ndarray, float]]:
    """
    权重评估函数（验证阶段）

    参数：
        m_name: 权重方法名称
        weights: 已计算的权重向量
        df: 验证集数据
        columns: 特征列名列表

    返回：
        tuple: (方法名称, (权重向量, MAP得分))

    流程：
        1. 计算验证集预测得分：X * weights
        2. 调用evaluate_fold计算MAP指标
    """
    Y = np.dot(df[columns], weights)
    return m_name, (weights, evaluate_fold(df, Y))


def fold_check(
    method: Callable[[pd.DataFrame, List[str]], np.ndarray],
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[str, Tuple[np.ndarray, float]]:
    """
    单折权重评估函数（非交叉验证模式）

    参数：
        method (function): 权重计算方法
        df (pd.DataFrame): 完整训练数据
        columns (list): 特征列名列表

    返回：
        tuple: (方法名称, (权重向量, MAP得分))

    说明：
        - 当use_prescoring_cross_validation=False时使用
        - 直接在整个数据集上计算和评估
    """
    weights = method(df, columns)
    Y = np.dot(df[columns], weights)
    return method.__name__, (weights, evaluate_fold(df, Y))


def evaluate_fold(df: pd.DataFrame, Y: np.ndarray) -> float:
    """
    评估预测结果的MAP指标

    参数：
        df: 待评估数据集（需包含used_in_fix列）
        Y: 预测得分向量

    返回：
        m_a_p: 平均精度均值（Mean Average Precision）

    流程：
        1. 构建结果数据框（含预测得分）
        2. 确定最小修复得分阈值（实际修复样本的最低得分）
        3. 生成候选集（预测得分≥阈值的样本）
        4. 调用calculate_metric_results计算指标
    """
    r = df[["used_in_fix"]].copy(deep=False)
    r["result"] = Y
    _, m_a_p, _, _ = calculate_metric_results(r)
    return m_a_p


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

    result = process(
        Adaptive_Process(),
        fold_number,
        fold_testing,
        fold_training,
        fold_dependency_testing,
        fold_dependency_training,
        file_prefix,
    )

    print("======Results======")
    print_metrics(*result["results"])


if __name__ == "__main__":
    main()
