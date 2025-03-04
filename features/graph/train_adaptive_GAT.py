#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <feature_files_prefix>

Requires results of save_normalized_fold_dataframes.py
"""

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
from scipy.stats import kruskal, ttest_ind, levene
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.model_selection import KFold
from sklearn.utils import safe_mask
from skopt import load

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data

from metrics import calculate_metric_results, print_metrics
from train_utils import eprint

node_feature_columns = ["f" + str(i) for i in range(1, 20)]
edge_feature_columns = ["t" + str(i) for i in range(1, 13)]


class GATModule(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, heads, dropout):
        """
        增强版GAT模型，具有更深的网络结构和多头注意力机制

        参数:
            node_dim: 节点特征维度
            edge_dim: 边特征维度
            hidden_dim: 隐藏层维度
            heads: 各层的注意力头数列表，若为空则不使用GAT层
            dropout: Dropout率
        """
        super(GATModule, self).__init__()

        # 节点特征变换
        self.node_lin = nn.Linear(node_dim, hidden_dim)
        self.use_gat = len(heads) > 0

        # 边特征变换为权重
        if self.use_gat:
            self.edge_lin = nn.Linear(edge_dim, hidden_dim // 4)
            self.edge_proj = nn.Linear(hidden_dim // 4, 1)

            # 多层GAT结构
            self.gat_layers = nn.ModuleList()

            # 第一层GAT
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads[0],
                    dropout=dropout,
                    add_self_loops=True,
                )
            )

            # 中间层GAT
            for i in range(1, len(heads)):
                self.gat_layers.append(
                    GATConv(
                        hidden_dim * heads[i - 1],
                        hidden_dim,
                        heads=heads[i],
                        dropout=dropout,
                        add_self_loops=True,
                    )
                )

            # 层间激活和正则化
            self.layer_norm = nn.LayerNorm(hidden_dim * heads[-1])
            self.final_dim = hidden_dim * heads[-1]
        else:
            self.final_dim = hidden_dim
            self.layer_norm = nn.LayerNorm(hidden_dim)

        # 共用组件
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

        # 预测层
        self.out_lin1 = nn.Linear(self.final_dim, hidden_dim)
        self.out_lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        # 节点特征变换
        x = self.node_lin(x)
        x = F.relu(x)

        if self.use_gat:
            # 边特征处理
            edge_emb = self.edge_lin(edge_attr)
            edge_emb = self.activation(edge_emb)
            edge_weights = torch.sigmoid(self.edge_proj(edge_emb)).squeeze(-1)

            # 多层GAT处理
            for i, gat_layer in enumerate(self.gat_layers):
                if i == 0:
                    x = gat_layer(x, edge_index, edge_attr=edge_weights)
                else:
                    # 添加残差连接
                    identity = x
                    x = gat_layer(x, edge_index, edge_attr=edge_weights)
                    if identity.shape[-1] == x.shape[-1]:  # 维度匹配时添加残差
                        x = x + identity

                if i < len(self.gat_layers) - 1:  # 非最后一层
                    x = self.activation(x)
                    x = self.dropout(x)

        # 最终层规范化和预测
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_lin2(x)

        return x


class GATRegressor:
    """
    用于bug定位的GAT回归器，提供与scikit-learn兼容的接口
    """

    def __init__(
        self,
        node_feature_columns,
        dependency_feature_columns,
        hidden_dim=32,
        heads=[4, 4, 2],
        dropout=0.1,
        alpha=0.0001,
        loss="MSE",
        penalty="l2",
        l1_ratio=0.15,
        max_iter=1000,
        tol=1e-4,
        shuffle=True,
        epsilon=0.1,
        random_state=None,
        lr=0.0001,
        warm_start=False,
        n_iter_no_change=5,
    ):
        """
        初始化回归器

        参数:
            node_feature_columns: 用于训练的节点特征列名列表
            dependency_feature_columns: 用于训练的边特征列名列表
            hidden_dim: GAT隐藏层维度
            heads: 注意力头数
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

    def __str__(self):
        """返回模型的字符串表示，用于映射和调试"""
        # 正则化类型
        penalty_str = self.penalty if self.penalty is not None else "none"

        # 注意力头数
        heads_str = (
            "_".join([str(h) for h in self.heads])
            if isinstance(self.heads, list)
            else str(self.heads)
        )

        base_str = f"GATRegressor_{self.loss}_{self.hidden_dim}_{penalty_str}_{heads_str}_lr{self.lr}"

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
        bug_ids = node_features.index.levels[0].unique()

        for bug_id in bug_ids:
            # 获取该bug的节点特征
            bug_nodes = node_features.loc[bug_id]

            # 获取文件ID列表及其映射
            file_ids = bug_nodes.index.tolist()
            file_to_idx = {file_id: idx for idx, file_id in enumerate(file_ids)}

            # 准备节点特征
            x = torch.tensor(
                bug_nodes[self.node_feature_columns].values, dtype=torch.float
            )

            # 准备目标值
            y = torch.tensor(bug_nodes["score"].values, dtype=torch.float).reshape(
                -1, 1
            )

            # 获取该bug的边信息
            bug_edges = edge_features[edge_features.index == bug_id]

            if len(bug_edges) > 0:
                # 使用pandas向量化操作映射source和target到索引
                source_indices = bug_edges["source"].map(file_to_idx).values
                target_indices = bug_edges["target"].map(file_to_idx).values

                # 准备边索引和边特征
                edge_index = torch.tensor(
                    np.vstack([source_indices, target_indices]), dtype=torch.long
                )
                edge_attr = torch.tensor(
                    bug_edges[self.dependency_feature_columns].values, dtype=torch.float
                )
            else:
                # 如果没有边，创建自环
                edge_index = torch.stack(
                    [torch.arange(len(file_ids)), torch.arange(len(file_ids))]
                )
                edge_attr = torch.zeros(
                    (len(file_ids), len(self.dependency_feature_columns))
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
        self.optimizer = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.alpha
        )

        # 训练循环
        self.model.train()
        best_loss = float("inf")
        no_improvement_count = 0

        # 全部数据一次性加载到内存
        all_data = [data.to(self.device) for data in data_list]

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

                # 添加正则化
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
                eprint(
                    f"{str(self)}: Converged after {epoch+1} epochs. Loss: {best_loss}"
                )
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
                pred = out.cpu().numpy().flatten()
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

    def __init__(self):
        """初始化方法，配置所有可用算法组件"""
        # region 算法组件配置
        # 特征权重计算方法列表（统计检验/树模型/降维方法等）
        self.weights_methods = [
            weights_AdaBoostClassifier,
            weights_ExtraTreesClassifier,
            weights_GradientBoostingClassifier,
            weights_const,
            weights_variance,
            weights_chi2,
            weights_mutual_info_classif,
            weights_FastICA,
            weights_kruskal_classif,
            weights_ttest_ind_classif,
            weights_levene_median,
            weights_mean_var,
            weights_maximum_absolute_deviation,
        ]

        # 回归模型集合
        self.reg_models: List[GATRegressor] = []
        self.reg_models.extend(get_skmodels())

        # 评分方法（目前仅标准评分）
        self.score_methods: List[
            Callable[[pd.DataFrame, List[str], np.ndarray], np.ndarray]
        ] = [normal_score]
        # endregion

        # region 运行时状态存储
        self.weights = None  # 当前最佳特征权重向量（np.ndarray）
        self.weights_score = 0  # 当前最佳权重方法得分
        self.reg_model = None  # 当前选择的回归模型对象
        self.reg_model_score = 0  # 当前最佳回归模型得分
        self.score_method = None  # 当前选择的评分方法（函数引用）
        # endregion

        # region 配置参数
        self.name = "Adaptive"  # 算法标识名
        self.use_prescoring_always = False  # 是否始终使用预评分权重
        self.use_reg_model_always = True  # 是否强制使用回归模型
        self.use_prescoring_cross_validation = True  # 权重计算阶段交叉验证开关
        self.cross_validation_fold_number = 2  # 交叉验证折数
        # endregion

        # region 性能日志
        self.training_time_list = []  # 各折训练耗时：(总时间, 缺陷报告数, 文件数)
        self.prescoring_log = []  # 权重方法评估：(方法名, (权重向量, 评估得分))
        self.best_prescoring_log = []  # 各折最佳权重： (方法名, 评估得分)
        self.regression_log = []  # 回归模型评估：(模型名, 筛选方法名, 评分方法名, 得分)
        self.best_regression_log = []  # 各折最佳模型，格式同regression_log
        # endregion

        # region 配置映射表（内部使用）
        self.weights_methods_map = {m.__name__: m for m in self.weights_methods}
        self.score_methods_map = {m.__name__: m for m in self.score_methods}
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
        self, df: pd.DataFrame, columns: List[str], dependency_df: pd.DataFrame
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
            self.score_method: 最佳评分方法
        """
        eprint("=============== Weights Select")
        self.compute_weights(df, columns)

        # 选择最佳权重方法
        w_maks: float = 0
        w_method: str = None
        w_weights: np.ndarray = None
        for k, v in self.weights.items():
            # 存储每种方法的评估结果
            self.prescoring_log.append((k, v[1]))
            # 记录最佳方法
            if v[1] > w_maks:
                w_maks = v[1]
                w_method = k
                w_weights = v[0]

        self.weights = w_weights
        self.weights_score = w_maks
        eprint(f"Best weights method: {w_method} MAP: {w_maks}")
        self.best_prescoring_log.append((w_method, w_maks))
        eprint("===============")

        eprint("=============== Size and regression model select")

        results: List[tuple[GATRegressor, str, float]] = Parallel(n_jobs=12)(
            delayed(self._train)(
                df,
                columns,
                dependency_df,
                w_weights,
                score_method,
                reg_model,
            )
            for score_method, reg_model in product(self.score_methods, self.reg_models)
        )

        res_max = 0
        for res in results:
            current_reg_model = res[0]
            current_score_function = res[1]
            current_score = res[2]

            self.regression_log.append(
                (str(current_reg_model), current_score_function, current_score)
            )
            if current_score > res_max:
                res_max = current_score
                self.reg_model = current_reg_model
                self.reg_model_name = str(current_reg_model)
                self.score_method_name = current_score_function

        self.score_method = self.score_methods_map[self.score_method_name]

        self.reg_model_score = res_max
        current_reg_model = self.reg_model
        self.best_regression_log.append(
            (self.reg_model_name, self.score_method_name, self.reg_model_score)
        )

        eprint(self.reg_model_name, self.score_method_name, self.reg_model_score)
        eprint("===============")

    def train(self, df: pd.DataFrame, dependency_df: pd.DataFrame):
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

        self.adapt_process(df, columns, dependency_df)

        after_training = default_timer()
        total_training = after_training - before_training
        self.training_time_list.append(
            (
                total_training,
                df.index.get_level_values(0).unique().shape[0],
                df.index.get_level_values(1).unique().shape[0],
            )
        )

        return self.reg_model

    def predict(self, clf: GATRegressor, df: pd.DataFrame, df_dependency: pd.DataFrame):
        """
        预测函数

        参数：
            clf: 训练好的回归模型（实际可能未使用）
            df (pd.DataFrame): 测试数据
            df_dependency (pd.DataFrame）: 依赖数据

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
            print("Using weights")
            result = np.dot(X, self.weights)

        r = df[["used_in_fix"]].copy(deep=False)
        r["result"] = result

        return r

    def _train(
        self,
        df: pd.DataFrame,
        columns: List[str],
        dependency_df: pd.DataFrame,
        weights: np.ndarray,
        score_method: Callable[[pd.DataFrame, List[str], np.ndarray], np.ndarray],
        reg_model: GATRegressor,
    ):
        """
        单配置评估内部方法

        参数：
            df (pd.DataFrame): 训练数据
            columns (list): 特征列
            dependency_df (pd.DataFrame): 依赖数据
            weights (np.ndarray): 当前权重向量
            score_method (function): 评分方法
            reg_model: 回归模型实例

        流程：
            1. 计算特征得分并修正（增加修复样本权重）
            2. 应用特征筛选获取训练子集
            3. 使用交叉验证训练回归模型并评估

        返回：
            tuple: (模型, 筛选方法名, 评分方法名, 平均MAP得分)
        """
        score = score_method(df, columns, weights)
        score = score + df["used_in_fix"] * np.max(score)
        reg_model.fit(df, dependency_df, score)
        score = reg_model.predict(df, dependency_df)

        return (reg_model, score_method.__name__, evaluate_fold(df, score))

    # ---------------------- 辅助函数 ----------------------


def get_skmodels():
    hidden_dim = [8, 16, 32]
    heads = [[], [2], [4, 2], [4, 4, 2]]
    loss = ["MSE", "Huber"]
    lr_values = [0.001, 0.01]
    penalty = [None, "l2", "l1", "elasticnet"]
    return [
        GATRegressor(
            node_feature_columns.copy(),
            edge_feature_columns.copy(),
            hidden_dim=hd,
            heads=h,
            shuffle=False,
            loss=ls,
            lr=lr,
            penalty=p,
        )
        for hd, h, ls, lr, p in product(hidden_dim, heads, loss, lr_values, penalty)
    ]


def _process(
    ptemplate: Adaptive_Process,
    fold_training: pd.DataFrame,
    fold_dependency_training: pd.DataFrame,
    fold_testing: pd.DataFrame,
    fold_dependency_testing: pd.DataFrame,
):
    clf = ptemplate.train(fold_training, fold_dependency_training)
    result = ptemplate.predict(clf, fold_testing, fold_dependency_testing)
    return result


def process(
    ptemplate: Adaptive_Process,
    fold_number: int,
    fold_testing: Dict[int, pd.DataFrame],
    fold_training: Dict[int, pd.DataFrame],
    fold_dependency_testing: Dict[int, pd.DataFrame],
    fold_dependency_training: Dict[int, pd.DataFrame],
    file_prefix: str,
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
    """
    results_list = []

    for i in range(fold_number):
        r = _process(
            ptemplate,
            fold_training[i],
            fold_dependency_training[i],
            fold_testing[i + 1],
            fold_dependency_testing[i + 1],
        )
        if r is None:
            del ptemplate
            gc.collect()
            return None

        results_list.append(r)

    all_results_df = pd.concat(results_list)
    all_results_df.reset_index(level=1, drop=True, inplace=True)

    training_time_list = ptemplate.training_time_list.copy()

    eprint(training_time_list)
    time_sum, bug_reports_number_sum, file_number_sum = map(
        sum, zip(*training_time_list)
    )

    eprint("time_sum", time_sum)
    eprint("bug_reports_number_sum", bug_reports_number_sum)
    eprint("file_number_sum", file_number_sum)

    mean_time_bug_report_training = time_sum / bug_reports_number_sum
    mean_time_file_training = time_sum / file_number_sum

    eprint("mean_time_bug_report_training", mean_time_bug_report_training)
    eprint("mean_time_file_training", mean_time_file_training)

    results_timestamp = time.strftime("%Y%m%d%H%M%S")

    training_time = {
        "time_sum": time_sum,
        "bug_reports_number_sum": bug_reports_number_sum,
        "file_number_sum": file_number_sum,
        "mean_time_bug_report_training": mean_time_bug_report_training,
        "mean_time_file_training": mean_time_file_training,
    }
    with open(
        f"{file_prefix}_{ptemplate.name}_training_time_{results_timestamp}.json", "w"
    ) as time_file:
        json.dump(training_time, time_file)

    prescoring_log = ptemplate.prescoring_log.copy()
    with open(
        f"{file_prefix}_{ptemplate.name}_prescoring_log_{results_timestamp}.json", "w"
    ) as prescoring_log_file:
        json.dump(prescoring_log, prescoring_log_file)

    regression_log = ptemplate.regression_log.copy()
    with open(
        f"{file_prefix}_{ptemplate.name}_regression_log_{results_timestamp}.json", "w"
    ) as regression_log_file:
        json.dump(regression_log, regression_log_file)

    best_prescoring_log = ptemplate.best_prescoring_log.copy()
    with open(
        f"{file_prefix}_{ptemplate.name}_best_prescoring_log_{results_timestamp}.json",
        "w",
    ) as best_prescoring_log_file:
        json.dump(best_prescoring_log, best_prescoring_log_file)

    best_regression_log = ptemplate.best_regression_log.copy()
    with open(
        f"{file_prefix}_{ptemplate.name}_best_regression_log_{results_timestamp}.json",
        "w",
    ) as best_regression_log_file:
        json.dump(best_regression_log, best_regression_log_file)

    return {
        "name": ptemplate.name,
        "results": calculate_metric_results(all_results_df),
    }


# region 特征权重计算
def _weights_normalize(weights: np.ndarray):
    """
    权重归一化函数

    参数：
        weights: 原始权重向量

    返回：
        weights: L1归一化后的权重向量（总和为1）

    说明：
        - 当权重总和>0时执行归一化
        - 处理全零权重时保持原值
    """
    weights_sum = weights.sum()
    if weights_sum > 0:
        weights /= weights_sum

    return weights


def weights_chi2(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    基于卡方检验的特征权重计算

    参数：
        df (pd.DataFrame): 包含特征和标签的数据
        columns (list): 特征列名列表

    返回：
        np.ndarray: 归一化的卡方统计量作为特征权重

    实现：
        使用sklearn.feature_selection.chi2计算各特征与目标变量的卡方统计量
    """
    weights = chi2(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def weights_mutual_info_classif(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """基于互信息的特征权重计算，适用于连续特征"""
    weights = mutual_info_classif(
        df[columns], df["used_in_fix"], discrete_features=False
    )
    weights = weights

    return _weights_normalize(weights)


def weights_FastICA(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """使用独立成分分析(ICA)的首个成分作为特征权重"""
    m = FastICA(n_components=1)
    m.fit(df[columns])
    weights = m.components_[0]

    return _weights_normalize(weights)


def weights_variance(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """基于特征方差的权重计算，负方差置零处理"""
    fs = VarianceThreshold()
    fs.fit(df[columns])
    weights = fs.variances_
    weights[weights < 0] = 0

    return _weights_normalize(weights)


def weights_const(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """恒定权重（0.5），用作基准方法"""
    return np.ones(df[columns].shape[1]) * 0.5


def weights_ExtraTreesClassifier(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """基于极端随机树分类器的特征重要性"""
    tree = ExtraTreesClassifier(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_GradientBoostingClassifier(
    df: pd.DataFrame, columns: List[str]
) -> np.ndarray:
    """梯度提升回归树特征重要性"""
    tree = GradientBoostingRegressor(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_AdaBoostClassifier(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """AdaBoost分类器特征重要性"""
    tree = AdaBoostClassifier(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_kruskal_classif(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    Kruskal-Wallis检验统计量作为权重
    适用于非正态分布的组间差异检验
    """
    weights = kruskal_classif(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def kruskal_classif(X, y):
    """
    执行Kruskal-Wallis H检验
    返回各特征的检验统计量绝对值和p值
    """
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = kruskal(*args)
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_ttest_ind_classif(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    独立样本t检验统计量作为权重
    假设方差不相等（Welch's t-test）
    """
    weights = ttest_ind_classif(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def ttest_ind_classif(X, y):
    """执行Welch's t-test，返回绝对t值和p值"""
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = ttest_ind(*args, equal_var=False)
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_levene_median(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    基于中位数Levene检验的权重计算
    用于检测组间方差差异
    """
    weights = levene_median(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def levene_median(X, y):
    """执行基于中位数的Levene方差齐性检验"""
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = levene(args[0], args[1], center="median")
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_mean_var(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    均值-方差比率权重：
    (修复样本的变异系数) / (非修复样本的变异系数)
    变异系数 = 标准差 / 均值
    """
    weights_var = np.var(df[df["used_in_fix"] == 1][columns], axis=0)
    weights_mean = np.mean(df[df["used_in_fix"] == 1][columns], axis=0)
    weights_var1 = np.var(df[df["used_in_fix"] == 0][columns], axis=0)
    weights_var1_mean = np.mean(df[df["used_in_fix"] == 0][columns], axis=0)

    return (weights_var / weights_mean) / (weights_var1 / weights_var1_mean)


def weights_maximum_absolute_deviation(
    df: pd.DataFrame, columns: List[str]
) -> np.ndarray:
    """
    最大绝对偏差权重：
    计算修复样本各特征值与其最大值的平均绝对偏差
    """
    weights_max = np.max(df[df["used_in_fix"] == 1][columns], axis=0)
    weights_mad = np.mean(
        np.abs(df[df["used_in_fix"] == 1][columns] - weights_max), axis=0
    )

    return weights_mad


# endregion


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

    models = [Adaptive_Process()]
    results = []
    for m in models:
        results.append(
            process(
                m,
                fold_number,
                fold_testing,
                fold_training,
                fold_dependency_testing,
                fold_dependency_training,
                file_prefix,
            )
        )

    results = [r for r in results if r is not None]
    eprint("Results")
    for result in results:
        print("name ", result["name"])
        print_metrics(*result["results"])


if __name__ == "__main__":
    main()
