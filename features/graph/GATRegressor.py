import numpy as np
import pandas as pd
from typing import Annotated, List
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.optim import AdamW
import torch.nn.functional as F
from torch_geometric.data import Data

from train_utils import eprint
from ranking_losses import WeightedRankMSELoss
from metrics import calculate_metric_results


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
                val_score = self._evaluate_validation(val_data)
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
                    bug_id = data.bug_id  # 获取当前bug的ID

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
