import numpy as np
from typing import List
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.optim import AdamW
import torch.nn.functional as F
from torch_geometric.data import Data

from train_utils import eprint
from ranking_losses import WeightedRankMSELoss
from metrics import evaluate_fold


class ModelParameters:
    """GATRegressor 参数的中央定义类"""

    # 默认参数值
    DEFAULTS = {
        # 模型结构参数
        "node_feature_columns": None,  # 必须提供
        "dependency_feature_columns": None,  # 必须提供
        "hidden_dim": 16,
        "heads": None,
        "dropout": 0.3,
        "use_self_loops_only": False,
        # 优化器参数
        "alpha": 0.0001,
        "lr": 0.005,
        "penalty": "l2",
        # 损失函数参数
        "loss": "MSE",
        "epsilon": 0.1,
        # 训练控制参数
        "max_iter": 500,
        "tol": 1e-4,
        "n_iter_no_change": 5,
        "shuffle": True,
        "warm_start": False,
        "random_state": 42,
        "min_score_ratio": 0.8,
        # 评估参数
        "metric_type": "MRR",
    }

    @classmethod
    def get_all_params(cls):
        """获取所有参数名称"""
        return list(cls.DEFAULTS.keys())


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

        self.training_summary = {
            "stop_reason": None,
            "best_epoch": None,
            "final_epoch": None,
            "final_score": None,
            "weight_score": None,
            "performance_ratio": None,
            "final_loss": None,
        }

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
    用于bug定位的GAT回归器
    """

    def __init__(
        self,
        node_feature_columns,
        dependency_feature_columns,
        hidden_dim=16,
        heads=None,
        dropout=0.3,
        alpha=1e-4,
        loss="WeightedMSE",
        penalty="l2",
        max_iter=500,
        tol=1e-4,
        shuffle=True,
        epsilon=0.1,
        random_state=42,
        lr=1e-3,
        warm_start=False,
        n_iter_no_change=5,
        use_self_loops_only=False,
        metric_type="MRR",
        min_score_ratio=0.8,
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
            metric_type: 评估指标类型，可选值为"MAP"或"MRR"
            min_score_ratio: 训练时的最小分数比例
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
        self.metric_type = metric_type
        self.min_score_ratio = min_score_ratio

        # 模型注册时进行初始化
        self.model_id: str = None

        # 初始化模型和累积器
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        weight_score = evaluate_fold(node_features, score, self.metric_type)

        min_acceptable_score = weight_score * self.min_score_ratio

        # 添加修正后的分数用于训练
        fix_score = score + node_features["used_in_fix"] * np.max(score)
        node_features["score"] = fix_score

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
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.alpha
            )
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

        # 加载数据
        train_data = [data.to(self.device) for data in data_list]
        if not train_data:
            eprint("错误: 没有有效的训练数据")
            return self

        # 训练循环
        self.model.train()
        best_score = -float("inf")  # 用于评估分数的最佳值
        no_improvement_count = 0
        best_weights = None
        exceeded_weight_score = False
        current_score = 0.0

        # 初始化训练日志
        self.training_logs = []

        for epoch in range(self.max_iter):
            # 如果需要，打乱数据
            if self.shuffle:
                np.random.shuffle(train_data)

            # 训练一轮
            total_loss = 0
            self.optimizer.zero_grad()

            bug_predictions = {}

            for data in train_data:
                out = self.model(data.x, data.edge_index, data.edge_attr)
                loss = self.criterion(out, data.y)
                loss.backward()
                total_loss += loss.item()

                pred = out.detach().cpu().numpy().flatten()
                bug_predictions[data.bug_id] = pred

            self.optimizer.step()
            avg_train_loss = total_loss / len(train_data)

            predictions = np.zeros(len(score))
            idx = 0

            # 遍历node_features中的每个bug_id，保持原始顺序
            for bug_id in node_features.index.get_level_values(0).unique():
                if bug_id in bug_predictions:
                    bug_files = node_features.loc[bug_id]
                    pred_length = len(bug_files)
                    predictions[idx : idx + pred_length] = bug_predictions[bug_id]
                    idx += pred_length

            current_score = evaluate_fold(node_features, predictions, self.metric_type)
            exceeded_weight_score = current_score >= min_acceptable_score

            # 检查评估分数是否有改进
            if epoch == 0 or current_score - best_score > self.tol:
                best_score = current_score
                no_improvement_count = 0
                best_weights = {
                    name: param.clone().detach()
                    for name, param in self.model.state_dict().items()
                }
            else:
                no_improvement_count += 1

            # 记录每一轮日志
            self.training_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "current_score": current_score,
                }
            )

            # 检查是否应该提前停止
            if no_improvement_count >= self.n_iter_no_change and exceeded_weight_score:
                stop_reason = (
                    f"早停：连续{no_improvement_count}轮无改进且达到权重法性能要求"
                )
                break

        # 如果是因为达到最大迭代次数而停止
        if epoch == self.max_iter - 1:
            if exceeded_weight_score:
                stop_reason = f"达到最大迭代次数({self.max_iter})，已达到权重法性能要求"
            else:
                stop_reason = f"达到最大迭代次数({self.max_iter})，未达到权重法性能要求"

        # 恢复最佳权重
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        # 保存训练终止信息
        self.training_summary = {
            "stop_reason": stop_reason,
            "weight_score": weight_score,
            "all_scores": [log["current_score"] for log in self.training_logs],
            "all_losses": [log["train_loss"] for log in self.training_logs],
        }

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

    def clone(self):
        """创建当前回归器的深拷贝，保留所有参数但重置状态"""
        # 提取当前实例的所有标准参数
        params = {}
        for param_name in ModelParameters.get_all_params():
            if hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)

        # 创建新实例
        return GATRegressor(**params)
