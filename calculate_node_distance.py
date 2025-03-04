import torch
import numpy as np
import networkx as nx
from tqdm import tqdm


def calculate_node_distance(dependency_path, class_idx_path, output_path):
    # 加载依赖关系数据和类索引字典
    sources_tensor, targets_tensor, _ = torch.load(dependency_path, weights_only=True)
    class_idx_dict = torch.load(class_idx_path, weights_only=True)

    num_classes = len(class_idx_dict)

    # 创建无向图
    G = nx.Graph()
    G.add_nodes_from(range(num_classes))
    edges = zip(sources_tensor.tolist(), targets_tensor.tolist())
    G.add_edges_from(edges)

    # 初始化距离矩阵
    distance_matrix = np.full((num_classes, num_classes), float("inf"))
    np.fill_diagonal(distance_matrix, 0)

    # 使用 Dijkstra 算法计算最短路径
    for source in tqdm(range(num_classes), desc="Calculating shortest paths"):
        lengths = nx.single_source_dijkstra_path_length(G, source)
        for target, length in lengths.items():
            distance_matrix[source, target] = length

    # 转换为 PyTorch 张量并保存
    distance_tensor = torch.tensor(distance_matrix, dtype=torch.float)
    torch.save(distance_tensor, output_path)


if __name__ == "__main__":
    dependency_path = "model/dataset/dependency.pt"
    class_idx_path = "model/dataset/class_idx_dict.pt"
    output_path = "model/dataset/node_distance.pt"
    calculate_node_distance(dependency_path, class_idx_path, output_path)
