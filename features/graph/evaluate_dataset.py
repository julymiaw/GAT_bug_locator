#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集评估脚本：检测并报告数据集中的异常
用法: python evaluate_dataset.py <feature_files_prefix>
支持多进程加速计算
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import load
from collections import defaultdict
import multiprocessing as mp
from functools import partial

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 导入共享常量
node_feature_columns = ["f" + str(i) for i in range(1, 20)]
edge_feature_columns = ["t" + str(i) for i in range(1, 13)]


def numpy_to_native(obj):
    """
    递归地将字典中的NumPy类型转换为Python原生类型
    """
    if isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return numpy_to_native(obj.tolist())
    elif isinstance(obj, pd.Series):
        return numpy_to_native(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return numpy_to_native(obj.to_dict())
    else:
        return obj


# 多进程辅助函数
def _dfs_collect(node, adj_list, visited, component):
    """DFS收集连通分量中的节点"""
    visited.add(node)
    component.append(node)
    for neighbor in adj_list[node]:
        if neighbor not in visited:
            _dfs_collect(neighbor, adj_list, visited, component)


def _bfs_shortest_path(adj_list, source, target):
    """BFS计算两点间最短路径长度"""
    if source == target:
        return 0

    visited = set([source])
    queue = [(source, 0)]  # (节点, 路径长度)

    while queue:
        current, path_len = queue.pop(0)
        for neighbor in adj_list[current]:
            if neighbor not in visited:
                if neighbor == target:
                    return path_len + 1
                visited.add(neighbor)
                queue.append((neighbor, path_len + 1))

    return -1  # 不可达


def _identify_key_nodes(adj_list, nodes, top_k=5):
    """识别关键节点(基于简单的度中心性)"""
    # 计算每个节点的度
    degree = {node: len(adj_list[node]) for node in nodes}
    # 按度排序
    key_nodes = sorted(degree.keys(), key=lambda x: degree[x], reverse=True)
    return key_nodes[:top_k]


def _calculate_message_efficiency(adj_list, nodes, sample_size=100):
    """计算消息传递效率(近似)"""
    if len(nodes) <= 1:
        return 0, 0

    # 随机抽样节点对
    import random

    sample_pairs = []
    nodes_list = list(nodes)
    if len(nodes) * (len(nodes) - 1) // 2 <= sample_size:
        # 图较小，计算所有节点对
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                sample_pairs.append((nodes_list[i], nodes_list[j]))
    else:
        # 随机抽样
        while len(sample_pairs) < sample_size:
            i, j = random.sample(range(len(nodes_list)), 2)
            pair = (nodes_list[i], nodes_list[j])
            if pair not in sample_pairs and (pair[1], pair[0]) not in sample_pairs:
                sample_pairs.append(pair)

    # 计算最短路径
    path_lengths = []
    for source, target in sample_pairs:
        length = _bfs_shortest_path(adj_list, source, target)
        if length > 0:  # 如果节点间有路径
            path_lengths.append(length)

    avg_path = np.mean(path_lengths) if path_lengths else float("inf")
    # 消息传递效率 = 1 / 平均路径长度
    msg_efficiency = 1 / avg_path if avg_path > 0 else 0

    return avg_path, msg_efficiency


def _calculate_hops_to_fix(adj_list, all_nodes, fix_nodes):
    """计算每个节点到最近修复节点的跳数"""
    hops_distribution = defaultdict(int)

    for node in all_nodes:
        if node in fix_nodes:
            hops_distribution[0] += 1
            continue

        # BFS找最短路径
        visited = set([node])
        queue = [(node, 0)]  # (节点, 跳数)
        found = False

        while queue and not found:
            current, hops = queue.pop(0)
            for neighbor in adj_list[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in fix_nodes:
                        hops_distribution[hops + 1] += 1
                        found = True
                        break
                    queue.append((neighbor, hops + 1))

        if not found:
            hops_distribution[-1] += 1  # 无法到达任何修复节点

    return dict(hops_distribution)


def process_bug_structure(bug_id, node_features, edge_features):
    """处理单个bug的图结构分析（用于多进程）"""
    try:
        bug_nodes = node_features.loc[bug_id]
        file_ids = bug_nodes.index.tolist()
        bug_edges = edge_features.loc[bug_id]

        # 获取修复文件
        fix_files = [
            file_id
            for file_id in file_ids
            if bug_nodes.loc[file_id, "used_in_fix"] == 1
        ]

        if not fix_files:  # 跳过没有修复文件的bug
            return None

        # 构建邻接表
        adj_list = defaultdict(list)
        for _, edge in bug_edges.iterrows():
            source = edge["source"]
            target = edge["target"]
            if source in file_ids and target in file_ids:
                adj_list[source].append(target)
                adj_list[target].append(source)  # 假设是无向图

        # 图密度 = 实际边数 / 可能的最大边数
        node_count = len(file_ids)
        max_edges = node_count * (node_count - 1) / 2  # 无向图完全连接的边数
        edge_count = len(bug_edges)
        density = edge_count / max_edges if max_edges > 0 else 0

        # 计算连通分量
        components = []
        visited = set()
        for file_id in file_ids:
            if file_id not in visited:
                component = []
                _dfs_collect(file_id, adj_list, visited, component)
                components.append(component)

        # 分析修复文件在连通分量中的分布
        fix_components = set()  # 包含修复文件的连通分量索引
        for fix_file in fix_files:
            for i, component in enumerate(components):
                if fix_file in component:
                    fix_components.add(i)
                    break

        # 计算平均路径长度和消息传递效率
        avg_path_length, msg_efficiency = _calculate_message_efficiency(
            adj_list, file_ids
        )

        # 计算关键节点指标(介数中心性近似)
        key_nodes = _identify_key_nodes(adj_list, file_ids)

        # 保存该bug的统计信息
        bug_stat = {
            "bug_id": bug_id,
            "nodes": len(file_ids),
            "edges": edge_count,
            "density": density,
            "fix_files": len(fix_files),
            "components": len(components),
            "fix_components": len(fix_components),
            "largest_component": max(len(c) for c in components),
            "avg_path_length": avg_path_length,
            "message_efficiency": msg_efficiency,
            "key_nodes_are_fix": sum(
                1 for node_id in key_nodes[:3] if node_id in fix_files
            )
            / min(3, len(key_nodes)),
        }

        return bug_stat
    except Exception as e:
        print(f"处理bug {bug_id}时出错: {str(e)}")
        return None


def process_bug_message_propagation(bug_stat, node_features, edge_features):
    """处理单个bug的消息传播分析（用于多进程）"""
    try:
        if bug_stat is None:
            return None

        bug_id = bug_stat["bug_id"]
        bug_nodes = node_features.loc[bug_id]
        file_ids = bug_nodes.index.tolist()

        # 获取修复文件
        fix_files = [
            file_id
            for file_id in file_ids
            if bug_nodes.loc[file_id, "used_in_fix"] == 1
        ]

        if not fix_files:
            return None

        # 构建邻接表
        bug_edges = edge_features.loc[bug_id]
        adj_list = defaultdict(list)
        for _, edge in bug_edges.iterrows():
            source = edge["source"]
            target = edge["target"]
            if source in file_ids and target in file_ids:
                adj_list[source].append(target)
                adj_list[target].append(source)

        # 计算到修复文件的最短跳数分布
        hops_distribution = _calculate_hops_to_fix(adj_list, file_ids, fix_files)

        # 计算节点平均邻居数
        avg_neighbors = (
            sum(len(neighbors) for neighbors in adj_list.values()) / len(adj_list)
            if adj_list
            else 0
        )

        return {
            "bug_id": bug_id,
            "distribution": hops_distribution,
            "unreachable": (
                hops_distribution.get(-1, 0) / len(file_ids) if file_ids else 0
            ),
            "avg_hops": (
                np.mean([h for h in hops_distribution.keys() if h >= 0])
                if any(h >= 0 for h in hops_distribution.keys())
                else float("inf")
            ),
            "avg_neighbors": avg_neighbors,
        }
    except Exception as e:
        print(f"处理bug {bug_id}消息传播时出错: {str(e)}")
        return None


class DatasetEvaluator:
    """数据集评估类，用于检测数据异常"""

    def __init__(self, node_features, edge_features, output_dir):
        self.node_features = node_features
        self.edge_features = edge_features
        self.report = {}
        self.output_dir = output_dir
        self.bug_stats = []

    def run_all_checks(self, n_jobs=None):
        """运行所有检查，聚焦于与GNN消息传播相关的指标，支持多进程"""
        if n_jobs is None:
            n_jobs = mp.cpu_count()

        print(f"使用 {n_jobs} 个CPU核心进行计算...")

        self.check_basic_stats()
        self.check_graph_structure_parallel(n_jobs)  # 并行分析图结构
        self.check_message_propagation_parallel(n_jobs)  # 并行分析消息传播特性
        self.visualize_gnn_insights()  # 整合的可视化方法
        return self.report

    def check_basic_stats(self):
        """基本统计信息检查"""
        stats = {
            "节点统计": {
                "总bug数": len(self.node_features.index.get_level_values(0).unique()),
                "总文件数": len(self.node_features.index.levels[1].unique()),
                "修复文件比例": self.node_features["used_in_fix"].mean(),
            },
            "边统计": {
                "总bug数": len(self.edge_features.index.get_level_values(0).unique()),
                "平均边数": (
                    len(self.edge_features)
                    / len(self.edge_features.index.get_level_values(0).unique())
                    if len(self.edge_features.index.get_level_values(0).unique()) > 0
                    else 0
                ),
            },
        }
        self.report["basic_stats"] = stats

    def check_graph_structure_parallel(self, n_jobs):
        """并行分析图结构特性"""
        start_time = time.time()
        print("开始并行分析图结构...")

        node_bug_ids = set(self.node_features.index.get_level_values(0).unique())
        edge_bug_ids = set(self.edge_features.index.get_level_values(0).unique())
        common_bugs = list(node_bug_ids.intersection(edge_bug_ids))

        # 创建进程池
        with mp.Pool(processes=n_jobs) as pool:
            # 使用partial指定固定参数
            process_func = partial(
                process_bug_structure,
                node_features=self.node_features,
                edge_features=self.edge_features,
            )

            # 并行处理所有bug，带进度条
            results = list(pool.map(process_func, common_bugs))

        # 过滤有效结果
        valid_results = [r for r in results if r is not None]
        self.bug_stats = valid_results

        # 收集统计数据
        graph_stats = defaultdict(list)
        for bug_stat in valid_results:
            for k, v in bug_stat.items():
                if isinstance(v, (int, float)):
                    graph_stats[k].append(v)

        # 汇总统计
        summary = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in graph_stats.items()
        }

        # 添加有意义的指标
        summary["消息传递效率与修复文件关系"] = np.corrcoef(
            [s["message_efficiency"] for s in self.bug_stats],
            [
                s["fix_components"] / s["components"] if s["components"] > 0 else 0
                for s in self.bug_stats
            ],
        )[0, 1]

        self.report["graph_structure"] = summary

        print(f"图结构分析完成，耗时: {time.time() - start_time:.2f}秒")

    def check_message_propagation_parallel(self, n_jobs):
        """并行分析消息传播特性"""
        start_time = time.time()
        print("开始并行分析消息传播...")

        # 创建进程池
        with mp.Pool(processes=n_jobs) as pool:
            # 使用partial指定固定参数
            process_func = partial(
                process_bug_message_propagation,
                node_features=self.node_features,
                edge_features=self.edge_features,
            )

            # 并行处理所有bug，带进度条
            results = list(pool.map(process_func, self.bug_stats))

        # 过滤有效结果
        hops_to_fix_stats = [r for r in results if r is not None]

        # 提取平均邻居数
        avg_neighbors_stats = [stat["avg_neighbors"] for stat in hops_to_fix_stats]

        self.report["message_propagation"] = {
            "avg_hops_to_fix": np.mean(
                [
                    s["avg_hops"]
                    for s in hops_to_fix_stats
                    if s["avg_hops"] != float("inf")
                ]
            ),
            "unreachable_nodes_ratio": np.mean(
                [s["unreachable"] for s in hops_to_fix_stats]
            ),
            "avg_neighbors": np.mean(avg_neighbors_stats),
            "hops_distribution": {
                k: np.mean(
                    [
                        s["distribution"].get(k, 0)
                        / len(self.node_features.loc[s["bug_id"]])
                        for s in hops_to_fix_stats
                    ]
                )
                for k in set().union(
                    *[s["distribution"].keys() for s in hops_to_fix_stats]
                )
            },
        }

        print(f"消息传播分析完成，耗时: {time.time() - start_time:.2f}秒")

    def visualize_gnn_insights(self):
        """生成针对GNN洞察的可视化图表"""
        start_time = time.time()
        print("生成可视化图表...")

        os.makedirs(self.output_dir, exist_ok=True)

        # 1. 可视化消息传播效率与修复文件分散度的关系
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [s["message_efficiency"] for s in self.bug_stats],
            [
                s["fix_components"] / s["components"] if s["components"] > 0 else 0
                for s in self.bug_stats
            ],
            alpha=0.6,
        )
        plt.xlabel("消息传播效率")
        plt.ylabel("修复文件分散度(修复文件连通分量/总连通分量)")
        plt.title("消息传播效率与修复文件分散度的关系")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(
            os.path.join(self.output_dir, "message_efficiency_vs_fix_distribution.png")
        )
        plt.close()

        # 2. 可视化"到修复文件的跳数"分布
        hops_dist = self.report["message_propagation"]["hops_distribution"]
        hops = sorted([k for k in hops_dist.keys() if k >= 0])
        plt.figure(figsize=(10, 6))
        plt.bar(
            [str(h) for h in hops] + ["不可达"],
            [hops_dist[h] for h in hops] + [hops_dist.get(-1, 0)],
        )
        plt.xlabel("到最近修复文件的跳数")
        plt.ylabel("节点占比")
        plt.title("节点到修复文件的距离分布")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "hops_to_fix_distribution.png"))
        plt.close()

        # 3. 图密度与连通性的关系
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [s["density"] for s in self.bug_stats],
            [s["components"] for s in self.bug_stats],
            alpha=0.6,
        )
        plt.xlabel("图密度")
        plt.ylabel("连通分量数量")
        plt.title("图密度与连通分量数量的关系")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "density_vs_connectivity.png"))
        plt.close()

        # 4. 关键节点是修复文件的比例分布
        plt.figure(figsize=(10, 6))
        plt.hist(
            [s["key_nodes_are_fix"] for s in self.bug_stats], bins=10, range=(0, 1)
        )
        plt.xlabel("关键节点是修复文件的比例")
        plt.ylabel("Bug数量")
        plt.title("关键节点包含修复文件的比例分布")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "key_nodes_fix_ratio.png"))
        plt.close()

        print(f"可视化完成，耗时: {time.time() - start_time:.2f}秒")


def main():
    parser = argparse.ArgumentParser(description="数据集异常检测 (多进程加速版)")
    parser.add_argument(
        "file_prefix", type=str, help="特征文件前缀", default="aspectj", nargs="?"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="dataset_analysis", help="输出目录"
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="使用的CPU核心数，默认使用所有可用核心",
    )
    args = parser.parse_args()

    file_prefix = args.file_prefix
    output_dir = (
        f"{file_prefix}_{args.output}"
        if not args.output.startswith(file_prefix)
        else args.output
    )
    n_jobs = args.jobs if args.jobs is not None else mp.cpu_count()

    # 显示系统信息
    print(f"系统CPU核心数: {mp.cpu_count()}")
    print(f"将使用{n_jobs}个核心进行计算")

    # 加载数据
    start_time = time.time()
    print(f"加载数据集 {file_prefix}...")
    try:
        (
            fold_number,
            fold_testing,
            _,
            fold_dependency_testing,
            _,
        ) = load(f"../joblib_memmap_{file_prefix}_graph/data_memmap", mmap_mode="r")
        print(f"成功加载数据，折数: {fold_number}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        sys.exit(1)

    # 拼接所有fold的数据
    node_features = pd.concat(fold_testing, axis=0)
    node_features.index = node_features.index.droplevel(0)

    edge_features = pd.concat(fold_dependency_testing, axis=0)
    edge_features.index = edge_features.index.droplevel(0)
    print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 开始分析
    total_start_time = time.time()
    print(f"开始分析数据集...")
    evaluator = DatasetEvaluator(node_features, edge_features, output_dir)
    report = evaluator.run_all_checks(n_jobs=n_jobs)

    # 保存报告
    with open(os.path.join(output_dir, "dataset_report.json"), "w") as f:
        # 将NumPy类型转换为Python原生类型
        native_report = numpy_to_native(report)
        json.dump(native_report, f, indent=4, ensure_ascii=False)

    total_time = time.time() - total_start_time
    print(f"分析完成，总耗时: {total_time:.2f}秒")
    print(f"报告已保存至 {output_dir}/")


if __name__ == "__main__":
    # 对于多进程代码，必须使用这种保护
    mp.freeze_support()
    main()
