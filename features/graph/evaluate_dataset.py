#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集评估脚本：检测并报告数据集中的异常
用法: python evaluate_dataset.py <feature_files_prefix>
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import load
from collections import defaultdict

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


class DatasetEvaluator:
    """数据集评估类，用于检测数据异常"""

    def __init__(self, node_features, edge_features):
        self.node_features = node_features
        self.edge_features = edge_features
        self.report = {}

    def run_all_checks(self):
        """运行所有检查"""
        self.check_basic_stats()
        self.check_consistency()
        self.check_dangling_edges()
        self.check_feature_distributions()
        self.check_bug_sizes()
        self.check_graph_connectivity()

        # 添加新的分析
        bug_stats = self.analyze_fix_components()
        self.visualize_fix_component_distribution(bug_stats, self.output_dir)
        return self.report

    def check_basic_stats(self):
        """基本统计信息检查"""
        stats = {
            "节点统计": {
                "总行数": len(self.node_features),
                "唯一bug数": len(self.node_features.index.get_level_values(0).unique()),
                "唯一文件数": len(self.node_features.index.levels[1].unique()),
                "修复文件比例": self.node_features["used_in_fix"].mean(),
            },
            "边统计": {
                "总行数": len(self.edge_features),
                # 修改这里，使用get_level_values(0)而不是index.unique()
                "唯一bug数": len(self.edge_features.index.get_level_values(0).unique()),
                "平均每bug边数": (
                    len(self.edge_features)
                    / len(self.edge_features.index.get_level_values(0).unique())
                    if len(self.edge_features.index.get_level_values(0).unique()) > 0
                    else 0
                ),
            },
        }
        self.report["basic_stats"] = stats

    def check_consistency(self):
        """检查节点和边数据的一致性"""
        node_bug_ids = set(self.node_features.index.get_level_values(0).unique())
        edge_bug_ids = set(self.edge_features.index.get_level_values(0).unique())

        common_bugs = node_bug_ids.intersection(edge_bug_ids)
        only_in_nodes = node_bug_ids - edge_bug_ids
        only_in_edges = edge_bug_ids - node_bug_ids

        consistency = {
            "共同bug数": len(common_bugs),
            "仅在节点中": len(only_in_nodes),
            "仅在边中": len(only_in_edges),
            "仅在节点中的bug ID示例": list(only_in_nodes)[:5] if only_in_nodes else [],
            "仅在边中的bug ID示例": list(only_in_edges)[:5] if only_in_edges else [],
        }
        self.report["consistency"] = consistency

    def check_dangling_edges(self):
        """检查悬空边（源节点或目标节点不存在）"""
        dangling_edges = defaultdict(int)
        valid_edges = 0
        invalid_edges = 0

        # 对于每个common bug
        node_bug_ids = set(self.node_features.index.get_level_values(0).unique())
        edge_bug_ids = set(self.edge_features.index.get_level_values(0).unique())
        common_bugs = node_bug_ids.intersection(edge_bug_ids)

        for bug_id in common_bugs:
            # 获取该bug的节点文件ID
            bug_nodes = self.node_features.loc[bug_id]
            file_ids = set(bug_nodes.index.tolist())

            # 获取该bug的边
            bug_edges = self.edge_features.loc[bug_id]

            for _, edge in bug_edges.iterrows():
                if edge["source"] not in file_ids:
                    dangling_edges["source_missing"] += 1
                    invalid_edges += 1
                elif edge["target"] not in file_ids:
                    dangling_edges["target_missing"] += 1
                    invalid_edges += 1
                else:
                    valid_edges += 1

        edge_quality = {
            "有效边数": valid_edges,
            "无效边数": invalid_edges,
            "无效边比例": (
                invalid_edges / (valid_edges + invalid_edges)
                if (valid_edges + invalid_edges) > 0
                else 0
            ),
            "悬空边详情": dict(dangling_edges),
        }
        self.report["edge_quality"] = edge_quality

    def check_feature_distributions(self):
        """检查特征值分布，寻找异常值"""
        node_stats = {}
        for col in node_feature_columns:
            data = self.node_features[col]
            node_stats[col] = {
                "均值": data.mean(),
                "标准差": data.std(),
                "最小值": data.min(),
                "最大值": data.max(),
                "Q1": data.quantile(0.25),
                "中位数": data.median(),
                "Q3": data.quantile(0.75),
                "缺失值": data.isna().sum(),
                "零值比例": (data == 0).mean(),
                "异常值比例": ((data - data.mean()).abs() > 3 * data.std()).mean(),
            }

        edge_stats = {}
        for col in edge_feature_columns:
            data = self.edge_features[col]
            edge_stats[col] = {
                "均值": data.mean(),
                "标准差": data.std(),
                "最小值": data.min(),
                "最大值": data.max(),
                "Q1": data.quantile(0.25),
                "中位数": data.median(),
                "Q3": data.quantile(0.75),
                "缺失值": data.isna().sum(),
                "零值比例": (data == 0).mean(),
                "异常值比例": ((data - data.mean()).abs() > 3 * data.std()).mean(),
            }

        self.report["feature_stats"] = {
            "node_features": node_stats,
            "edge_features": edge_stats,
        }

    def check_bug_sizes(self):
        """检查不同bug的节点和边数量分布"""
        # 统计每个bug的节点数
        bug_node_counts = self.node_features.groupby(level=0).size()

        # 统计每个bug的边数
        bug_edge_counts = self.edge_features.groupby(level=0).size()

        # 计算统计量
        node_count_stats = {
            "均值": bug_node_counts.mean(),
            "标准差": bug_node_counts.std(),
            "最小值": bug_node_counts.min(),
            "最大值": bug_node_counts.max(),
            "中位数": bug_node_counts.median(),
            "零节点bug数": (bug_node_counts == 0).sum(),
        }

        edge_count_stats = {
            "均值": bug_edge_counts.mean(),
            "标准差": bug_edge_counts.std(),
            "最小值": bug_edge_counts.min(),
            "最大值": bug_edge_counts.max(),
            "中位数": bug_edge_counts.median(),
            "零边bug数": (bug_edge_counts == 0).sum(),
        }

        # 寻找异常bug（节点特别多或特别少）
        outlier_threshold = bug_node_counts.mean() + 3 * bug_node_counts.std()
        outlier_bugs = bug_node_counts[bug_node_counts > outlier_threshold]

        self.report["bug_sizes"] = {
            "node_counts": node_count_stats,
            "edge_counts": edge_count_stats,
            "outlier_bugs": {
                "数量": len(outlier_bugs),
                "示例": outlier_bugs.head().to_dict() if not outlier_bugs.empty else {},
            },
        }

    def check_graph_connectivity(self):
        """检查图的连通性"""
        connectivity_stats = defaultdict(int)
        node_bug_ids = set(self.node_features.index.get_level_values(0).unique())
        edge_bug_ids = set(self.edge_features.index.get_level_values(0).unique())
        common_bugs = node_bug_ids.intersection(edge_bug_ids)

        isolated_nodes_by_bug = {}

        for bug_id in common_bugs:
            # 获取该bug的所有节点
            bug_nodes = self.node_features.loc[bug_id]
            file_ids = bug_nodes.index.tolist()
            file_to_idx = {file_id: idx for idx, file_id in enumerate(file_ids)}

            # 获取该bug的所有边
            bug_edges = self.edge_features.loc[bug_id]

            # 构建邻接表
            adj_list = defaultdict(list)
            for _, edge in bug_edges.iterrows():
                source = edge["source"]
                target = edge["target"]
                if source in file_to_idx and target in file_to_idx:
                    adj_list[source].append(target)
                    adj_list[target].append(source)  # 假设是无向图

            # 计算连通分量
            visited = set()
            components = 0
            isolated = 0

            for file_id in file_ids:
                if file_id not in visited:
                    components += 1
                    component_size = self._dfs(file_id, adj_list, visited)
                    if component_size == 1:
                        isolated += 1

            connectivity_stats["总连通分量"] += components
            connectivity_stats["平均每bug连通分量"] += components / len(common_bugs)
            connectivity_stats["孤立节点数"] += isolated

            if isolated > 0:
                isolated_nodes_by_bug[bug_id] = isolated

        self.report["connectivity"] = {
            "stats": dict(connectivity_stats),
            "isolated_nodes_by_bug": isolated_nodes_by_bug,
        }

    def analyze_fix_components(self):
        """分析修复文件在连通分量中的分布"""
        node_bug_ids = set(self.node_features.index.get_level_values(0).unique())
        edge_bug_ids = set(self.edge_features.index.get_level_values(0).unique())
        common_bugs = node_bug_ids.intersection(edge_bug_ids)

        bug_stats = []

        for bug_id in common_bugs:
            # 获取该bug的所有节点
            bug_nodes = self.node_features.loc[bug_id]
            file_ids = bug_nodes.index.tolist()

            # 获取修复文件
            fix_files = [
                file_id
                for file_id in file_ids
                if bug_nodes.loc[file_id, "used_in_fix"] == 1
            ]

            if not fix_files:  # 跳过没有修复文件的bug
                continue

            # 构建邻接表
            adj_list = defaultdict(list)
            bug_edges = self.edge_features.loc[bug_id]

            for _, edge in bug_edges.iterrows():
                source = edge["source"]
                target = edge["target"]
                if source in file_ids and target in file_ids:
                    adj_list[source].append(target)
                    adj_list[target].append(source)  # 假设是无向图

            # 找出所有连通分量
            components = []
            visited = set()

            for file_id in file_ids:
                if file_id not in visited:
                    component = []
                    self._dfs_collect(file_id, adj_list, visited, component)
                    components.append(component)

            # 分析修复文件在连通分量中的分布
            fix_components = set()  # 包含修复文件的连通分量索引

            for fix_file in fix_files:
                for i, component in enumerate(components):
                    if fix_file in component:
                        fix_components.add(i)
                        break

            bug_stats.append(
                {
                    "bug_id": bug_id,
                    "total_files": len(file_ids),
                    "fix_files": len(fix_files),
                    "total_components": len(components),
                    "components_with_fix": len(fix_components),
                    "component_sizes": [len(c) for c in components],
                    "largest_component": max(len(c) for c in components),
                    "fix_in_largest": any(
                        fix_file
                        in components[components.index(max(components, key=len))]
                        for fix_file in fix_files
                    ),
                }
            )

        # 汇总统计
        summary = {
            "分析的bug总数": len(bug_stats),
            "平均文件数": np.mean([s["total_files"] for s in bug_stats]),
            "平均修复文件数": np.mean([s["fix_files"] for s in bug_stats]),
            "平均连通分量数": np.mean([s["total_components"] for s in bug_stats]),
            "平均包含修复文件的连通分量数": np.mean(
                [s["components_with_fix"] for s in bug_stats]
            ),
            "修复文件集中在单个连通分量的bug比例": sum(
                1 for s in bug_stats if s["components_with_fix"] == 1
            )
            / len(bug_stats),
            "修复文件在最大连通分量中的bug比例": sum(
                1 for s in bug_stats if s["fix_in_largest"]
            )
            / len(bug_stats),
        }

        self.report["fix_component_analysis"] = {
            "summary": summary,
            "bug_details": bug_stats[:10],  # 只保存前10个作为示例
        }

        return bug_stats

    def _dfs_collect(self, start, adj_list, visited, result):
        """收集连通分量中的所有节点"""
        stack = [start]
        visited.add(start)
        result.append(start)

        while stack:
            node = stack.pop()
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    stack.append(neighbor)

    def _dfs(self, start, adj_list, visited):
        """深度优先搜索，返回连通分量大小"""
        stack = [start]
        visited.add(start)
        size = 1

        while stack:
            node = stack.pop()
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
                    size += 1

        return size

    def generate_plots(self, output_dir):
        """生成数据可视化图表"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 每个bug的节点数分布
        bug_node_counts = self.node_features.groupby(level=0).size()
        plt.figure(figsize=(10, 6))
        plt.hist(bug_node_counts, bins=30)
        plt.xlabel("每个bug的节点数")
        plt.ylabel("bug数量")
        plt.title("Bug节点数分布")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bug_node_counts.png"))
        plt.close()

        # 2. 每个bug的边数分布
        bug_edge_counts = self.edge_features.groupby(level=0).size()
        plt.figure(figsize=(10, 6))
        plt.hist(bug_edge_counts, bins=30)
        plt.xlabel("每个bug的边数")
        plt.ylabel("bug数量")
        plt.title("Bug边数分布")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bug_edge_counts.png"))
        plt.close()

        # 3. 节点特征箱线图
        plt.figure(figsize=(15, 8))
        self.node_features[node_feature_columns].boxplot()
        plt.xticks(rotation=90)
        plt.title("节点特征分布")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "node_features_boxplot.png"))
        plt.close()

        # 4. 边特征箱线图
        plt.figure(figsize=(15, 8))
        self.edge_features[edge_feature_columns].boxplot()
        plt.xticks(rotation=90)
        plt.title("边特征分布")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "edge_features_boxplot.png"))
        plt.close()

        # 5. 用于修复的文件比例
        plt.figure(figsize=(8, 6))
        self.node_features["used_in_fix"].value_counts().plot(kind="bar")
        plt.xlabel("是否用于修复")
        plt.ylabel("文件数")
        plt.title("修复与非修复文件比例")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fix_ratio.png"))
        plt.close()

        # 6. 边-节点比例散点图
        common_bugs = set(bug_node_counts.index) & set(bug_edge_counts.index)
        data = pd.DataFrame(
            {
                "node_count": bug_node_counts[list(common_bugs)],
                "edge_count": bug_edge_counts[list(common_bugs)],
            }
        )
        plt.figure(figsize=(10, 8))
        plt.scatter(data["node_count"], data["edge_count"], alpha=0.5)
        plt.xlabel("节点数")
        plt.ylabel("边数")
        plt.title("Bug的节点数与边数关系")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "node_edge_ratio.png"))
        plt.close()

    def visualize_fix_component_distribution(self, bug_stats, output_dir):
        """可视化修复文件在连通分量中的分布"""

        # 1. 修复文件连通分量分布
        fix_component_ratios = [
            s["components_with_fix"] / s["total_components"] for s in bug_stats
        ]

        plt.figure(figsize=(10, 6))
        plt.hist(fix_component_ratios, bins=20)
        plt.xlabel("包含修复文件的连通分量比例")
        plt.ylabel("Bug数量")
        plt.title("修复文件在连通分量中的分布")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(output_dir, "fix_component_distribution.png"))
        plt.close()

        # 2. 修复文件与连通分量关系散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [s["total_components"] for s in bug_stats],
            [s["components_with_fix"] for s in bug_stats],
            alpha=0.5,
        )
        plt.xlabel("连通分量总数")
        plt.ylabel("包含修复文件的连通分量数")
        plt.title("连通分量总数与包含修复文件的连通分量关系")
        plt.grid(True, linestyle="--", alpha=0.7)

        # 添加45度参考线（表示所有连通分量都包含修复文件）
        max_val = max(
            max(s["total_components"] for s in bug_stats),
            max(s["components_with_fix"] for s in bug_stats),
        )
        plt.plot([0, max_val], [0, max_val], "r--", alpha=0.3)

        plt.savefig(os.path.join(output_dir, "component_fix_relationship.png"))
        plt.close()

        # 3. 比较包含修复文件的连通分量与不包含的大小差异
        sizes_with_fix = []
        sizes_without_fix = []

        for s in bug_stats:
            if s["fix_in_largest"]:
                sizes_with_fix.append(s["largest_component"])
            else:
                # 找出所有连通分量中最大的那个包含修复文件的分量
                max_size = 0
                for i, comp_size in enumerate(s["component_sizes"]):
                    if i < s["components_with_fix"] and comp_size > max_size:
                        max_size = comp_size
                if max_size > 0:
                    sizes_with_fix.append(max_size)

        plt.figure(figsize=(8, 6))
        plt.boxplot(
            [sizes_with_fix, sizes_without_fix],
            labels=["包含修复文件", "不包含修复文件"],
        )
        plt.ylabel("连通分量大小（节点数）")
        plt.title("包含修复文件与不包含修复文件的连通分量大小比较")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(output_dir, "fix_component_sizes.png"))
        plt.close()


def main():
    # parser = argparse.ArgumentParser(description="数据集异常检测")
    # parser.add_argument("file_prefix", type=str, help="特征文件前缀")
    # parser.add_argument(
    #     "--output", "-o", type=str, default="dataset_analysis", help="输出目录"
    # )
    # args = parser.parse_args()

    # file_prefix = args.file_prefix
    # output_dir = f"{file_prefix}_{args.output}" if not args.output.startswith(file_prefix) else args.output
    file_prefix = "aspectj"
    output_dir = f"{file_prefix}_dataset_analysis"

    # 加载数据
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

    # 使用第一折进行分析
    node_features = fold_testing[0]
    edge_features = fold_dependency_testing[0]

    print(f"开始分析数据集...")
    evaluator = DatasetEvaluator(node_features, edge_features)
    report = evaluator.run_all_checks()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存报告
    with open(os.path.join(output_dir, "dataset_report.json"), "w") as f:
        # 将NumPy类型转换为Python原生类型
        native_report = numpy_to_native(report)
        json.dump(native_report, f, indent=4, ensure_ascii=False)

    # 生成图表
    evaluator.generate_plots(output_dir)

    print(f"分析完成，报告已保存至 {output_dir}/")

    # 打印关键问题
    print("\n===== 数据集关键问题 =====")

    # 一致性问题
    consistency = report["consistency"]
    if consistency["仅在节点中"] > 0 or consistency["仅在边中"] > 0:
        print(f"警告: 节点和边数据不一致")
        print(f"- 共同bug数: {consistency['共同bug数']}")
        print(f"- 仅在节点中的bug数: {consistency['仅在节点中']}")
        print(f"- 仅在边中的bug数: {consistency['仅在边中']}")

    # 悬空边问题
    edge_quality = report["edge_quality"]
    if edge_quality["无效边比例"] > 0:
        print(f"警告: 存在悬空边")
        print(f"- 无效边比例: {edge_quality['无效边比例']:.2%}")
        print(f"- 无效边详情: {edge_quality['悬空边详情']}")

    # 零节点或零边的bug
    bug_sizes = report["bug_sizes"]
    if (
        bug_sizes["node_counts"]["零节点bug数"] > 0
        or bug_sizes["edge_counts"]["零边bug数"] > 0
    ):
        print(f"警告: 存在没有节点或边的bug")
        print(f"- 零节点bug数: {bug_sizes['node_counts']['零节点bug数']}")
        print(f"- 零边bug数: {bug_sizes['edge_counts']['零边bug数']}")

    # 连接性问题
    connectivity = report["connectivity"]["stats"]
    if connectivity["孤立节点数"] > 0:
        print(f"警告: 存在孤立节点")
        print(f"- 孤立节点总数: {connectivity['孤立节点数']}")
        print(f"- 平均每bug连通分量: {connectivity['平均每bug连通分量']:.2f}")


if __name__ == "__main__":
    main()
