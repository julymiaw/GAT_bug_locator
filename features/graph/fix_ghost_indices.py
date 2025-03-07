#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复数据集中的幽灵索引问题

此脚本重建所有训练和测试fold的MultiIndex，确保索引与实际数据一致，
解决"幽灵索引"导致的KeyError问题。

使用方法:
    python fix_ghost_indices.py <dataset_name>

示例:
    python fix_ghost_indices.py aspectj
"""

import sys
import os
import pandas as pd
from train_utils import eprint, load_graph_data_folds


def fix_multiindex(df: pd.DataFrame, is_edge_data=False) -> pd.DataFrame:
    """
    修复DataFrame的MultiIndex结构，消除幽灵索引

    Args:
        df: 包含幽灵索引的DataFrame
        is_edge_data: 是否为边数据(与节点数据处理方式不同)

    Returns:
        修复后的DataFrame
    """
    # 处理边数据的特殊情况
    if is_edge_data:
        df.index.names = ["bid", "index"]

    # 保存原始索引名称
    index_names = df.index.names

    # 重建索引(先重置后重建)
    fixed_df = df.reset_index().set_index(list(index_names))

    # 检查是否有实际bug_id数量变化
    original_bugs = len(df.index.levels[0])
    fixed_bugs = len(fixed_df.index.levels[0])

    if original_bugs != fixed_bugs:
        if is_edge_data:
            eprint(f"边索引修复: 原始边数 {original_bugs} -> 实际边数 {fixed_bugs}")
        else:
            eprint(f"节点索引修复: 原始bug数 {original_bugs} -> 实际bug数 {fixed_bugs}")

    return fixed_df


def fix_dataset(file_prefix: str):
    """修复指定数据集的所有fold文件"""
    # 直接使用train_utils中的函数加载所有数据
    (
        fold_number,
        fold_dependency_testing,
        fold_testing,
        fold_dependency_training,
        fold_training,
    ) = load_graph_data_folds(file_prefix)
    datasets_fixed = 0

    # 处理每个fold
    for fold in range(fold_number + 1):
        # 创建要处理的数据字典
        data_to_fix = {
            "训练节点": (
                fold_training[fold],
                f"../{file_prefix}/{file_prefix}_normalized_training_fold_{fold}_graph",
                False,
            ),
            "测试节点": (
                fold_testing[fold],
                f"../{file_prefix}/{file_prefix}_normalized_testing_fold_{fold}_graph",
                False,
            ),
            "训练边": (
                fold_dependency_training[fold],
                f"../{file_prefix}/{file_prefix}_dependency_training_fold_{fold}",
                True,
            ),
            "测试边": (
                fold_dependency_testing[fold],
                f"../{file_prefix}/{file_prefix}_dependency_testing_fold_{fold}",
                True,
            ),
        }

        for desc, (df, filepath, is_edge) in data_to_fix.items():
            try:
                # 修复MultiIndex
                fixed_df = fix_multiindex(df, is_edge_data=is_edge)

                # 创建备份
                backup_path = f"{filepath}.bak"
                if not os.path.exists(backup_path):
                    pd.to_pickle(df, backup_path)

                # 保存修复后的数据
                pd.to_pickle(fixed_df, filepath)

                eprint(f"Fold {fold} {desc}数据修复完成: {filepath}")
                datasets_fixed += 1

            except Exception as e:
                eprint(f"处理 {desc} 时出错: {str(e)}")

    eprint(f"总计修复 {datasets_fixed} 个数据文件")


def verify_dataset(file_prefix: str):
    """验证修复后的数据集是否存在幽灵索引问题"""
    # 重新加载修复后的数据
    (
        fold_number,
        fold_dependency_testing,
        fold_testing,
        fold_dependency_training,
        fold_training,
    ) = load_graph_data_folds(file_prefix)
    all_valid = True
    valid_folds = set()

    # 创建要验证的数据字典
    for fold in range(fold_number + 1):
        fold_valid = True
        data_to_verify = {
            "训练节点": fold_training[fold],
            "测试节点": fold_testing[fold],
            "训练边": fold_dependency_training[fold],
            "测试边": fold_dependency_testing[fold],
        }

        for desc, df in data_to_verify.items():
            # 检查幽灵索引
            phantom_bugs = []
            for bug_id in df.index.levels[0]:
                try:
                    if len(df.loc[bug_id]) == 0:
                        phantom_bugs.append(bug_id)
                except KeyError:
                    phantom_bugs.append(bug_id)

            if phantom_bugs:
                eprint(
                    f"{desc} Fold {fold} 仍有幽灵索引: {phantom_bugs[:5]}... (共{len(phantom_bugs)}个)"
                )
                fold_valid = False
            else:
                eprint(f"{desc} Fold {fold} 验证通过，无幽灵索引")

        if fold_valid:
            valid_folds.add(fold)
        else:
            all_valid = False

    if all_valid:
        eprint("所有数据集验证通过，幽灵索引问题已解决!")
    else:
        eprint("部分数据集仍存在问题，请检查日志")

    return all_valid, valid_folds


def clean_backups(file_prefix: str, valid_folds: set):
    """清除已验证通过的fold的备份文件"""
    cleaned_files = 0

    for fold in valid_folds:
        backup_files = [
            f"../{file_prefix}/{file_prefix}_normalized_training_fold_{fold}_graph.bak",
            f"../{file_prefix}/{file_prefix}_normalized_testing_fold_{fold}_graph.bak",
            f"../{file_prefix}/{file_prefix}_dependency_training_fold_{fold}.bak",
            f"../{file_prefix}/{file_prefix}_dependency_testing_fold_{fold}.bak",
        ]

        for backup_file in backup_files:
            if os.path.exists(backup_file):
                try:
                    os.remove(backup_file)
                    eprint(f"已清除备份文件: {backup_file}")
                    cleaned_files += 1
                except Exception as e:
                    eprint(f"清除备份文件失败: {backup_file}, 错误: {str(e)}")

    eprint(f"总计清除 {cleaned_files} 个备份文件")


def main():
    """主函数"""
    # if len(sys.argv) != 2:
    #     eprint("用法: python fix_ghost_indices.py <dataset_name>")
    #     sys.exit(1)

    # file_prefix = sys.argv[1]
    file_prefix = "aspectj"
    eprint(f"开始修复数据集 {file_prefix} 的幽灵索引问题...")

    # 修复数据集
    fix_dataset(file_prefix)

    # 验证修复结果
    _, valid_folds = verify_dataset(file_prefix)

    # 清除验证通过的fold的备份文件
    if valid_folds:
        eprint(f"开始清除已验证通过的fold ({len(valid_folds)}个) 的备份文件...")
        clean_backups(file_prefix, valid_folds)


if __name__ == "__main__":
    main()
