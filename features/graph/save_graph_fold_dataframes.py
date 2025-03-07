#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成用于图神经网络的文件依赖图数据集

此脚本处理从类级别的依赖关系到文件级别依赖关系的转换，并过滤保存用于图神经网络训练和测试的数据。
主要工作流程：
1. 读取类信息和类依赖关系
2. 将类依赖关系转换为文件依赖关系
3. 过滤训练集和测试集，保留有效的文件及其依赖关系
4. 保存处理后的数据集，包括：
   - 按 fold 分割的训练和测试数据
   - 对应的文件依赖关系图

使用方法:
    python save_graph_fold_dataframes.py <feature_files_prefix>

参数:
    feature_files_prefix: 数据集前缀，用于定位和命名相关文件

依赖条件:
    - 需要先运行 save_normalized_fold_dataframes.py 生成基础数据
    - 需要 class_info.csv 和 dependency.csv 类依赖关系文件

输出:
    - {prefix}_normalized_training_fold_{i}_graph: 训练集 fold 数据
    - {prefix}_normalized_testing_fold_{i}_graph: 测试集 fold 数据
    - {prefix}_dependency_training_fold_{i}_graph: 训练集依赖图数据
    - {prefix}_dependency_testing_fold_{i}_graph: 测试集依赖图数据
    - {prefix}_dependency_type_mapping.json: 依赖类型映射字典

作者: SRTP 团队, 2024-2025
"""

import sys
import json
import pickle
import pandas as pd
from unqlite import UnQLite
from multiprocessing import Pool, cpu_count
from typing import Dict, List

from train_utils import load_data_folds, eprint


def create_file_dependency_dataframe(
    bug_id: str,
    data: bytes,
    class_info_df: pd.DataFrame,
    class_dependency_df: pd.DataFrame,
    inverse_dependency_type_mapping: Dict[str, str],
):
    """
    根据类依赖关系创建文件依赖关系。

    通过将类级别的依赖关系映射到文件级别，创建用于图神经网络的文件依赖关系图。
    使用类名到 SHA 的映射，将类之间的依赖关系转换为文件之间的依赖关系。

    Args:
        bug_id: 当前 bug 的长 ID。
        data: 数据集信息。
        class_info_df: 类信息 DataFrame。
        class_dependency_df: 依赖关系 DataFrame。
        inverse_dependency_type_mapping: 依赖类型映射。

    Returns:
        bug_id: 当前 bug 的短 ID。
        shas: 原始 shas 列表。
        new_shas: 新 shas 列表。
        file_dependency_df: 文件依赖关系 DataFrame。
    """
    data: dict = pickle.loads(data)
    class_name_to_sha: Dict[str, str] = data.get("class_name_to_sha", {})
    sha_to_file_name: Dict[str, str] = data.get("sha_to_file_name", {})
    shas: List[str] = data.get("shas", [])

    # 创建类名到 sha 的映射
    class_to_sha: Dict[str, str] = {}

    # 使用文件路径匹配 sha
    for _, row in class_info_df.iterrows():
        for sha, file_name in sha_to_file_name.items():
            if file_name == row["filePath"]:
                class_to_sha[row["className"]] = sha
                break

    # 使用类名匹配 sha
    for class_name in class_info_df["className"]:
        if class_name not in class_to_sha and class_name in class_name_to_sha:
            class_to_sha[class_name] = class_name_to_sha[class_name]

    new_shas = list(set(class_to_sha.values()))

    # 过滤依赖关系
    filtered_dependencies = [
        {
            "source": class_to_sha[row["source"]],
            "target": class_to_sha[row["target"]],
            "type": inverse_dependency_type_mapping[row["type"]],
        }
        for _, row in class_dependency_df.iterrows()
        if row["source"] in class_to_sha
        and row["target"] in class_to_sha
        and class_to_sha[row["source"]] != class_to_sha[row["target"]]
    ]

    # 创建文件依赖图
    if filtered_dependencies:
        file_dependency_df = pd.DataFrame(filtered_dependencies)
        file_dependency_df = pd.get_dummies(
            file_dependency_df, columns=["type"], prefix="", prefix_sep=""
        )
        file_dependency_df = (
            file_dependency_df.groupby(["source", "target"]).sum().reset_index()
        )
    else:
        eprint(f"Bug {bug_id} 中没有依赖关系")
        file_dependency_df = pd.DataFrame()

    return bug_id[0:7], shas, new_shas, file_dependency_df


def process_dependency_graph(file_prefix):
    """
    将类依赖图转换为文件依赖图。

    该函数读取类信息和类依赖关系，然后将它们转换为文件级别的依赖关系。
    它还计算了由于类到文件映射不完整而导致的文件（SHA）丢失比例。

    Args:
        file_prefix: 数据集前缀

    Returns:
        bug_dependency_graphs: 一个字典，键是 bug ID（短 ID），值是文件依赖关系 DataFrame。
        new_shas_dict: 一个字典，键是 bug ID（短 ID），值是该 bug 中包含的新 SHA 列表。
    """

    class_info_df = pd.read_csv(f"../../dataset/{file_prefix}_dataset/class_info.csv")
    class_dependency_df = pd.read_csv(
        f"../../dataset/{file_prefix}_dataset/dependency.csv"
    )

    # 自动分配依赖类型
    dependency_types = sorted(class_dependency_df["type"].unique())
    dependency_type_mapping = {
        f"t{i+1}": type_name for i, type_name in enumerate(dependency_types)
    }
    inverse_dependency_type_mapping = {v: k for k, v in dependency_type_mapping.items()}

    with open(f"../{file_prefix}/{file_prefix}_dependency_type_mapping.json", "w") as f:
        json.dump(dependency_type_mapping, f, ensure_ascii=False)

    # 读取 bug_report_files_collection_db
    db_path = f"../{file_prefix}/{file_prefix}_bug_report_files_collection_db"
    bug_report_files_collection_db = UnQLite(db_path, flags=0x00000100 | 0x00000001)

    # 获取 bug_id 列表
    bug_ids = list(bug_report_files_collection_db.keys())

    # 使用 Pool 并行处理每个 bug_id
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            create_file_dependency_dataframe,
            [
                (
                    bug_id,
                    bug_report_files_collection_db[bug_id],
                    class_info_df,
                    class_dependency_df,
                    inverse_dependency_type_mapping,
                )
                for bug_id in bug_ids
            ],
        )

    bug_report_files_collection_db.close()

    # 处理每个 bug 的依赖关系
    bug_dependency_graphs: Dict[str, pd.DataFrame] = {}
    new_shas_dict: Dict[str, List[str]] = {}

    # 计算丢失的 shas
    all_old_shas = set()
    all_new_shas = set()

    for bug_id, shas, new_shas, file_dependency_df in results:
        all_old_shas.update(shas)
        all_new_shas.update(new_shas)
        if not file_dependency_df.empty:
            bug_dependency_graphs[bug_id] = file_dependency_df
        new_shas_dict[bug_id] = new_shas

    loss_ratio = (len(all_old_shas) - len(all_new_shas)) / len(all_old_shas)
    print(f"类文件丢失的总比例: {loss_ratio:.2%}")

    return bug_dependency_graphs, new_shas_dict


def filter_and_save_fold_data(
    file_prefix: str,
    fold_index: int,
    fold_type: str,
    fold_data: pd.DataFrame,
    new_shas_dict: Dict[str, List[str]],
    bug_dependency_graphs: Dict[str, pd.DataFrame],
):
    """
    过滤fold数据并保存依赖关系图和fold数据。

    过滤流程:
    1. 找出new_shas_dict和fold_data中共有的bug_id
    2. 对每个共有bug_id，找出new_shas_dict[bug_id]和fold数据中对应bug_id下的文件ID交集
    3. 过滤依赖关系图，只保留有效文件间的依赖关系
    4. 分别保存过滤后的fold数据和依赖关系图，都保存为DataFrame格式

    Args:
        file_prefix: 数据集前缀
        fold_index: 当前fold的索引
        fold_type: fold类型，"training"或"testing"
        fold_data: 原始fold数据（多级索引DataFrame，一级为bug_id，二级为file_id）
        new_shas_dict: bug_id到有效file_id列表的映射
        bug_dependency_graphs: bug_id到依赖关系DataFrame的映射
    """
    # 对重复数据进行去重
    fold_data = fold_data.groupby(level=[0, 1]).first()

    # 过滤fold数据和依赖关系
    filtered_fold_data = {}
    filtered_dependency_data = {}

    # 处理每个bug
    for bug_id, bug_data in fold_data.groupby(level=0):
        if bug_id not in new_shas_dict:
            eprint(f"Bug {bug_id} 中包含的文件已全部丢失")
            continue

        # 获取有效文件ID集合
        new_shas = set(new_shas_dict[bug_id])
        bug_file_ids = set(bug_data.index.get_level_values(1))
        valid_file_ids = bug_file_ids.intersection(new_shas)

        if not valid_file_ids:
            eprint(f"Bug {bug_id} 没有匹配到有效的文件")
            continue

        # 过滤bug数据，只保留有效文件
        filtered_bug_data = bug_data.loc[
            bug_data.index.get_level_values(1).isin(valid_file_ids)
        ]
        filtered_fold_data[bug_id] = filtered_bug_data

        # 过滤依赖关系图
        if bug_id in bug_dependency_graphs:
            dependency_data = bug_dependency_graphs[bug_id]
            temp_filtered_dependency = dependency_data[
                dependency_data["source"].isin(valid_file_ids)
                & dependency_data["target"].isin(valid_file_ids)
            ]
            if not temp_filtered_dependency.empty:
                filtered_dependency_data[bug_id] = temp_filtered_dependency

    # 合并并保存过滤后的fold数据
    if filtered_fold_data:
        filtered_fold_df = pd.concat(filtered_fold_data.values())
        filtered_fold_df = filtered_fold_df.reset_index().set_index(["bid", "fid"])
        fold_file = f"../{file_prefix}/{file_prefix}_normalized_{fold_type}_fold_{fold_index}_graph"
        pd.to_pickle(filtered_fold_df, fold_file)
        print(
            f"已保存 {fold_type} fold {fold_index} 数据，包含 {len(filtered_fold_data)} 个bug"
        )
    else:
        eprint(f"{fold_type} fold {fold_index} 中没有有效的数据")

    # 保存依赖关系图
    if filtered_dependency_data:
        filtered_dependency_df = pd.concat(
            filtered_dependency_data.values(), keys=filtered_dependency_data.keys()
        )
        dependency_file = f"../{file_prefix}/{file_prefix}_dependency_{fold_type}_fold_{fold_index}_graph"
        pd.to_pickle(filtered_dependency_df, dependency_file)
        print(
            f"已保存 {fold_type} fold {fold_index} 依赖图，包含 {len(filtered_dependency_data)} 个bug"
        )
    else:
        eprint(f"{fold_type} fold {fold_index} 中没有有效的依赖关系")


def main(file_prefix):
    """
    主函数，处理图数据生成流程。

    流程:
    1. 生成文件依赖关系图
    2. 加载训练和测试fold
    3. 处理每个fold的数据，生成对应的图数据集

    Args:
        file_prefix: 数据集前缀
    """
    # 获取文件依赖关系
    bug_dependency_graphs, new_shas_dict = process_dependency_graph(file_prefix)

    # 读取 fold 信息和 fold 数据
    fold_number, fold_testing, fold_training = load_data_folds(file_prefix)

    # 处理每个 fold
    for i in range(fold_number + 1):
        # 处理训练集
        filter_and_save_fold_data(
            file_prefix,
            i,
            "training",
            fold_training[i],
            new_shas_dict,
            bug_dependency_graphs,
        )

        # 处理测试集
        filter_and_save_fold_data(
            file_prefix,
            i,
            "testing",
            fold_testing[i],
            new_shas_dict,
            bug_dependency_graphs,
        )


if __name__ == "__main__":
    # file_prefix = sys.argv[1]
    file_prefix = "aspectj"
    main(file_prefix)
