#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算数据集映射阶段的统计数据 - 优化版

此脚本专注于计算类到文件映射阶段的统计数据，与原始脚本保持一致。
使用多进程加速处理，并优化数据库连接方式，减少连接开销。
"""

import os
import sys
import pandas as pd
import pickle
from unqlite import UnQLite
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import numpy as np

use_tqdm = True  # 是否使用tqdm进度条
processes = cpu_count()  # 进程数，默认为CPU核心数


def process_bug_batch(batch_data, proc_id, file_prefix, class_info_df):
    """处理一批bug报告，每个进程只建立一次数据库连接"""
    bug_ids, show_progress = batch_data

    # 在进程中打开数据库连接(只连接一次)
    db_path = f"../{file_prefix}/{file_prefix}_bug_report_files_collection_db"
    bug_report_db = UnQLite(db_path, flags=0x00000100 | 0x00000001)

    results = []

    # 使用进度条或普通迭代
    if show_progress and use_tqdm:
        iterator = tqdm(bug_ids, desc=f"进程 {proc_id} 处理中", unit="个")
    else:
        iterator = bug_ids

    for bug_id in iterator:
        try:
            data = pickle.loads(bug_report_db[bug_id])
            short_bug_id = bug_id[:7]

            class_name_to_sha = data.get("class_name_to_sha", {})
            sha_to_file_name = data.get("sha_to_file_name", {})
            shas = data.get("shas", [])

            # 处理数据
            result = {
                "short_bug_id": short_bug_id,
                "initial_shas": set(shas),
                "mapped_shas": set(),
            }

            # 类到文件映射阶段
            class_to_sha = {}

            # 使用文件路径匹配SHA
            for _, row in class_info_df.iterrows():
                for sha, file_name in sha_to_file_name.items():
                    if file_name == row["filePath"]:
                        class_to_sha[row["className"]] = sha
                        break

            # 使用类名匹配SHA
            for class_name in class_info_df["className"]:
                if class_name not in class_to_sha and class_name in class_name_to_sha:
                    class_to_sha[class_name] = class_name_to_sha[class_name]

            # 记录映射阶段数据
            mapped_shas = set(class_to_sha.values())
            result["mapped_shas"] = mapped_shas

            results.append(result)
        except Exception as e:
            print(f"处理 {bug_id} 时出错: {e}")
            continue

    # 关闭数据库连接(一次性关闭)
    bug_report_db.close()

    return results


def calculate_mapping_statistics(file_prefix):
    """计算映射阶段的统计数据，保持与原始脚本结果一致"""
    print(f"\n===== 分析 {file_prefix} 数据集 =====")
    print(f"使用 {processes} 个进程进行计算")

    # 计算fold总数
    fold_files = glob.glob(
        f"../{file_prefix}/{file_prefix}_normalized_training_fold_*_graph"
    )
    fold_numbers = [int(f.split("_fold_")[1].split("_")[0]) for f in fold_files]
    fold_count = max(fold_numbers) + 1 if fold_numbers else 0
    print(f"fold总数: {fold_count}")

    try:
        # 读取原始数据库获取信息
        db_path = f"../{file_prefix}/{file_prefix}_bug_report_files_collection_db"
        if not os.path.exists(db_path):
            print(f"错误: 找不到数据库文件 {db_path}")
            return None

        # 仅读取bug_id列表
        with UnQLite(db_path, flags=0x00000100 | 0x00000001) as db:
            bug_ids = list(db.keys())

        print(f"开始处理 {len(bug_ids)} 个bug报告...")

        # 读取类信息文件
        try:
            class_info_path = f"../../dataset/{file_prefix}_dataset/class_info.csv"
            print(f"读取类信息文件: {class_info_path}")
            class_info_df = pd.read_csv(class_info_path)
        except Exception as e:
            print(f"警告: 无法读取类信息文件: {e}")
            class_info_df = pd.DataFrame(columns=["className", "filePath"])

        # 将bug_ids分成多个批次，为每个进程准备一批数据
        chunks = np.array_split(bug_ids, processes)
        batch_data = [(chunk.tolist(), i == 0) for i, chunk in enumerate(chunks)]

        # 使用进程池处理
        process_batch = partial(
            process_bug_batch, file_prefix=file_prefix, class_info_df=class_info_df
        )
        batch_with_id = [(data, i) for i, data in enumerate(batch_data)]

        with Pool(processes=processes) as pool:
            all_results = pool.starmap(process_batch, batch_with_id)

        # 汇总所有结果
        results = [result for batch_results in all_results for result in batch_results]

        # 统计初始和映射后的文件数
        all_old_shas = set()
        all_new_shas = set()
        bug_with_no_mapping = set()

        for result in results:
            short_bug_id = result["short_bug_id"]
            old_shas = result["initial_shas"]
            new_shas = result["mapped_shas"]

            all_old_shas.update(old_shas)
            all_new_shas.update(new_shas)

            if not new_shas:
                bug_with_no_mapping.add(short_bug_id)

        # 计算统计结果
        initial_files_count = len(all_old_shas)
        mapped_files_count = len(all_new_shas)
        files_loss = initial_files_count - mapped_files_count
        loss_ratio = files_loss / initial_files_count if initial_files_count else 0
        bugs_with_no_mapping_count = len(bug_with_no_mapping)

        # 输出统计结果
        print(f"\n===== 类到文件映射阶段 =====")
        print(f"初始文件总数: {initial_files_count}")
        print(f"映射成功文件数: {mapped_files_count}")
        print(f"文件丢失数: {files_loss}")
        print(f"类文件丢失的总比例: {loss_ratio:.2%}")
        print(f"报告完全丢失数: {bugs_with_no_mapping_count}")

        # 返回原始表格需要的数据格式
        return {
            "文件丢失比例": f"{loss_ratio:.2%}",
            "报告完全丢失": bugs_with_no_mapping_count,
            "fold总数": fold_count,
        }

    except Exception as e:
        print(f"处理 {file_prefix} 数据集时出错: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def main():
    if len(sys.argv) < 2:
        print("使用方法: python calculate_mapping_statistics.py <feature_files_prefix>")
        print("例如: python calculate_mapping_statistics.py jdt")
        datasets = ["aspectj", "swt", "tomcat", "birt", "eclipse", "jdt"]
        print(f"可选数据集: {', '.join(datasets)}")
        return

    file_prefix = sys.argv[1]
    stats = calculate_mapping_statistics(file_prefix)

    if stats:
        print("\n统计结果摘要:")
        print(f"| 仓库名称     | {file_prefix} |")
        print(f"| ------------ | ------------- |")
        print(f"| 文件丢失比例 | {stats['文件丢失比例']} |")
        print(f"| 报告完全丢失 | {stats['报告完全丢失']} |")
        print(f"| fold总数     | {stats['fold总数']} |")


if __name__ == "__main__":
    main()
