#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <feature_files_prefix>

Loads data from each fold for training and testing

Requires results of save_normalized_fold_dataframes.py
"""
import json
from typing import Dict
import pandas as pd
import sys


def eprint(*args, **kwargs):
    """
    将打印输出重定向到标准错误输出。

    Args:
        *args: 要打印的参数
        **kwargs: 额外的关键字参数
    """
    print(*args, file=sys.stderr, **kwargs)


def main():
    """
    主函数，从命令行获取参数并加载数据。
    """
    file_prefix = sys.argv[1]

    load_data_folds(file_prefix)


def load_fold_number(file_prefix: str):
    """
    加载数据集的折数信息。

    Args:
        file_prefix: 数据集前缀

    Returns:
        int: 数据集的折数
    """
    with open(f"../{file_prefix}/{file_prefix}_fold_info.json, "r") as f:
        fold_info: Dict[str, int] = json.load(f)
        fold_number = fold_info["fold_number"]
        eprint("fold number", fold_number)
        return fold_number


def load_data_folds(file_prefix: str):
    """
    加载每个折的训练和测试数据。

    Args:
        file_prefix: 文件名前缀

    Returns:
        Tuple[int, Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]: 包含以下元素的元组:
            - fold_number: 折数
            - fold_testing: 每个折的测试数据
            - fold_training: 每个折的训练数据
    """
    fold_number = load_fold_number(file_prefix)
    fold_training: Dict[int, pd.DataFrame] = {}
    fold_testing: Dict[int, pd.DataFrame] = {}
    for k in range(fold_number + 1):
        fold_training[k] = pd.read_pickle(
            f"../{file_prefix}/{file_prefix}_normalized_training_fold_{str(k)}_flim"
        )
        fold_testing[k] = pd.read_pickle(
            f"../{file_prefix}/{file_prefix}_normalized_testing_fold_{str(k)}_flim"
        )
        eprint("fold_training", str(k), "shape", fold_training[k].shape)
        eprint("fold_testing", str(k), "shape", fold_testing[k].shape)
    eprint("data loaded")
    return fold_number, fold_testing, fold_training


def load_graph_data_folds(file_prefix: str):
    """
    加载每个折的特征和依赖关系数据。

    Args:
        file_prefix: 文件名前缀

    Returns:
        Tuple: 包含以下元素的元组:
            - fold_number: 折数
            - fold_dependency_testing: 每个折的测试依赖关系数据
            - fold_testing: 每个折的测试特征数据
            - fold_dependency_training: 每个折的训练依赖关系数据
            - fold_training: 每个折的训练特征数据
    """
    fold_number = load_fold_number(file_prefix)
    fold_training: Dict[int, pd.DataFrame] = {}
    fold_dependency_training: Dict[int, pd.DataFrame] = {}
    fold_testing: Dict[int, pd.DataFrame] = {}
    fold_dependency_testing: Dict[int, pd.DataFrame] = {}
    for k in range(fold_number + 1):
        fold_training[k] = pd.read_pickle(
            f"../{file_prefix}/{file_prefix}_normalized_training_fold_{str(k)}_graph"
        )
        fold_dependency_training[k] = pd.read_pickle(
            f"../{file_prefix}/{file_prefix}_dependency_training_fold_{str(k)}_graph"
        )
        fold_testing[k] = pd.read_pickle(
            f"../{file_prefix}/{file_prefix}_normalized_testing_fold_{str(k)}_graph"
        )
        fold_dependency_testing[k] = pd.read_pickle(
            f"../{file_prefix}/{file_prefix}_dependency_testing_fold_{str(k)}_graph"
        )
        eprint("fold_training", str(k), "shape", fold_training[k].shape)
        eprint(
            "fold_dependency_training",
            str(k),
            "shape",
            fold_dependency_training[k].shape,
        )
        eprint("fold_testing", str(k), "shape", fold_testing[k].shape)
        eprint(
            "fold_dependency_testing", str(k), "shape", fold_dependency_testing[k].shape
        )
    eprint("data loaded")
    return (
        fold_number,
        fold_dependency_testing,
        fold_testing,
        fold_dependency_training,
        fold_training,
    )


if __name__ == "__main__":
    main()
