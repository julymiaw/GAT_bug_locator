#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <feature_files_prefix>

Requires results of save_graph_fold_dataframes.py
"""

import os

import sys
import shutil

from skopt import dump
from train_utils import load_graph_data_folds, eprint


def main():
    file_prefix = sys.argv[1]
    cwd = os.getcwd()
    folder = os.path.join(cwd, "../joblib_memmap_" + file_prefix + "_graph")

    try:
        shutil.rmtree(folder)
    except:
        eprint("Could not clean-up automatically.")

    (
        fold_number,
        fold_dependency_testing,
        fold_testing,
        fold_dependency_training,
        fold_training,
    ) = load_graph_data_folds(file_prefix)

    print("===fold_testing===")
    fold_testing[0].info(memory_usage="deep")
    print("===fold_training===")
    fold_training[0].info(memory_usage="deep")
    print("===fold_dependency_testing===")
    fold_dependency_testing[0].info(memory_usage="deep")
    print("===fold_dependency_training===")
    fold_dependency_training[0].info(memory_usage="deep")
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    data_filename_memmap = os.path.join(folder, "data_memmap")
    dump(
        [
            fold_number,
            fold_testing,
            fold_training,
            fold_dependency_testing,
            fold_dependency_training,
        ],
        data_filename_memmap,
    )


if __name__ == "__main__":
    main()
