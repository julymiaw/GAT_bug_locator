#!/bin/bash

PROJECT_NAME=aspectj
DATASET_NAME=Birt

# 切换到项目目录
cd $PROJECT_NAME

# 转换 xml 文件为 json 格式
# ../tracking_buggy_files/process_bug_reports.py ../../bug_report/${DATASET_NAME}.xml ../../source/aspectj_dataset/ aspectj_base.json

# 使用 Python 2，获取 timestamp 和 preceding_commit 信息
# /home/july/anaconda3/envs/dataset/bin/python ../tracking_buggy_files/fix_and_augment.py aspectj_base.json ../../source/aspectj_dataset/ > aspectj_aug.json

# 使用 Python 2，计算 bug 频率
# /home/july/anaconda3/envs/dataset/bin/python ../tracking_buggy_files/pick_bug_freq.py aspectj_aug.json ../../source/aspectj_dataset/ > aspectj.json

# 添加缺少的缺陷报告描述
# python ../tracking_buggy_files/add_missing_description_as_separate_reports.py aspectj_base.json aspectj_base_with_descriptions.json rwQASfXyTqPxr6z4AiWMvBzX2EetoufpNYr912Bo https://bugs.eclipse.org/bugs/rest/

# 创建 ast 缓存（通过noSql数据库）
# ../tracking_buggy_files/create_ast_cache.py ../../source/aspectj_dataset/ aspectj.json aspectj

# 将 AST（抽象语法树）和 bug 报告数据进行向量化处理
# ../tracking_buggy_files/vectorize_ast.py aspectj.json aspectj

# 对丰富的 API 数据进行向量化处理，并生成 TF-IDF 矩阵
# ../tracking_buggy_files/vectorize_enriched_api.py aspectj.json aspectj

# 将 bug 报告和相关的 AST（抽象语法树）数据转换为 TF-IDF 矩阵
# ../tracking_buggy_files/convert_tf_idf.py aspectj.json aspectj

# 计算特征 3：使用协同过滤计算缺陷报告和源代码文件的相似度
# ../tracking_buggy_files/calculate_feature_3.py aspectj.json aspectj

# 计算特征 5，6：缺陷修复的最近程度和修复频率
# ../tracking_buggy_files/retrieve_features_5_6.py aspectj.json aspectj

# 计算特征 15，16，17，18，19：与图相关的特征（入边，出边，PageRank，Authority score，Hub score）
../tracking_buggy_files/calculate_notes_graph_features.py aspectj.json aspectj ../../source/aspectj_dataset/

# 整合 19 个特征
../tracking_buggy_files/calculate_vectorized_features.py aspectj.json aspectj

# 按照 500 个一组，进行打包
../tracking_buggy_files/save_normalized_fold_dataframes.py aspectj.json aspectj

# 将特征文件中的数据载入到内存映射文件中
../tracking_buggy_files/load_data_to_joblib_memmap.py aspectj

cd ../joblib_memmap_aspectj

# 运行自适应方法的训练脚本
../tracking_buggy_files/train_adaptive.py aspectj