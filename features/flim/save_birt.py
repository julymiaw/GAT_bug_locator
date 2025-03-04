import pandas as pd
import numpy as np
from collections import defaultdict
import gc
from tqdm import tqdm
from multiprocessing import Pool

# project='jdt'
# fold_number=12

# project='eclipse_platform_ui'
# fold_number=12

project = "tomcat"
fold_number = 2

# project = "aspectj"
# fold_number = 1

# project = "swt"
# fold_number = 8

# project='birt'
# fold_number=8

# import sys
# project=sys.argv[1]
# fold_number=int(sys.argv[2])
# print(project,fold_number)

USE_TQDM = False
NUM_THREADS = 28  # 使用的CPU核心数


def dynamic_chunksize(total_tasks, num_threads):
    return max(total_tasks // (num_threads * 4), 1)


def load_test_fold(k):
    return pd.read_pickle(
        "../" + project + "/" + project + "_normalized_testing_fold_" + str(k)
    )


print("正在加载测试集分片...")
total_tasks = fold_number + 1
chunksize = dynamic_chunksize(total_tasks, NUM_THREADS)
with Pool(NUM_THREADS) as p:
    test_fold_03 = list(
        tqdm(
            p.imap(load_test_fold, range(total_tasks), chunksize=chunksize),
            total=total_tasks,
            desc="加载测试集分片",
            disable=not USE_TQDM,
        )
    )

test_fold_all = pd.concat(test_fold_03)
del test_fold_03
gc.collect()

report_idx = test_fold_all.index.get_level_values(0).unique()
code_idx = test_fold_all.index.get_level_values(1).unique()

reportId2codeId = dict()
print("\n建立报告-代码映射关系...")


def process_report(report):
    test_set = test_fold_all.loc[report].index.get_level_values(0).unique()
    report_code_map = {}
    for codeId in test_set:
        report_code_map[report + "_" + codeId] = codeId
    return report_code_map


total_tasks = len(report_idx)
chunksize = dynamic_chunksize(total_tasks, NUM_THREADS)
if USE_TQDM:
    with Pool(NUM_THREADS) as p:
        results = list(
            tqdm(
                p.imap(process_report, report_idx, chunksize=chunksize),
                total=total_tasks,
                desc="处理报告",
            )
        )
else:
    with Pool(NUM_THREADS) as p:
        results = p.map(process_report, report_idx, chunksize=chunksize)

for result in results:
    reportId2codeId.update(result)

nl_vecs = np.load("../joblib_memmap_" + project + "_flim/nl_vecs.npy")
code_vecs = np.load("../joblib_memmap_" + project + "_flim/code_vecs.npy")
code_urls = np.load("../joblib_memmap_" + project + "_flim/code_urls.npy")
nl_urls = np.load("../joblib_memmap_" + project + "_flim/nl_urls.npy")

reportId2nlvec = dict()
for reportId, nl_vec in zip(nl_urls, nl_vecs):
    reportId2nlvec[reportId] = nl_vec
codeId2codevec = defaultdict(list)
for codeId, code_vec in zip(code_urls, code_vecs):
    codeId2codevec[codeId].append(code_vec)

print("\n计算特征分数...")


def compute_scores(item):
    reportId, codeId = item
    reportId = reportId.split("_")[0]
    nl_vec = reportId2nlvec[reportId]
    code_vec = codeId2codevec[codeId]
    code_vec = np.array(code_vec)
    scores = np.matmul(nl_vec, code_vec.T)
    return list(scores)


total_tasks = len(reportId2codeId)
chunksize = dynamic_chunksize(total_tasks, NUM_THREADS)
if USE_TQDM:
    with Pool(NUM_THREADS) as p:
        report2codeScores = list(
            tqdm(
                p.imap(compute_scores, reportId2codeId.items(), chunksize=chunksize),
                total=total_tasks,
                desc="计算相似度分数",
            )
        )
else:
    with Pool(NUM_THREADS) as p:
        report2codeScores = p.map(
            compute_scores, reportId2codeId.items(), chunksize=chunksize
        )

bid_list = []
codeId_list = []
for key, val in reportId2codeId.items():
    try:
        bid_list.append(key.split("_")[0])
        codeId_list.append(val)
    except:
        print(key, val)

print("\n生成特征矩阵...")


def fill_features(i_val):
    i, val = i_val
    feature = [0] * 19
    for j, fea in enumerate(val):
        feature[j] = fea
    feature[-2] = max(val)
    feature[-1] = np.mean(val)
    return feature


total_tasks = len(report2codeScores)
chunksize = dynamic_chunksize(total_tasks, NUM_THREADS)
if USE_TQDM:
    with Pool(NUM_THREADS) as p:
        features = list(
            tqdm(
                p.imap(
                    fill_features, enumerate(report2codeScores), chunksize=chunksize
                ),
                total=total_tasks,
                desc="填充特征",
            )
        )
else:
    with Pool(NUM_THREADS) as p:
        features = p.map(
            fill_features, enumerate(report2codeScores), chunksize=chunksize
        )

df = pd.DataFrame(features, columns=["f" + str(i) for i in range(20, 39)])
min_df = pd.DataFrame(df.min()).transpose()
max_df = pd.DataFrame(df.max()).transpose()
df = (df - min_df.min()) / (max_df.max() - min_df.min())
del features
del code_vecs
del nl_vecs
gc.collect()
df["bid"] = bid_list
df["fid"] = codeId_list
test_fold_all.index.names = ["bid", "fid"]
test_fold_all.head()
df.set_index(["bid", "fid"], inplace=True)

print("\n合并训练集数据...")
for k in tqdm(range(fold_number + 1), desc="训练集分片处理", disable=not USE_TQDM):
    training_fold_k = pd.read_pickle(
        "../" + project + "/" + project + "_normalized_training_fold_" + str(k)
    )
    training_fold_k.index.names = ["bid", "fid"]
    all_dataframe = training_fold_k.join(df, how="inner")
    all_dataframe.to_pickle(
        "../"
        + project
        + "/"
        + project
        + "_normalized_training_fold_"
        + str(k)
        + "_flim"
    )

print("\n合并测试集数据...")
for k in tqdm(range(fold_number + 1), desc="测试集分片处理", disable=not USE_TQDM):
    training_fold_k = pd.read_pickle(
        "../" + project + "/" + project + "_normalized_testing_fold_" + str(k)
    )
    training_fold_k.index.names = ["bid", "fid"]
    all_dataframe = training_fold_k.join(df, how="inner")
    all_dataframe.to_pickle(
        "../" + project + "/" + project + "_normalized_testing_fold_" + str(k) + "_flim"
    )

print("\n完成！")
