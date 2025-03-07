import pandas as pd
import numpy as np
from collections import defaultdict
import gc
from multiprocessing import Pool, cpu_count
import time

project = "jdt"
fold_number = 12

# project='eclipse_platform_ui'
# fold_number=12

# project = "tomcat"
# fold_number = 2

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

num_cpu = cpu_count()
print(f"CPU核心数: {num_cpu}")

start_time = time.time()


def load_test_fold_mmap(k):
    filename = f"../{project}/{project}_normalized_testing_fold_{k}"
    return pd.read_pickle(filename, compression=None)


print("正在加载测试集分片...")
with Pool(num_cpu) as p:
    test_fold_03 = p.map(load_test_fold_mmap, range(fold_number + 1))

test_fold_all = pd.concat(test_fold_03)
del test_fold_03
gc.collect()

print(f"测试集加载完成，耗时: {time.time() - start_time:.2f}秒")
phase_time = time.time()

print("建立报告-代码映射关系...")
report_idx = test_fold_all.index.get_level_values(0).unique()


# 优化1: 将报告按批次分组处理，减少进程间通信开销
def process_report_batch(report_batch):
    results = {}
    for report in report_batch:
        test_set = test_fold_all.loc[report].index.get_level_values(0).unique()
        for codeId in test_set:
            results[(report, codeId)] = codeId
    return results


# 创建更大批次的任务
batch_size = max(1, len(report_idx) // (num_cpu * 4))  # 每个CPU处理多个批次
report_batches = [
    report_idx[i : i + batch_size] for i in range(0, len(report_idx), batch_size)
]
print(f"将{len(report_idx)}个报告分成{len(report_batches)}批处理")

# 优化2: 减少进程数或直接串行处理(如果数据量不大)
if len(report_batches) > num_cpu * 2:  # 足够多的批次才值得并行
    with Pool(num_cpu) as p:
        batch_results = p.map(process_report_batch, report_batches)

    reportId2codeId = {}
    for result in batch_results:
        reportId2codeId.update(result)
else:
    # 串行处理，避免进程通信开销
    reportId2codeId = {}
    for batch in report_batches:
        results = process_report_batch(batch)
        reportId2codeId.update(results)

print(f"报告-代码映射关系建立完成，耗时: {time.time() - phase_time:.2f}秒")
phase_time = time.time()

print("加载向量数据...")
nl_vecs = np.load(f"../joblib_memmap_{project}_flim/nl_vecs.npy")
nl_urls = np.load(f"../joblib_memmap_{project}_flim/nl_urls.npy")
code_vecs = np.load(f"../joblib_memmap_{project}_flim/code_vecs.npy")
code_urls = np.load(f"../joblib_memmap_{project}_flim/code_urls.npy")

reportId2nlvec = {reportId: nl_vec for reportId, nl_vec in zip(nl_urls, nl_vecs)}
codeId2codevec = defaultdict(list)
for codeId, code_vec in zip(code_urls, code_vecs):
    codeId2codevec[codeId].append(code_vec)

for codeId in codeId2codevec:
    codeId2codevec[codeId] = np.array(codeId2codevec[codeId])

print(f"向量数据加载完成，耗时: {time.time() - phase_time:.2f}秒")
phase_time = time.time()

print("计算特征分数...")


# 优化3: 对特征分数计算同样应用批处理策略
def compute_scores_batch(items_batch):
    results = []
    for (reportId, codeId), _ in items_batch:
        nl_vec = reportId2nlvec[reportId]
        code_vec = codeId2codevec[codeId]
        scores = np.matmul(nl_vec, code_vec.T)
        max_score = np.max(scores)
        mean_score = np.mean(scores)
        results.append((reportId, codeId, max_score, mean_score))
    return results


items = list(reportId2codeId.items())
batch_size = max(1, len(items) // (num_cpu * 4))
item_batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
print(f"将{len(items)}个特征分数计算任务分成{len(item_batches)}批处理")

with Pool(num_cpu) as p:
    batch_results = p.map(compute_scores_batch, item_batches)

report2codeScores = []
for batch in batch_results:
    report2codeScores.extend(batch)

print(f"特征分数计算完成，耗时: {time.time() - phase_time:.2f}秒")
phase_time = time.time()

print("生成特征矩阵...")
bid_list, fid_list, max_values, mean_values = zip(*report2codeScores)

df = pd.DataFrame(
    0, index=range(len(max_values)), columns=[f"f{i}" for i in range(20, 39)]
)
df["f37"] = max_values
df["f38"] = mean_values

min_vals = df.min()
max_vals = df.max()
df = (df - min_vals) / (max_vals - min_vals)

df["bid"] = bid_list
df["fid"] = fid_list
df.set_index(["bid", "fid"], inplace=True)

test_fold_all.index.names = ["bid", "fid"]

print(f"特征矩阵生成完成，耗时: {time.time() - phase_time:.2f}秒")
phase_time = time.time()


# 优化4: 调整并行处理fold的策略
def process_fold(k, is_training=True):
    fold_type = "training" if is_training else "testing"
    input_path = f"../{project}/{project}_normalized_{fold_type}_fold_{k}"
    output_path = f"../{project}/{project}_normalized_{fold_type}_fold_{k}_flim"

    fold_data = pd.read_pickle(input_path)
    fold_data.index.names = ["bid", "fid"]
    all_dataframe = fold_data.join(df, how="inner")
    all_dataframe.to_pickle(output_path)
    return k


print("并行合并训练集和测试集数据...")
training_args = [(k, True) for k in range(fold_number + 1)]
testing_args = [(k, False) for k in range(fold_number + 1)]
all_args = training_args + testing_args

worker_count = min(num_cpu, len(all_args))
print(f"使用{worker_count}个工作进程处理{len(all_args)}个数据合并任务")

with Pool(worker_count) as p:
    p.starmap(process_fold, all_args)

print(f"数据合并完成，耗时: {time.time() - phase_time:.2f}秒")

# 清理内存
del nl_vecs, code_vecs, nl_urls, code_urls, reportId2nlvec, codeId2codevec
gc.collect()

print(f"总耗时: {time.time() - start_time:.2f}秒")
print("完成！")
