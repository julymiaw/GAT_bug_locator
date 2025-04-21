# 复现 tracking buggy files

## 克隆仓库，准备数据集

```bash
git clone https://github.com/mfejzer/tracking_buggy_files.git
```

数据集存放于 `../../source/` 目录下，包含 6 个项目的数据集。为了获取 ast 树和 import graph，使用以下命令：

```bash
git fetch origin refs/notes/commits:refs/notes/commits
git fetch origin refs/notes/tokenized_counters:refs/notes/tokenized_counters
git fetch origin refs/notes/graph:refs/notes/graph
```

## 提取特征

执行完成后，可以把数据集中的 xml 文件转化为 json 格式：

对于 AspectJ 项目：

```bash
cd aspectj
# 转换 xml 文件为 json 格式
../tracking_buggy_files/process_bug_reports.py ../../bug_report/AspectJ.xml ../../source/aspectj_dataset/ aspectj_base.json

# 使用 Python 2，获取 timestamp 和 preceding_commit 信息
../tracking_buggy_files/fix_and_augment.py aspectj_base.json ../../source/aspectj_dataset/ > aspectj_aug.json

# 使用 Python 2，计算 bug 频率
../tracking_buggy_files/pick_bug_freq.py aspectj_aug.json ../../source/aspectj_dataset/ > aspectj.json
```

这个脚本用于计算论文中提到的 19 种特征的第 5 种， `缺陷修复最近程度` ，和第 6 种， `缺陷修复频率` 。

* **缺陷修复最近程度 (f₅)：**
  * 公式：$f_5(r, s) = (r.month - last(r, s).month + 1)^{-1}$
  * 解释：
    * r: 缺陷报告集合
    * s: 文件路径
    * last(r, s): 文件 s 最近一次被修复的缺陷报告
    * r.month: 缺陷报告 r 创建的月份
    * 计算方法：计算缺陷报告创建月份与文件最近一次修复月份的差，加 1 后取倒数。如果文件在缺陷报告创建的当月被修复，则 f₅ = 1；如果文件在上个月被修复，则 f₅ = 0.5，以此类推。
* **缺陷修复频率 (f₆)：**
  * 公式：f₆(r, s) = |br(r, s)|
  * 解释：
    * br(r, s): 在缺陷报告 r 创建之前，对文件 s 进行修改的 commit 列表。

接下来，添加缺少的缺陷报告描述：

```bash
cd aspectj
../tracking_buggy_files/add_missing_description_as_separate_reports.py aspectj_base.json aspectj_base_with_descriptions.json BUGZILLA_API_KEY BUGZILLA_API_URL
```

其中，我的 BUGZILLA_API_KEY 是 `rwQASfXyTqPxr6z4AiWMvBzX2EetoufpNYr912Bo` ，BUGZILLA_API_URL 是 `https://bugs.eclipse.org/bugs/rest/` 。

然后，提取其他特征。

```bash
cd aspectj

# 创建 ast 缓存（通过noSql数据库）
../tracking_buggy_files/create_ast_cache.py ../../source/aspectj_dataset/ aspectj.json aspectj
# ../flim/create_ast_cache.py ../../source/aspectj_dataset/ aspectj.json aspectj

# 将 AST（抽象语法树）和 bug 报告数据进行向量化处理。
../tracking_buggy_files/vectorize_ast.py aspectj.json aspectj

# 对丰富的 API 数据进行向量化处理，并生成 TF-IDF 矩阵。
../tracking_buggy_files/vectorize_enriched_api.py aspectj.json aspectj

# 将 bug 报告和相关的 AST（抽象语法树）数据转换为 TF-IDF 矩阵。
../tracking_buggy_files/convert_tf_idf.py aspectj.json aspectj

# 计算特征 3：使用协同过滤计算缺陷报告和源代码文件的相似度。
../tracking_buggy_files/calculate_feature_3.py aspectj.json aspectj

# 计算特征 5，6：缺陷修复的最近程度和修复频率。
../tracking_buggy_files/retrieve_features_5_6.py aspectj.json aspectj
# 输出：
# max recency 1.0
# max frequency 77.0

# 计算特征 15，16，17，18，19：与图相关的特征（入边，出边，PageRank，Authority score，Hub score）。
../tracking_buggy_files/calculate_notes_graph_features.py aspectj.json aspectj ../../source/aspectj_dataset/

# 整合 19 个特征，其中 5，6，15~19 直接使用此前保存的结果，其他特征重新计算。
# 特征 1：`缺陷报告` 和 `源代码文件` 的 tf-idf 向量的余弦相似度。（表面词汇相似度）
# 特征 2：`丰富后的报告` 与 `丰富后的 API` 的 tf-idf 向量的余弦相似度。（API 增强词汇相似度）
#（以上 2 者均使用所有特征的最大值）
# 特征3：缺陷报告摘要向量 data[report_summary_index, :] 与文件向量 data[file_index, :] 之间的余弦相似度。
# 特征4，缺陷报告摘要 current_bug_report_summary 中包含的类名的最大长度
# 特征7~14：缺陷报告的摘要或描述与代码的类名、方法名、变量名、注释之间的余弦相似度
# 特征7：摘要与类名的相似度
# 特征8：摘要与方法名的相似度
# 特征9：摘要与变量名的相似度
# 特征10：摘要与注释的相似度
# 特征11：描述与类名的相似度
# 特征12：描述与方法名的相似度
# 特征13：描述与变量名的相似度
# 特征14：描述与注释的相似度
../tracking_buggy_files/calculate_vectorized_features.py aspectj.json aspectj

# 按照 500 个一组，进行打包。
# aspectj_fold_info 中，存储 fold 个数：{"fold_number": 1}
# 训练集中，保存相关文件与 200 个不相关文件的 19 个特征，优先选择特征 2 更高的不相关文件。
# 测试集中，保存所有文件的 19 个特征。
# 记录每种特征的最大和最小值，分别归一化。
../tracking_buggy_files/save_normalized_fold_dataframes.py aspectj.json aspectj
```

## 复现自适应方法的结果

这一部分，tracking_buggy_files 仓库通过 load_data_to_joblib_memmap.py 脚本将特征文件中的数据载入到内存映射文件中，它使用 train_utils.py 来读取上一步生成的训练和测试文件。

FLIM 仓库对 train_utils.py 进行了修改，使其能读取自己获取的特征文件，共包含 38 种特征。其中，$f_{33}$ ~ $f_{36}$ 是论文中提到的四种特征，而 $f_{37}$ 和 $f_{38}$ 分别是四种特征的最大值和平均值。

此外，tracking_buggy_files 实现了 train_adaptive 脚本，FLIM 在此基础上实现了 4 个脚本，区别在于特征。

* train_adaptive_mean.py：$f_1$ ~ $f_{19}$ + $f_{38}$
* train_adaptive_max.py：$f_1$ ~ $f_{19}$ + $f_{37}$
* train_adaptive_LSTM.py：$f_1$ ~ $f_{19}$ + $f_{33}$ ~ $f_{38}$
* train_adaptive_feature_combine.py：$f_4$ + $f_{37}$ ~ $f_{38}$

使用方式类似：

```bash
../tracking_buggy_files/load_data_to_joblib_memmap.py aspectj
../tracking_buggy_files/train_adaptive.py aspectj
```

## 主要流程脚本

| 脚本名称                                                     | tracking_buggy_files                                         | flim                                                         | me                                                           | 输入                                                         | 输出                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| _____________________________________________________________________________________________ | _____________________________________________________________________________________________ | _____________________________________________________________________________________________ | _____________________________________________________________________________________________ | ___________________________________________________________________________________________________________________________________________________________________________________________________________________ | ___________________________________________________________________________________________________________________________________________________________________________________________________________________ |
| process_bug_reports                                          | 使用 git 提取 xml 文件中 commit ID 对应的完整 commit 信息    | 无                                                           | 数据集中 commit ID 冲突，根据 bug ID 确认正确的 commit       | {Name}.xml                                                   | {name}\_base.json                                            |
| fix_and_augment                                              | 将比 `bug_report` 更早的第一个提交作为提交者编写 bug 报告时使用的版本，并将该提交的 `sha` 存入 `preceding_commit` 字段 | 改写 lambda 表达式写法，使其兼容 Python 3                    | 采用 flim 改写后的版本                                       | {name}\_base.json                                            | {name}\_aug.json                                             |
| pick_bug_freq                                                | 计算 `缺陷修复最近程度` 和 `缺陷修复频率`                    | 去除了时间戳到 int 的类型转换                                | 采用原版，并增加了一些 tqdm 的控制逻辑，在云服务器上运行时可以关闭 tqdm | {name}\_aug.json                                             | {name}.json                                                  |
| add_missing_description as_separate_reports                  | 调用 Bugzilla 的 api，补全缺少描述的错误报告。在横向对比中，或许应该使用未补全的版本 | 无                                                           | 增加了失败重试逻辑，防止网络波动的影响。                     | {name}\_base.json                                            | {name}\_base_with_descriptions.json                          |
| create_ast_cache                                             | ast_cache 是一个字典，key 是 java 文件的 sha，值是该文件的类名、方法名、变量名等，信息拆分成词并计数。<br />report_file_collection 包含了 一次提交中与 ast_cache 相符的 sha，以及文件中所有类名到 sha 的映射，和 sha 到文件名的映射。 | 期待其中包含：<br />"tokenizedMethods"<br />"tokenizedClassName"<br />"tokenizedMethodNames"<br />"tokenizedVariableNames"<br />"tokenizedComments"<br />"rawSourceContent"<br />"methodContent"<br />"classNames"<br />"superclassNames"<br />"interfaceNames"<br />"variableNames" | 增加了注释，说明这两个文件中分别包含哪些信息。               | {name}.json                                                  | {name}\_ast_cache_collection_db<br />{name}\_bug_report_files_collection_db |
| vectorize_ast                                                | 将上一步的两个 db 文件中的每一个 Counter 对象统一放入一个列表，并记录每个 sha 或 bug ID 对应的每个字段在列表中的索引范围 | 将一个函数替换为了其他形式                                   | 保留原版，并增加了一些 tqdm 控制逻辑                         | {name}.json<br />{name}\_ast_cache_collection_db<br />{name}\_bug_report_files_collection_db | {name}\_feature_names_dict: 未使用<br />{name}\_raw_count_data: Counter 列表<br />{name}\_ast_index_collection_index_db：<br />ast_cache 中每一项在 Counter 列表中的范围<br />{name}\_bug_report_index_collection_index_db：<br />bug_report 中每一项在 Counter 列表中的范围<br />{name}\_ast_types_collection_index_db：<br />ast_cache 中的无法 tokenize 的类型信息 |
| vectorize_enriched_api                                       | 对于每一个错误报告，将其中的类信息和方法类型信息对应的 Counter 信息嵌入其中。 | 无                                                           | 增加了一些 tqdm 控制逻辑，增加了进程池大小，取消任务完成时销毁进程的限制 | {name}.json<br />{name}\_raw_count_data<br />{name}\_bug_report_files_collection_db<br />{name}\_ast_index_collection_index_db<br />{name}\_ast_types_collection_index_db<br />{name}\_bug_report_index_collection_index_db | {name}\_{bug_id}\_partial_enriched_api<br />{name}\_{bug_id}\_partial_enriched_api_index_lookup<br />{name}\_{bug_id}\_tfidf_enriched_api |
| convert_tf_idf                                               | 对于每一个错误报告，将其中可向量化的部分转换为 tf_idf        | 无                                                           | 增加了一些 tqdm 控制逻辑，增加了进程池大小，取消任务完成时销毁进程的限制 | {name}.json<br />{name}\_bug_report_files_collection_db<br />{name}\_raw_count_data<br />{name}\_bug_report_index_collection_index_db | {name}\_{bug_id}\_tf_idf_data<br />{name}\_{bug_id}\_tf_idf_index_lookup |
| calculate_feature_3                                          | 提供计算特征 3 需要的数据<br />特征 3 是报告向量与文件向量的余弦相似度<br />这里提供的是报告向量 | 无                                                           | 增加了一些 tqdm 控制逻辑，增加了进程池大小，取消任务完成时销毁进程的限制 | {name}.json<br />{name}\_bug_report_files_collection_db<br />{name}\_raw_count_data<br />{name}\_bug_report_index_collection_index_db | {name}\_{bug_id}\_feature_3_data<br />{name}\_{bug_id}\_feature_3_index_lookup |
| retrieve_features_5_6                                        | 将 `pick_bug_freq` 中获取的两个特征保存为类似的格式          | 无                                                           | 增加了一些 tqdm 控制逻辑，增加了进程池大小，取消任务完成时销毁进程的限制 | {name}.json<br />{name}\_bug_report_files_collection_db      | {name}\_features_5_6_max<br />{name}\_{bug_id}\_features_5_6_data<br />{name}\_{bug_id}\_features_5_6_index_lookup |
| calculate_notes_graph_features                               | 使用 git 提取 java 代码获取的 graph 信息，图相关的特征：入边，出边，PageRank，Authority score，Hub score | 无                                                           | 增加了一些 tqdm 控制逻辑，增加了进程池大小，取消任务完成时销毁进程的限制 | {name}.json<br />{name}\_bug_report_files_collection_db      | {name}\_{bug_id}\_graph_features_data<br />{name}\_{bug_id}\_graph_features_index_lookup |
| calculate_vectorized_features                                | 获取两个列表，前者包含 19 个特征，后者是对应文件的 sha       | 无                                                           | 增加了一些 tqdm 控制逻辑，增加了进程池大小，取消任务完成时销毁进程的限制 | {name}.json<br />{name}\_features_5_6_max<br />{name}\_bug_report_files_collection_db<br />{name}\_{bug_id}\_tf_idf_data<br />{name}\_{bug_id}\_tf_idf_index_lookup<br />{name}\_{bug_id}\_tfidf_enriched_api<br />{name}\_{bug_id}\_partial_enriched_api_index_lookup<br />{name}\_ast_cache_collection_db<br />{name}\_{bug_id}\_feature_3_data<br />{name}\_{bug_id}\_feature_3_index_lookup<br />{name}\_{bug_id}\_graph_features_data<br />{name}\_{bug_id}\_graph_features_index_lookup<br />{name}\_{bug_id}\_features_5_6_data<br />{name}\_{bug_id}\_features_5_6_index_lookup | {name}\_{bug_id}\_features<br />{name}\_{bug_id}\_files      |
| save_normalized_fold_dataframes                              | 按照 500 个一组打包<br/>训练集：<br />保存该报告提交时的相关文件与 200 个不相关文件的特征，优先选择特征 2 更高的不相关文件。<br/>测试集：<br />保存该报告提交时的所有文件的 19 个特征。<br/>记录每种特征的最大最小值，分别归一化。 | 无                                                           | 增加了一些 tqdm 控制逻辑，增加了进程池大小，取消任务完成时销毁进程的限制 | {name}.json<br />{name}\_{bug_id}\_features<br />{name}\_{bug_id}\_files | {name}\_training_fold\_{k}<br />{name}\_normalized_training_fold\_{k}<br />{name}\_testing_fold\_{k}<br />{name}\_normalized_testing_fold\_{k}<br />{name}\_fold_info：存储 fold 个数 |
| load_data_to_joblib_memmap                                   | 使用joblib保存测试集和训练集                                 | 修改了一些并未使用的变量                                     | 无                                                           | {name}\_normalized_training_fold\_{k}<br />{name}\_normalized_testing_fold\_{k}<br />{name}\_fold_info | data_memmap                                                  |
| train_adaptive                                               |                                                              |                                                              |                                                              | data_memmap                                                  |                                                              |
| train_replication                                            | 复现论文：<br />Mapping Bug Reports to Relevant Files: A Ranking Model, a Fine-Grained Benchmark, and Feature Evaluation |                                                              |                                                              |                                                              |                                                              |

## 工具脚本

主要为Python2准备，造了一堆轮子。

| 脚本名称                      | tracking_buggy_files                          | flim                           | me                                       | 使用的地方                                                              |
| ----------------------------- | --------------------------------------------- | ------------------------------ | ---------------------------------------- | ----------------------------------------------------------------------- |
| arg_utils                     | 提供了一些通用的方法                          | 修改了一些微不足道的地方       | 修改了 json 的保存逻辑，增加了 with 语句 | fix_and_augment<br />pick_bug_freq                                      |
| dataset_utils                 | 操作json数据集的方法                          | 无                             | 无                                       | fix_and_augment<br />pick_bug_freq                                      |
| datastore_utils               | 提供json数据结构                              | 无                             | 无                                       | pick_bug_freq                                                           |
| date_utils                    | 提供了日期相关的方法<br />同时兼容python 2和3 | 无                             | 无                                       | 所有需要处理commit时间的地方，将日期转换为时间戳                        |
| git_utils                     | 提供了git相关的方法                           | 试图让它兼容python3            | 无                                       | fix_and_augment<br />pick_bug_freq                                      |
| misc_utils                    | 提供了其他方法                                | 无                             | 无                                       | pick_bug_freq                                                           |
| project_import_graph_features | 提供处理依赖图的方法                          | 无                             | 无                                       | calculate_notes_graph_features<br />calculate_buglocator_graph_features |
| train_utils                   | 提供了载入数据集的方法                        | 修改使其能载入自己添加的数据集 | 无                                       | load_data_to_joblib_memmap<br />train_adaptive                          |
| metrics                       | 提供了计算Accuracy@k, MAP 和 MRR 矩阵的方法   | 无                             | 无                                       | train_adaptive<br />train_replication                                   |

## bug locator相关脚本

| 脚本名称                                                     | tracking_buggy_files                                         | flim                                                         | me                                                           | 输入                                                         | 输出                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| _________________________________________________________________________________________________________________________________________________________ | _____________________________________________________________________________________________ | _____________________________________________________________________________________________ | _____________________________________________________________________________________________ | ___________________________________________________________________________________________________ | ___________________________________________________________________________________________________________________________________________________________________________________________________________________ |
| process_buglocator                                           |                                                              |                                                              |                                                              | EclipseBugRepository.xml                                     | EclipseBugRepository.json                                    |
| tokenize_buglocator_source                                   |                                                              |                                                              |                                                              |                                                              |                                                              |
| vectorize_buglocator_source                                  |                                                              |                                                              |                                                              |                                                              |                                                              |
| vectorize_buglocator_enriched_api                            |                                                              |                                                              |                                                              |                                                              |                                                              |
| calculate_buglocator_time_features                           |                                                              |                                                              |                                                              |                                                              |                                                              |
| calculate_buglocator_feature_3                               |                                                              |                                                              |                                                              |                                                              |                                                              |
| calculate_buglocator_graph_features                          |                                                              |                                                              |                                                              |                                                              |                                                              |
| calculate_buglocator_features                                |                                                              |                                                              |                                                              |                                                              |                                                              |
| save_normalized_fold_dataframes_buglocator                   |                                                              |                                                              |                                                              |                                                              |                                                              |

## flim 获取特征脚本

| 脚本名称                                                     | flim                                                         | 输入                                                         | 输出                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| _________________________________________________________________________________________________________________________ | _____________________________________________________________________________________________________________________ | _________________________________________________________________________________________________________________________________________________ | _______________________________________________________________________________________________________________________________________________________ |
| create_ast_cache                                             | 提取包含原始代码的 ast 信息                                  | {name}.json                                                  | {name}\_ast_cache_collection_db                              |
| generate_function_and_text                                   | 提取报告和代码信息，其中代码的多个维度信息被拆分到多个数据对中。 | {name}.json<br />{name}\_normalized_testing_fold\_{k}<br />{name}\_ast_cache_collection_db | code.jsonl<br />report.jsonl                                 |
| cal_vec                                                      | 载入预训练模型，计算特征                                     | code.jsonl<br />report.jsonl                                 | nl_vecs<br />code_vecs<br />nl_urls<br />code_urls           |
| save_birt                                                    | 将新获取的19个特征嵌入到原来的特征文件中                     | nl_vecs<br />code_vecs<br />nl_urls<br />code_urls           | {name}\_normalized_training_fold\_{k}\_flim<br />{name}\_normalized_testing_fold\_{k}\_flim |
| load_data_to_joblib_memmap                                   | 将上一步获取的文件载入到内存                                 | {name}\_normalized_training_fold\_{k}\_flim<br />{name}\_normalized_testing_fold\_{k}\_flim | data_memmap                                                  |

## 我的脚本

| 脚本名称                                                     | 说明                                                         | 输入                                                         | 输出                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| _________________________________________________________________________________________________________________________ | _______________________________________________________________________________________ | _____________________________________________________________________________________________________________________________________________________________ | _______________________________________________________________________________________________________________________________________________________ |
| save_grapy_fold_dataframes                                   | 将类依赖图的基本单位转换为文件                               | class_info.csv<br />dependency.csv<br />{name}\_normalized_training_fold\_{k}\_flim<br />{name}\_normalized_testing_fold\_{k}\_flim<br />{name}_bug_report_files_collection_db | {name}\_dependency_type_mapping.json<br />{name}\_normalized_training_fold\_{k}\_graph<br />{name}\_normalized_testing_fold\_{k}\_graph<br />{name}\_dependency_training_fold\_{k}\_graph<br />{name}\_dependency_testing_fold\_{k}\_graph |
| load_data_to_joblib_memmap                                   |                                                              | {name}\_normalized_training_fold\_{k}\_graph<br />{name}\_normalized_testing_fold\_{k}\_graph<br />{name}\_dependency_training_fold\_{k}\_graph<br />{name}\_dependency_testing_fold\_{k}\_graph | data_memmap                                                  |
| train_adaptive_GAT                                           |                                                              | data_memmap                                                  |                                                              |

### 数据预处理过程中丢失文件占比

在提取依赖图时，使用的是最终仓库中的代码文件，而错误报告来自不同的时期，有些文件在某次更改中已经永久删除了，有些文件的类路径发生了改变，那些文件路径和类路径均改变的文件将无法跟踪，我们将其从数据集中删除。

文件丢失比例：即在最终的项目仓库中，哪些文件已经被彻底删除了

报告完全丢失：该报告中提到的文件已全部丢失

fold总数k：每500个报告组成一个fold，得到的fold总数（不包括最后一个不完整的fold）

| 仓库名称     | aspectj | swt    | tomcat | birt  | eclipse | jdt    |
| ------------ | ------- | ------ | ------ | ----- | ------- | ------ |
| 文件丢失比例 | 28.21%  | 51.14% | 24.35% | 0.57% | 31.71%  | 29.35% |
| 报告完全丢失 | 0       | 4      | 2      | 10    | 7       | 0      |
| fold总数     | 1       | 8      | 2      | 8     | 12      | 13     |

## 复现结果

### AspectJ 评估数据集

| 算法       | Top1      | Top5      | Top10     | MAP       | MRR       |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| AdaptiveBL | 0.184     | 0.434     | **0.618** | 0.259     | 0.307     |
| Max        | **0.197** | 0.421     | **0.618** | **0.265** | **0.315** |
| Mean       | **0.197** | **0.447** | **0.618** | 0.263     | 0.314     |
| MaxMean    | **0.197** | 0.421     | 0.605     | 0.262     | 0.313     |

### SWT 评估数据集

| 算法       | Top1      | Top5      | Top10     | MAP       | MRR       |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| AdaptiveBL | **0.334** | 0.622     | 0.734     | **0.399** | **0.464** |
| Max        | 0.332     | 0.617     | 0.733     | 0.396     | 0.461     |
| Mean       | 0.328     | 0.624     | **0.737** | **0.399** | 0.462     |
| MaxMean    | 0.330     | **0.626** | 0.736     | **0.399** | 0.462     |

### Tomcat 评估数据集

| 算法       | Top1      | Top5      | Top10     | MAP       | MRR       |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| AdaptiveBL | **0.457** | **0.691** | **0.771** | **0.504** | **0.565** |
| Max        | 0.435     | 0.682     | 0.761     | 0.489     | 0.549     |
| Mean       | 0.446     | 0.675     | 0.762     | 0.493     | 0.554     |
| MaxMean    | 0.435     | 0.667     | 0.755     | 0.486     | 0.547     |

### BIRT 评估数据集

| 算法       | Top1      | Top5      | Top10     | MAP       | MRR       |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| AdaptiveBL | 0.133     | 0.297     | 0.383     | 0.165     | **0.217** |
| Max        | 0.133     | **0.299** | **0.385** | **0.166** | **0.217** |
| Mean       | 0.132     | **0.299** | 0.384     | 0.165     | 0.216     |
| MaxMean    | **0.134** | **0.299** | 0.384     | 0.165     | **0.217** |

### Eclipse Platform UI 评估数据集

| 算法       | Top1      | Top5      | Top10     | MAP       | MRR       |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| AdaptiveBL | **0.400** | **0.642** | 0.720     | **0.436** | **0.509** |
| Max        | 0.396     | 0.635     | **0.721** | 0.433     | 0.507     |
| Mean       | 0.397     | 0.638     | **0.721** | 0.434     | 0.507     |
| MaxMean    | 0.393     | 0.635     | 0.716     | 0.430     | 0.503     |
