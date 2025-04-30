# 仓库配置

本仓库用于复现 tracking buggy files。

## 克隆数据集

以下脚本将创建6个文件夹并分别克隆相应的数据集：

```bash
#!/bin/bash
# 创建文件夹并克隆 aspectj 数据集
mkdir -p aspectj_dataset
git clone https://bitbucket.org/mfejzer/tracking_buggy_files_aspectj_dataset/ aspectj_dataset

# 创建文件夹并克隆 birt 数据集
mkdir -p birt_dataset
git clone https://bitbucket.org/mfejzer/tracking_buggy_files_birt_dataset/ birt_dataset

# 创建文件夹并克隆 eclipse_platform_ui 数据集
mkdir -p eclipse_platform_ui_dataset
git clone https://bitbucket.org/mfejzer/tracking_buggy_files_eclipse_platform_ui_dataset/ eclipse_platform_ui_dataset

# 创建文件夹并克隆 jdt 数据集
mkdir -p jdt_dataset
git clone https://bitbucket.org/mfejzer/tracking_buggy_files_jdt_dataset/ jdt_dataset

# 创建文件夹并克隆 swt 数据集
mkdir -p swt_dataset
git clone https://bitbucket.org/mfejzer/tracking_buggy_files_swt_dataset/ swt_dataset

# 创建文件夹并克隆 tomcat 数据集
mkdir -p tomcat_dataset
git clone https://bitbucket.org/mfejzer/tracking_buggy_files_tomcat_dataset/ tomcat_dataset
```

## 提取Java根路径

由于这些仓库都使用模块化开发，不同模块的Java文件有着不同的根路径。我们可以使用 `getRootPath.py` 脚本来提取每个Java文件对应的项目根路径，并处理冲突。

使用 `getRootPath.py` 脚本提取每个数据集中的Java文件根路径，结果将保存在 `output` 文件夹下的对应文件夹中。

`--check-conflicts-only` 参数用于检查冲突，将显示相同类路径的Java文件及其对应的Java项目根路径，以及如果移除这个Java项目根路径会损失的Java文件数，保存在 `output` 文件夹下的 `conflict.txt` 文件中。

`--priority` 参数用于在处理冲突时指定优先级，当有多个Java文件对应同一个Java项目根路径时，将使用优先级最高的Java项目根路径。

`--ignore-conflicts` 参数用于忽略冲突，将获取所有路径。

获取的Java文件根路径将保存在 `output` 文件夹下的 `javaRoots.txt` 文件中。

```bash
python getRootPath.py source/aspectj_dataset --check-conflicts-only
python getRootPath.py source/birt_dataset --check-conflicts-only
python getRootPath.py source/eclipse_platform_ui_dataset --check-conflicts-only
python getRootPath.py source/jdt_dataset --check-conflicts-only
python getRootPath.py source/swt_dataset --check-conflicts-only
python getRootPath.py source/tomcat_dataset --check-conflicts-only
```

```bash
python getRootPath.py source/aspectj_dataset --ignore-conflicts
python getRootPath.py source/birt_dataset --ignore-conflicts
python getRootPath.py source/eclipse_platform_ui_dataset --ignore-conflicts
python getRootPath.py source/jdt_dataset --ignore-conflicts
python getRootPath.py source/swt_dataset --ignore-conflicts
python getRootPath.py source/tomcat_dataset --ignore-conflicts
```

```bash
python getRootPath.py source/aspectj_dataset --priority win32 mozilla common_j2se
python getRootPath.py source/birt_dataset --priority win32 mozilla common_j2se
python getRootPath.py source/eclipse_platform_ui_dataset --priority win32 mozilla common_j2se
python getRootPath.py source/jdt_dataset --priority win32 mozilla common_j2se
python getRootPath.py source/swt_dataset --priority win32 mozilla common_j2se
python getRootPath.py source/tomcat_dataset --priority win32 mozilla common_j2se
```

## 提取并处理依赖关系

使用我们自己编写的 java_ast_parser 提取每个数据集中的Java文件依赖关系，结果将保存在 `dataset` 文件夹下的对应文件夹中。

| 数据集              | 类节点数 | 类间依赖数 |
| ------------------- | -------- | ---------- |
| aspectj             | 9,733    | 112,505    |
| birt                | 11,440   | 202,969    |
| eclipse_platform_ui | 7,490    | 87,925     |
| jdt                 | 4,817    | 57,859     |
| swt                 | 2,003    | 101,415    |
| tomcat              | 3,360    | 33,842     |

依赖类型分布如下所示：

![aspectj_dataset](dataset/aspectj_dataset/dependency_distribution.png)

![birt_dataset](dataset/birt_dataset/dependency_distribution.png)

![eclipse_platform_ui_dataset](dataset/eclipse_platform_ui_dataset/dependency_distribution.png)

![jdt_dataset](dataset/jdt_dataset/dependency_distribution.png)

![swt_dataset](dataset/swt_dataset/dependency_distribution.png)

![tomcat_dataset](dataset/tomcat_dataset/dependency_distribution.png)

虽然我们设计了 13 种依赖类型，但是，我们发现这 6 个数据集中都没有出现 “泛型约束依赖”，也就是说，没有出现形如 `A<T extends B>` 的依赖关系。此外，SWT数据集没有出现“泛型参数依赖”，可能是因为其使用的Java版本还未引入泛型。
