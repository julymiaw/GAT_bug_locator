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

### 使用步骤

1. 克隆数据集

   首先，使用以下脚本克隆所有数据集：

   ```bash
   #!/bin/bash
   mkdir source && cd source

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

2. 提取Java根路径

   使用 `getRootPath.py` 脚本提取每个数据集中的Java文件根路径。以下是使用该脚本的步骤：

   ```bash
   # 进入脚本所在目录
   cd /home/july/srtp/tracking_buggy_files

   # 提取 aspectj 数据集的Java根路径
   python getRootPath.py source/aspectj_dataset --priority win32 mozilla common_j2se

   # 提取 birt 数据集的Java根路径
   python getRootPath.py source/birt_dataset --priority win32 mozilla common_j2se

   # 提取 eclipse_platform_ui 数据集的Java根路径
   python getRootPath.py source/eclipse_platform_ui_dataset --priority win32 mozilla common_j2se

   # 提取 jdt 数据集的Java根路径
   python getRootPath.py source/jdt_dataset --priority win32 mozilla common_j2se

   # 提取 swt 数据集的Java根路径
   python getRootPath.py source/swt_dataset --priority win32 mozilla common_j2se

   # 提取 tomcat 数据集的Java根路径
   python getRootPath.py source/tomcat_dataset --priority win32 mozilla common_j2se
   ```

   每次运行脚本后，结果将输出到当前目录下的 `output_<处理文件夹名>` 文件夹中，例如 `output_aspectj_dataset`。

3. 检查冲突

   如果只需要检查冲突，可以使用 `--check-conflicts-only` 参数：

   ```bash
   python getRootPath.py source/aspectj_dataset --check-conflicts-only
   python getRootPath.py source/birt_dataset --check-conflicts-only
   python getRootPath.py source/eclipse_platform_ui_dataset --check-conflicts-only
   python getRootPath.py source/jdt_dataset --check-conflicts-only
   python getRootPath.py source/swt_dataset --check-conflicts-only
   python getRootPath.py source/tomcat_dataset --check-conflicts-only
   ```

   这样可以查看相同类路径的Java文件及其对应的Java项目根路径，以及如果移除这个Java项目根路径会损失的Java文件数。

   如果不考虑冲突，获取所有路径，可以使用 `--ignore-conflicts` 参数：

   ```bash
   python getRootPath.py source/aspectj_dataset --ignore-conflicts
   python getRootPath.py source/birt_dataset --ignore-conflicts
   python getRootPath.py source/eclipse_platform_ui_dataset --ignore-conflicts
   python getRootPath.py source/jdt_dataset --ignore-conflicts
   python getRootPath.py source/swt_dataset --ignore-conflicts
   python getRootPath.py source/tomcat_dataset --ignore-conflicts
   ```
