import os
import csv
import sys
from tqdm import tqdm

# 增加 CSV 模块的字段大小限制
csv.field_size_limit(sys.maxsize)


def process_dataset(dataset_name):
    input_file = f"output/{dataset_name}/javaRoots.txt"
    output_dir = f"dataset/{dataset_name}"

    # 读取javaRoots.txt文件中的路径
    with open(input_file, "r") as file:
        paths = [line.strip() for line in file.readlines()]

    # 替换路径中的"sources"为"output"
    output_paths = [path.replace("source", "output") for path in paths]

    all_classes = set()
    class_info_rows = []
    dependency_rows = []

    header = None

    # 遍历所有路径
    for path in tqdm(output_paths, desc=f"Processing paths for {dataset_name}"):
        class_info_path = os.path.join(path, "class_info.csv")

        # 读取class_info.csv中的类信息
        if os.path.exists(class_info_path):
            with open(class_info_path, "r") as class_info_file:
                reader = csv.reader(class_info_file)
                header = next(reader)  # 读取表头
                for row in reader:
                    class_info_rows.append(row)
                    all_classes.add(row[0])

    # 将所有class_info汇总到一个文件中
    with open(
        os.path.join(output_dir, "class_info.csv"), "w", newline=""
    ) as class_info_file:
        writer = csv.writer(class_info_file)
        writer.writerow(header)  # 写入表头
        writer.writerows(class_info_rows)

    # 遍历所有路径，检查依赖文件的source和target
    for path in tqdm(output_paths, desc=f"Processing dependencies for {dataset_name}"):
        dependency_folder_path = os.path.join(path, "dependency")

        if os.path.exists(dependency_folder_path):
            for dependency_file in os.listdir(dependency_folder_path):
                dependency_file_path = os.path.join(
                    dependency_folder_path, dependency_file
                )
                if os.path.isfile(
                    dependency_file_path
                ) and dependency_file_path.endswith(".csv"):
                    with open(dependency_file_path, "r") as dependency_file:
                        reader = csv.reader(dependency_file)
                        header = next(reader)  # 读取表头
                        for row in reader:
                            source, target = row[0], row[1]
                            if source in all_classes and target in all_classes:
                                dependency_rows.append(row)

    # 将所有有效的依赖关系汇总到一个文件中
    with open(
        os.path.join(output_dir, "dependency.csv"), "w", newline=""
    ) as dependency_file:
        writer = csv.writer(dependency_file)
        writer.writerow(header)  # 写入表头
        writer.writerows(dependency_rows)

    print(f"Total number of classes for {dataset_name}: {len(class_info_rows)}")
    print(f"Total number of dependencies for {dataset_name}: {len(dependency_rows)}")


# 处理所有数据集
datasets = [
    "aspectj_dataset",
    "birt_dataset",
    "eclipse_platform_ui_dataset",
    "jdt_dataset",
    "swt_dataset",
    "tomcat_dataset",
]
for dataset in datasets:
    process_dataset(dataset)
