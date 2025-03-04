import csv
import pickle
from unqlite import UnQLite
import sys

# 增加字段大小限制
csv.field_size_limit(sys.maxsize)


def read_bug_report_files(data_prefix):
    db_path = f"{data_prefix}/{data_prefix}_bug_report_files_collection_db"
    bug_report_files_collection_db = UnQLite(db_path, flags=0x00000100 | 0x00000001)

    class_name_to_sha = {}
    sha_to_file_name = {}
    for k in bug_report_files_collection_db.keys():
        data = pickle.loads(bug_report_files_collection_db[k])
        class_name_to_sha.update(data.get("class_name_to_sha", {}))
        sha_to_file_name.update(data.get("sha_to_file_name", {}))

    return class_name_to_sha, sha_to_file_name


def load_class_info(data_prefix):
    class_info_path = f"../dataset/{data_prefix}_dataset/class_info.csv"
    class_info = {}
    with open(class_info_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_info[row["className"]] = row["filePath"]
    return class_info


def main(data_prefix):
    class_name_to_sha, sha_to_file_name = read_bug_report_files(data_prefix)
    class_info = load_class_info(data_prefix)

    # 统计sha_to_file_name中提到的文件路径总数
    total_sha_to_file_name_paths = len(set(sha_to_file_name.values()))
    print(
        f"Total file paths in {data_prefix} sha_to_file_name: {total_sha_to_file_name_paths}"
    )

    # 统计class_info中包含的信息总数
    total_class_info = len(set(class_info.values()))
    print(f"Total file paths in {data_prefix} class_info: {total_class_info}")

    # 统计位于class_info中的文件路径未找到对应sha的总数
    missing_file_paths_in_class_info = set()
    for class_name, file_path in class_info.items():
        if file_path not in sha_to_file_name.values():
            missing_file_paths_in_class_info.add(file_path)

    print(
        f"File paths in {data_prefix} class_info without corresponding SHA: {len(missing_file_paths_in_class_info)}"
    )

    # 统计位于sha_to_file_name但不在class_info中的文件路径总数
    class_info_file_paths = set(class_info.values())
    missing_file_paths = set()
    for sha, file_path in sha_to_file_name.items():
        if file_path not in class_info_file_paths:
            missing_file_paths.add(file_path)

    print(
        f"File paths in {data_prefix} sha_to_file_name not in class_info: {len(missing_file_paths)}"
    )


# 数据集列表
data_prefixes = [
    "aspectj",
    # "swt",
    # "jdt",
    # "tomcat",
    # "birt",
    # "eclipse_platform_ui"
]

# 循环处理每个数据集
for data_prefix in data_prefixes:
    main(data_prefix)
