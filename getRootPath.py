import os
import re
from collections import defaultdict


def find_all_java_files(directory):
    java_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    return java_files


def extract_package_from_java_file(java_file):
    package_pattern = re.compile(r"^\s*package\s+([\w\.]+)\s*;", re.MULTILINE)
    try:
        with open(java_file, "r", encoding="utf-8") as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(java_file, "r", encoding="latin-1") as file:
            content = file.read()
    match = package_pattern.search(content)
    if match:
        return match.group(1)
    return None


def find_root_path_for_java_file(java_file, package):
    if package is None:
        return None
    package_path = package.replace(".", os.sep)
    java_file_dir = os.path.dirname(java_file)
    root_path = java_file_dir
    while package_path and root_path.endswith(package_path):
        root_path = os.path.dirname(root_path)
        package_path = os.path.dirname(package_path)
    return root_path


def find_all_java_roots(directory):
    java_files = find_all_java_files(directory)
    java_roots = defaultdict(set)
    for java_file in java_files:
        package = extract_package_from_java_file(java_file)
        root_path = find_root_path_for_java_file(java_file, package)
        if root_path:
            package_and_file = f"{package}.{os.path.basename(java_file)}"
            java_roots[package_and_file].add(root_path)
    return java_roots


def calculate_unique_classes(java_roots):
    unique_classes = defaultdict(int)
    for _, roots in java_roots.items():
        if len(roots) == 1:
            unique_classes[list(roots)[0]] += 1
    return unique_classes


def filter_conflicting_roots(java_roots, unique_classes, priority):
    all_roots = set()
    remained_roots = set()
    deleted_roots = set()

    for _, roots in java_roots.items():
        roots = list(roots)
        all_roots.update(roots)
        if len(roots) == 2:
            root1, root2 = roots
            loss1 = unique_classes.get(root1, 0)
            loss2 = unique_classes.get(root2, 0)
            if loss1 == 0 and loss2 > 0:
                remained_roots.add(root2)
                deleted_roots.add(root1)
            elif loss2 == 0 and loss1 > 0:
                remained_roots.add(root1)
                deleted_roots.add(root2)
        elif len(roots) > 2:
            # 计算不同部分
            diffs = [set(root.split(os.sep)) for root in roots]
            matched = False
            for p in priority:
                for i, diff in enumerate(diffs):
                    if p in diff:
                        remained_roots.add(roots[i])
                        deleted_roots.update(roots[:i] + roots[i + 1 :])
                        matched = True
                        break
                if matched:
                    break

    if remained_roots & deleted_roots:
        raise Exception("需要删除的路径与保留的路径有交集")

    return all_roots - deleted_roots


def main(directory, check_conflicts_only, priority, ignore_conflicts):
    java_roots = find_all_java_roots(directory)
    unique_classes = calculate_unique_classes(java_roots)

    output_dir = os.path.join("output", os.path.basename(directory))
    os.makedirs(output_dir, exist_ok=True)

    if check_conflicts_only:
        conflict_file = os.path.join(output_dir, "conflict.txt")
        roots_to_classes = defaultdict(list)
        roots_to_loss = {}

        # 先扫描一遍数据，将具有相同根路径集合的类文件合并
        for package_and_file, roots in java_roots.items():
            roots_tuple = tuple(sorted(roots))
            roots_to_classes[roots_tuple].append(package_and_file)
            for root in roots:
                if root not in roots_to_loss:
                    roots_to_loss[root] = unique_classes.get(root, 0)

        # 输出合并后的结果
        with open(conflict_file, "w") as f:
            for roots, classes in roots_to_classes.items():
                if len(roots) > 1:  # 只输出根路径长度大于1的类文件
                    for class_file in classes:
                        f.write(f"Class: {class_file}\n")
                    for root in roots:
                        loss = roots_to_loss[root]
                        f.write(f"  Root: {root}, Loss: {loss}\n")
                    f.write("\n")
        return

    if not ignore_conflicts:
        java_roots = filter_conflicting_roots(java_roots, unique_classes, priority)

    with open(os.path.join(output_dir, "javaRoots.txt"), "w") as f:
        for path in java_roots:
            f.write(f"{path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="处理Java文件根路径")
    parser.add_argument("directory", help="要处理的目录")
    parser.add_argument(
        "--check-conflicts-only", action="store_true", help="只检查冲突"
    )
    parser.add_argument(
        "--priority",
        nargs="+",
        help="冲突时的优先级",
    )
    parser.add_argument(
        "--ignore-conflicts", action="store_true", help="忽略冲突，输出所有根路径"
    )

    args = parser.parse_args()
    main(
        args.directory, args.check_conflicts_only, args.priority, args.ignore_conflicts
    )
