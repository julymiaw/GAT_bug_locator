import pandas as pd
import torch

# 定义依赖类型的映射
dependency_types = [
    "内部类依赖",
    "继承依赖",
    "接口依赖",
    "字段类型依赖",
    "方法参数依赖",
    "方法返回依赖",
    "异常类型依赖",
    "局部变量依赖",
    "对象实例化依赖",
    "类型转换依赖",
    "静态引用依赖",
    "泛型参数依赖",
    # "泛型约束依赖",
]

dependency_type_to_idx = {
    dep_type: idx for idx, dep_type in enumerate(dependency_types)
}


def preprocess_dependency(file_path, output_path):
    # 读取依赖关系数据
    df = pd.read_csv(file_path)
    class_idx_dict = torch.load("model/dataset/class_idx_dict.pt", weights_only=True)

    # 初始化存储字典
    dependency_dict = {}

    for _, row in df.iterrows():
        source = class_idx_dict[row["source"]]
        target = class_idx_dict[row["target"]]
        dep_type = row["type"]

        # 创建依赖向量
        if (source, target) not in dependency_dict:
            dependency_dict[(source, target)] = [0] * len(dependency_types)

        if dep_type in dependency_type_to_idx:
            dependency_dict[(source, target)][dependency_type_to_idx[dep_type]] += 1

    # 转换为 PyTorch 张量
    sources = []
    targets = []
    dependency_vectors = []

    for (source, target), dep_vector in dependency_dict.items():
        sources.append(source)
        targets.append(target)
        dependency_vectors.append(dep_vector)

    sources_tensor = torch.tensor(sources, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    dependency_vectors_tensor = torch.tensor(dependency_vectors, dtype=torch.float)

    # 保存为 .pt 文件
    torch.save((sources_tensor, targets_tensor, dependency_vectors_tensor), output_path)


if __name__ == "__main__":
    input_file_path = "dependency.csv"
    output_file_path = "model/dataset/dependency.pt"
    preprocess_dependency(input_file_path, output_file_path)
