import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 设置matlab字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载JSON数据
with open("aspectj_Adaptive_regression_log_20250303024219.json", "r") as f:
    data = json.load(f)

# 解析模型名称和得分
models = []
for item in data:
    model_name = item[0]
    score = item[2]

    # 解析模型名称
    parts = model_name.split("_")
    loss_type = parts[1]  # MSE 或 Huber
    hidden_dim = int(parts[2])  # 隐藏层维度

    # 解析正则化方法
    if parts[3] in ["none", "l2", "l1", "elasticnet"]:
        penalty = parts[3]
        idx = 4
    else:
        penalty = "none"  # 默认值
        idx = 3

    # 解析注意力头数配置
    heads = []
    while idx < len(parts) and not parts[idx].startswith("lr"):
        heads.append(parts[idx])
        idx += 1

    heads_str = "_".join(heads)
    num_layers = len(heads)

    # 解析学习率
    lr_part = next((p for p in parts if p.startswith("lr")), None)
    lr = float(lr_part.replace("lr", "")) if lr_part else None

    models.append(
        {
            "model": model_name,
            "loss_type": loss_type,
            "hidden_dim": hidden_dim,
            "penalty": penalty,
            "heads": heads_str,
            "num_layers": num_layers,
            "lr": lr,
            "score": score,
        }
    )

# 创建DataFrame
df = pd.DataFrame(models)


# 分析各参数对性能的影响
def analyze_parameter(df, param_name):
    param_stats = (
        df.groupby(param_name)["score"]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values("mean")
    )
    best_option = param_stats.iloc[-1].name
    worst_option = param_stats.iloc[0].name

    print(f"\n--- {param_name} 分析 ---")
    print(
        f"最佳选项: {best_option} (平均得分: {param_stats.loc[best_option, 'mean']:.4f})"
    )
    print(
        f"最差选项: {worst_option} (平均得分: {param_stats.loc[worst_option, 'mean']:.4f})"
    )
    print(f"各选项平均性能:")
    print(param_stats[["mean", "count"]])

    return param_stats


# 分析每个参数
loss_stats = analyze_parameter(df, "loss_type")
dim_stats = analyze_parameter(df, "hidden_dim")
penalty_stats = analyze_parameter(df, "penalty")
lr_stats = analyze_parameter(df, "lr")
heads_stats = analyze_parameter(df, "heads")
layers_stats = analyze_parameter(df, "num_layers")

# 深入分析：各参数交叉影响
print("\n--- 交叉影响分析 ---")
# 损失函数 × 学习率
loss_lr_stats = df.groupby(["loss_type", "lr"])["score"].mean().unstack()
print("\n损失函数 × 学习率:")
print(loss_lr_stats)

# 正则化 × 学习率
penalty_lr_stats = df.groupby(["penalty", "lr"])["score"].mean().unstack()
print("\n正则化方法 × 学习率:")
print(penalty_lr_stats)

# 隐藏层维度 × 正则化
dim_penalty_stats = df.groupby(["hidden_dim", "penalty"])["score"].mean().unstack()
print("\n隐藏层维度 × 正则化方法:")
print(dim_penalty_stats)

# 可视化
plt.figure(figsize=(20, 15))

# 1. 各参数的平均性能
plt.subplot(3, 2, 1)
sns.barplot(x=loss_stats.index, y=loss_stats["mean"])
plt.title("损失函数影响")
plt.ylabel("平均得分")

plt.subplot(3, 2, 2)
sns.barplot(x=dim_stats.index, y=dim_stats["mean"])
plt.title("隐藏层维度影响")
plt.ylabel("平均得分")

plt.subplot(3, 2, 3)
sns.barplot(x=penalty_stats.index, y=penalty_stats["mean"])
plt.title("正则化方法影响")
plt.ylabel("平均得分")

plt.subplot(3, 2, 4)
sns.barplot(x=lr_stats.index, y=lr_stats["mean"])
plt.title("学习率影响")
plt.ylabel("平均得分")

plt.subplot(3, 2, 5)
sns.boxplot(x="penalty", y="score", data=df)
plt.title("正则化方法性能分布")
plt.ylabel("得分")

plt.subplot(3, 2, 6)
sns.boxplot(x="lr", y="score", data=df)
plt.title("学习率性能分布")
plt.ylabel("得分")

plt.tight_layout()
plt.savefig("model_parameter_analysis.png")
plt.show()

# 识别表现最差的参数组合
worst_combinations = (
    df.groupby(["loss_type", "penalty", "lr"])["score"].mean().sort_values().head(5)
)
print("\n表现最差的参数组合:")
print(worst_combinations)


# 找出可以安全省略的参数选项
def find_consistently_poor_options(df, param_name):
    """查找在各种条件下都表现不佳的参数选项"""
    # 获取该参数的所有可能值
    param_values = df[param_name].unique()

    consistently_poor = []
    for value in param_values:
        # 对每个其他参数组合，检查该值是否总是表现不佳
        is_poor = True

        # 获取所有其他参数的组合
        other_params = [
            p for p in ["loss_type", "hidden_dim", "penalty", "lr"] if p != param_name
        ]

        for combo in df.groupby(other_params).groups:
            subset = df.loc[df.groupby(other_params).groups[combo]]
            param_scores = subset.groupby(param_name)["score"].mean()

            # 如果该值不是最差的，则不是consistently poor
            if value != param_scores.idxmin():
                is_poor = False
                break

        if is_poor:
            consistently_poor.append(value)

    return consistently_poor


# 检查各参数是否有consistently poor的选项
print("\n可以安全省略的参数选项:")
for param in ["loss_type", "hidden_dim", "penalty", "lr"]:
    poor_options = find_consistently_poor_options(df, param)
    if poor_options:
        print(f"{param}中可以省略的选项: {poor_options}")
    else:
        print(f"{param}中没有可以省略的选项")
