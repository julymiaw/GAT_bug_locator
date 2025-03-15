from train_adaptive_GAT import (
    GATModule,
    node_feature_columns,
    edge_feature_columns,
)


def count_parameters(model):
    """计算模型的参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total}")
    print(f"可训练参数量: {trainable}")

    # 打印各组件参数量
    print("\n各组件参数详情:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}")

    return total, trainable


def create_test_models():
    """创建不同配置的模型进行参数量比较"""
    # 参数配置
    node_dim = len(node_feature_columns)  # 19
    edge_dim = len(edge_feature_columns)  # 12
    hidden_dim = 16

    # 创建不同头数配置的模型
    models = {
        "无GAT层 (MLP)": GATModule(
            node_dim, edge_dim, hidden_dim, heads=None, dropout=0.1
        ),
        "单层GAT (2头)": GATModule(
            node_dim, edge_dim, hidden_dim, heads=2, dropout=0.1
        ),
    }

    # 比较各配置参数量
    results = {}
    for name, model in models.items():
        print(f"\n===== {name} =====")
        _, trainable = count_parameters(model)
        results[name] = trainable

    return results


if __name__ == "__main__":
    print("GAT模型参数量分析")
    results = create_test_models()

    print("\n===== 参数量比较 =====")
    for name, params in results.items():
        print(f"{name}: {params:,}参数")
