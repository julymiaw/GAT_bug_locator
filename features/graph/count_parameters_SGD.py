from sklearn.linear_model import SGDRegressor
import numpy as np
from train_adaptive_SGD import node_feature_columns


def count_parameters_sgd(model):
    """计算SGDRegressor模型的参数量"""
    total = sum(p.size for p in model.coef_)
    trainable = total + model.intercept_.size

    print(f"总参数量: {total + model.intercept_.size}")
    print(f"可训练参数量: {trainable}")

    # 打印各组件参数量
    print("\n各组件参数详情:")
    print(f"coef_: {model.coef_.size}")
    print(f"intercept_: {model.intercept_.size}")

    return total + model.intercept_.size, trainable


def create_sgd_model():
    """创建SGDRegressor模型并计算参数量"""
    # 参数配置
    input_dim = len(node_feature_columns)  # 输入特征维度

    # 创建SGDRegressor模型
    model = SGDRegressor(
        max_iter=1000, shuffle=False, loss="squared_error", penalty="l2", alpha=0.0001
    )

    # 模拟训练以初始化参数
    X_dummy = np.random.rand(10, input_dim)
    y_dummy = np.random.rand(10)
    model.fit(X_dummy, y_dummy)

    # 计算参数量
    count_parameters_sgd(model)


if __name__ == "__main__":
    print("SGDRegressor模型参数量分析")
    create_sgd_model()
