import argparse
from html import parser
from skopt import load

from metrics import print_metrics

from train_adaptive_GAT import Adaptive_Process, GATRegressor, process

node_feature_columns = ["f" + str(i) for i in range(1, 20)]
edge_feature_columns = ["t" + str(i) for i in range(1, 13)]


def get_params(model_name):
    params = {}
    parts = model_name.split("_")

    # 基本信息
    params["model_type"] = parts[0]
    params["loss_type"] = parts[1]
    params["hidden_dim"] = int(parts[2])
    params["penalty"] = parts[3]
    params["heads"] = int(parts[4]) if parts[4] != "nohead" else None

    # 解析其他超参数
    i = 5
    while i < len(parts):
        if parts[i].startswith("a"):
            params["alpha"] = float(parts[i].replace("a", ""))
        elif parts[i].startswith("dr"):
            params["dropout"] = float(parts[i].replace("dr", ""))
        elif parts[i].startswith("lr"):
            params["learning_rate"] = float(parts[i].replace("lr", ""))
        elif parts[i] == "selfloop":
            params["use_self_loops"] = True
        elif parts[i] == "shuf":
            params["shuffle"] = True
        i += 1

    # 设置默认值
    params.setdefault("use_self_loops", False)
    params.setdefault("shuffle", False)
    return params


def main():
    # parser = argparse.ArgumentParser(description="Train Adaptive Process")
    # parser.add_argument("file_prefix", type=str, help="Feature files prefix")
    # parser.add_argument("model_name", type=str, help="")
    # parser.add_argument("--max", action="store_true", help="Include feature 37")
    # parser.add_argument("--mean", action="store_true", help="Include feature 38")
    # args = parser.parse_args()

    args = argparse.Namespace(
        file_prefix="aspectj",
        model_name="GATRegressor_WeightedMSE_16_l2_2_a0.0001_dr0.3_lr0.005",
        max=False,
        mean=False,
    )

    # 根据参数决定是否添加特征37和38
    if args.max:
        node_feature_columns.append("f37")
    if args.mean:
        node_feature_columns.append("f38")

    file_prefix = args.file_prefix

    (
        fold_number,
        fold_testing,
        fold_training,
        fold_dependency_testing,
        fold_dependency_training,
    ) = load(f"../joblib_memmap_{file_prefix}_graph/data_memmap", mmap_mode="r")

    params = get_params(args.model_name)
    hd = params["hidden_dim"]
    h = params["heads"]
    loop = params["use_self_loops"]
    ls = params["loss_type"]
    lr = params["learning_rate"]
    p = params["penalty"]
    dr = params["dropout"]
    a = params["alpha"]

    model = Adaptive_Process()
    # model.use_training_cross_validation = False
    model.reg_models = [
        GATRegressor(
            node_feature_columns.copy(),
            edge_feature_columns.copy(),
            hidden_dim=hd,
            heads=h,
            shuffle=False,
            use_self_loops_only=loop,
            loss=ls,
            lr=lr,
            penalty=p,
            dropout=dr,
            alpha=a,
        )
    ]

    result = process(
        model,
        fold_number,
        fold_testing,
        fold_training,
        fold_dependency_testing,
        fold_dependency_training,
        file_prefix,
    )

    print("======Results======")
    print("name ", result["name"])
    print_metrics(*result["results"])


if __name__ == "__main__":
    main()
