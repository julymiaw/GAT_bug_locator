#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <feature_files_prefix>

Requires results of save_normalized_fold_dataframes.py
"""

import json
import time
import argparse
from collections import defaultdict
from itertools import product
from timeit import default_timer

import gc
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import kruskal, ttest_ind, levene
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.utils import safe_mask
from skopt import load

from metrics import calculate_metric_results, print_metrics
from train_utils import eprint

node_feature_columns = ["f" + str(i) for i in range(1, 20)]


class Adaptive_Process(object):
    """
    自适应缺陷定位算法核心类，通过动态选择特征权重计算、回归模型和特征筛选策略优化结果。
    实现多阶段交叉验证和并行计算以提升效率，支持模型持久化和性能日志记录。

    主要流程：
        1. 权重计算阶段：通过多种统计/机器学习方法评估特征重要性
        2. 模型选择阶段：遍历（评分方法 × 回归模型 × 特征筛选策略）组合寻找最优配置
        3. 预测阶段：根据训练结果选择线性权重组合或回归模型进行预测

    设计特性：
        - 支持增量学习：通过enforce_relearning控制是否复用历史模型
        - 双层交叉验证：在权重计算和模型选择阶段分别使用独立交叉验证
        - 并行计算：使用joblib并行化权重计算和模型评估过程
    """

    def __init__(self):
        """初始化方法，配置所有可用算法组件"""
        # region 算法组件配置
        # 特征权重计算方法列表（统计检验/树模型/降维方法等）
        self.weights_methods = [
            weights_AdaBoostClassifier,
            weights_ExtraTreesClassifier,
            weights_GradientBoostingClassifier,
            weights_const,
            weights_variance,
            weights_chi2,
            weights_mutual_info_classif,
            weights_FastICA,
            weights_kruskal_classif,
            weights_ttest_ind_classif,
            weights_levene_median,
            weights_mean_var,
            weights_maximum_absolute_deviation,
        ]

        # 回归模型集合（包含多种SGDRegressor配置）
        self.reg_models: List[SGDRegressor] = []
        self.reg_models.extend(get_skmodels())  # 生成SGD变体

        # 特征筛选策略（按修复样本百分比截断）
        self.cut_methods: List[Callable[[pd.DataFrame, np.ndarray], pd.Series]] = [
            size_selectf_only_fixes_p_perc_05,
            size_selectf_only_fixes_p_perc_10,
            size_selectf_only_fixes_p_perc_15,
            size_selectf_only_fixes_p_perc_20,
            size_selectf_only_fixes_p_perc_25,
            size_selectf_only_fixes_p_perc_30,
        ]

        # 评分方法（目前仅标准评分）
        self.score_methods: List[
            Callable[[pd.DataFrame, List[str], np.ndarray], np.ndarray]
        ] = [normal_score]
        # endregion

        # region 运行时状态存储
        self.weights = None  # 当前最佳特征权重向量（np.ndarray）
        self.weights_score = 0  # 当前最佳权重方法得分
        self.reg_model = None  # 当前选择的回归模型对象
        self.reg_model_score = 0  # 当前最佳回归模型得分
        self.cut_method: Callable[[pd.DataFrame, np.ndarray], pd.Series] = (
            None  # 当前选择的特征筛选方法（函数引用）
        )
        self.score_method = None  # 当前选择的评分方法（函数引用）
        # endregion

        # region 配置参数
        self.name = "Adaptive"  # 算法标识名
        self.first_fold_processed = False  # 首折处理标志
        self.enforce_relearning = True  # 强制重新学习开关
        self.use_prescoring_always = False  # 是否始终使用预评分权重
        self.use_reg_model_always = True  # 是否强制使用回归模型
        self.use_prescoring_cross_validation = True  # 权重计算阶段交叉验证开关
        self.use_training_cross_validation = True  # 模型选择阶段交叉验证开关
        self.cross_validation_fold_number = 2  # 交叉验证折数
        # endregion

        # region 性能日志
        self.training_time_list = []  # 各折训练耗时：(总时间, 缺陷报告数, 文件数)
        self.prescoring_log = []  # 权重方法评估：(方法名, (权重向量, 评估得分))
        self.best_prescoring_log = []  # 各折最佳权重： (方法名, 评估得分)
        self.regression_log = []  # 回归模型评估：(模型名, 筛选方法名, 评分方法名, 得分)
        self.best_regression_log = []  # 各折最佳模型，格式同regression_log
        # endregion

        # region 配置映射表（内部使用）
        self.weights_methods_map = {m.__name__: m for m in self.weights_methods}
        self.reg_models_map = {str(m): m for m in self.reg_models}
        self.cut_methods_map = {m.__name__: m for m in self.cut_methods}
        self.score_methods_map = {m.__name__: m for m in self.score_methods}
        # endregion

    def compute_weights(self, df: pd.DataFrame, columns: List[str]):
        """
        特征权重计算阶段（并行交叉验证）

        参数：
            df: 训练数据集，包含特征和used_in_fix标签
            columns: 特征列名列表

        流程：
            1. 使用KFold拆分数据为预设折数
            2. 对每折训练集并行计算所有权重方法的权重向量
            3. 在验证集评估各权重方法的预测效果（MAP指标）
            4. 聚合各方法在所有折的平均表现

        结果存储：
            self.weights更新为最佳方法的平均权重向量
            self.prescoring_log记录所有方法评估结果
            self.best_prescoring_log记录最佳方法信息
        """
        if self.use_prescoring_cross_validation:
            # 使用 k 折交叉验证计算权重
            kfold = KFold(
                n_splits=self.cross_validation_fold_number,
                random_state=None,
                shuffle=False,
            )
            # 保存每种方法在每折的结果，包括权重向量和 MAP 得分
            partial_result_dict: Dict[str, List[Tuple[np.ndarray, float]]] = (
                defaultdict(list)
            )
            for train_index, test_index in kfold.split(df):
                kdf = df.iloc[train_index]
                weights: List[Tuple[str, np.ndarray]] = Parallel(n_jobs=-1)(
                    delayed(weights_on_df)(m, kdf, columns)
                    for m in tqdm(self.weights_methods, desc="Calculating weights")
                )
                kdf_test = df.iloc[test_index]
                weights_results: List[Tuple[str, Tuple[np.ndarray, float]]] = Parallel(
                    n_jobs=-1
                )(
                    delayed(eval_weights)(m, w, kdf_test, columns)
                    for m, w in tqdm(weights, desc="Evaluating weights")
                )
                weights_results_dict: Dict[str, Tuple[np.ndarray, float]] = dict(
                    weights_results
                )
                for m_name in weights_results_dict:
                    partial_result_dict[m_name].append(weights_results_dict[m_name])
            results: Dict[str, Tuple[np.ndarray, float]] = {}
            for m_name in partial_result_dict:
                values: List[Tuple[np.ndarray, float]] = partial_result_dict[m_name]
                weights_list: List[np.ndarray] = []
                eval_list: List[float] = []
                for value in values:
                    weights_list.append(value[0])
                    eval_list.append(value[1])
                weights_avg: np.ndarray = np.mean(weights_list, axis=0)
                eval_avg: float = np.mean(eval_list)
                results[m_name] = (weights_avg, eval_avg)
            self.weights = results
        else:
            results: List[Tuple[str, Tuple[np.ndarray, float]]] = Parallel(n_jobs=-1)(
                delayed(fold_check)(m, df, columns)
                for m in tqdm(self.weights_methods, desc="Fold check")
            )
            self.weights = dict(results)

    def adapt_process(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ):
        """
        自适应训练核心流程

        参数：
            df (pd.DataFrame): 完整训练数据
            columns (list): 特征列名列表

        阶段：
            1. 权重计算阶段：调用compute_weights选择最佳权重方法
            2. 模型选择阶段：遍历所有可能的（评分×模型×筛选）组合
               - 使用交叉验证评估每个组合的性能
               - 选择最佳组合配置

        更新：
            self.reg_model: 最佳回归模型实例
            self.cut_method: 最佳特征筛选方法
            self.score_method: 最佳评分方法
        """
        eprint("=============== Weights Select")
        self.compute_weights(df, columns)

        # 选择最佳权重方法
        w_maks: float = 0
        w_method: str = None
        w_weights: np.ndarray = None
        for k, v in self.weights.items():
            # 存储每种方法的评估结果
            self.prescoring_log.append((k, v[1]))
            # 记录最佳方法
            if v[1] > w_maks:
                w_maks = v[1]
                w_method = k
                w_weights = v[0]

        self.weights = w_weights
        self.weights_score = w_maks
        eprint(f"Best weights method: {w_method} MAP: {w_maks}")
        self.best_prescoring_log.append((w_method, w_maks))
        eprint("===============")

        eprint("=============== Size and regression model select")

        results = Parallel(n_jobs=-1)(
            delayed(self._train)(
                df,
                columns,
                w_weights,
                score_method,
                reg_model,
                cut_method,
            )
            for score_method, reg_model, cut_method in tqdm(
                product(self.score_methods, self.reg_models, self.cut_methods),
                desc="Training models",
            )
        )

        res_max = 0
        for res in results:
            current_name = res[0]
            current_cut_function = res[1]
            current_score_function = res[2]
            current_score = res[3]
            current_reg_model = self.reg_models_map[current_name]

            name = self.prepare_regressor_name(current_reg_model)
            self.regression_log.append(
                (name, current_cut_function, current_score_function, current_score)
            )
            if current_score > res_max:
                res_max = current_score
                self.reg_model_name = current_name
                self.cut_method_name = current_cut_function
                self.score_method_name = current_score_function

        self.reg_model = self.reg_models_map[self.reg_model_name]
        self.cut_method = self.cut_methods_map[self.cut_method_name]
        self.score_method = self.score_methods_map[self.score_method_name]

        self.reg_model_score = res_max
        current_reg_model = self.reg_model
        name = self.prepare_regressor_name(current_reg_model)
        self.best_regression_log.append(
            (name, self.cut_method_name, self.score_method_name, self.reg_model_score)
        )

        eprint(
            f"Best regression model: {self.reg_model_name} MAP: {res_max} Cut: {self.cut_method_name}"
        )
        eprint("===============")

    def train(self, df: pd.DataFrame):
        """
        训练入口函数

        参数：
            df (pd.DataFrame): 当前折的训练数据

        逻辑：
            - 首折或强制学习模式下执行完整adapt_process
            - 后续折可复用已有配置（当enforce_relearning=False时）
            - 记录训练时间和数据规模

        返回：
            当前配置的回归模型对象
        """
        before_training = default_timer()
        columns = node_feature_columns.copy()

        if not self.first_fold_processed or self.enforce_relearning:
            self.adapt_process(df, columns)
            self.first_fold_processed = True

        self._train(
            df,
            columns,
            self.weights,
            self.score_method,
            self.reg_model,
            self.cut_method,
        )

        after_training = default_timer()
        total_training = after_training - before_training
        self.training_time_list.append(
            (
                total_training,
                df.index.get_level_values(0).unique().shape[0],
                df.index.get_level_values(1).unique().shape[0],
            )
        )

        return self.reg_model

    def predict(self, clf: SGDRegressor, df: pd.DataFrame):
        """
        预测函数

        参数：
            clf: 训练好的回归模型（实际可能未使用）
            df (pd.DataFrame): 测试数据
            data (Data): 测试数据图数据

        决策逻辑：
            - 当回归模型得分高于权重方法得分时（或强制使用回归模型），使用回归预测
            - 否则使用特征权重线性组合得分

        返回：
            pd.DataFrame: 包含预测结果（result列）及原始字段的副本
        """
        columns = node_feature_columns.copy()

        X = df[columns].values

        # Check if weights method gives better results on training
        if not self.use_prescoring_always and (
            self.reg_model_score >= self.weights_score or self.use_reg_model_always
        ):
            # 将 X 转换为带有列名的 DataFrame
            X_df = pd.DataFrame(X, columns=columns)
            result = clf.predict(X_df)
        else:
            print("Using weights")
            result = np.dot(X, self.weights)

        r = df[["used_in_fix"]].copy(deep=False)
        r["result"] = result

        return r

    def _train(
        self,
        df: pd.DataFrame,
        columns: List[str],
        weights: np.ndarray,
        score_method: Callable[[pd.DataFrame, List[str], np.ndarray], np.ndarray],
        reg_model: SGDRegressor,
        cut_method: Callable[[pd.DataFrame, np.ndarray], pd.Series],
    ):
        """
        单配置评估内部方法

        参数：
            df (pd.DataFrame): 训练数据
            columns (list): 特征列
            dependency_df (pd.DataFrame): 依赖数据
            dependency_columns (list): 依赖特征列
            weights (np.ndarray): 当前权重向量
            score_method (function): 评分方法
            reg_model: 回归模型实例
            cut_method (function): 特征筛选方法

        流程：
            1. 计算特征得分并修正（增加修复样本权重）
            2. 应用特征筛选获取训练子集
            3. 使用交叉验证训练回归模型并评估

        返回：
            tuple: (模型标识, 筛选方法名, 评分方法名, 平均MAP得分)
        """
        score = score_method(df, columns, weights)
        score_fixed = score + df["used_in_fix"] * np.max(score)

        if self.use_training_cross_validation:
            kfold = KFold(
                n_splits=self.cross_validation_fold_number,
                random_state=None,
                shuffle=False,
            )
            partial_eval_results = []
            for train_index, test_index in kfold.split(df):
                kdf = df.iloc[train_index]
                kscore = score[train_index]
                kscore_fixed = score_fixed.iloc[train_index]

                kdf_test = df.iloc[test_index]
                pres = cut_fit_predict(
                    kdf, kdf_test, columns, kscore, kscore_fixed, cut_method, reg_model
                )

                partial_eval_results.append(pres)
            eval_result = np.mean(partial_eval_results)
            return (
                str(reg_model),
                cut_method.__name__,
                score_method.__name__,
                eval_result,
            )
        else:
            return (
                str(reg_model),
                cut_method.__name__,
                score_method.__name__,
                cut_fit_predict(
                    df, df, columns, score, score_fixed, cut_method, reg_model
                ),
            )

    def prepare_regressor_name(self, current_reg_model: SGDRegressor):
        if isinstance(current_reg_model, SGDRegressor):
            penalty = (
                current_reg_model.penalty
                if current_reg_model.penalty is not None
                else "none"
            )
            name = (
                "SGDRegressor"
                + "_"
                + current_reg_model.loss
                + "_"
                + penalty
                + "_"
                + str(current_reg_model.alpha)
                + "_"
                + str(current_reg_model.shuffle)
            )
        else:
            name = self.reg_model_name
        return name

    # ---------------------- 辅助函数 ----------------------


def cut_fit_predict(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    columns: List[str],
    score: np.ndarray,
    score_fixed: np.ndarray,
    cut_method: Callable[[pd.DataFrame, np.ndarray], pd.Series],
    reg_model: SGDRegressor,
):
    """
    特征筛选-训练-预测流水线

    参数：
        df (pd.DataFrame): 训练数据
        df_test (pd.DataFrame): 测试数据
        columns (list): 特征列
        score (np.ndarray): 原始特征得分
        score_fixed (np.ndarray): 修正后的特征得分（增强修复样本）
        cut_method (function): 特征筛选方法
        reg_model: 回归模型实例

    流程：
        1. 应用cut_method筛选训练样本
        2. 在筛选集上训练回归模型
        3. 在完整测试集上预测

    返回：
        float: 当前配置的MAP评估得分
    """
    cut_set = cut_method(df, score)
    X = df[cut_set]

    reg_model.fit(X[columns], score_fixed[cut_set])
    Y = reg_model.predict(df_test[columns])

    return evaluate_fold(df_test, Y)


def get_skmodels():
    sgd_loss = [
        "squared_error",
        "huber",
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ]
    sgd_penalty = [None, "l2", "l1", "elasticnet"]
    alpha = 10.0 ** -np.arange(4, 5)
    return [
        SGDRegressor(max_iter=1000, shuffle=False, loss=l, penalty=p, alpha=a)
        for l, p, a in product(sgd_loss, sgd_penalty, alpha)
    ]


def _process(
    ptemplate: Adaptive_Process,
    fold_training: pd.DataFrame,
    fold_testing: pd.DataFrame,
):
    clf = ptemplate.train(fold_training)
    result = ptemplate.predict(clf, fold_testing)
    return result


def process(
    ptemplate: Adaptive_Process,
    fold_number: int,
    fold_testing: Dict[int, pd.DataFrame],
    fold_training: Dict[int, pd.DataFrame],
    file_prefix: str,
):
    """
    主处理函数

    参数：
        ptemplate (Adaptive_Process): 自适应算法实例
        fold_number: 折数
        fold_testing: 测试数据
        fold_training: 训练数据
        file_prefix: 文件前缀
    """
    results_list = []

    for i in range(fold_number):
        r = _process(
            ptemplate,
            fold_training[i],
            fold_testing[i + 1],
        )
        if r is None:
            del ptemplate
            gc.collect()
            return None

        results_list.append(r)

    all_results_df = pd.concat(results_list)
    all_results_df.reset_index(level=1, drop=True, inplace=True)

    training_time_list = ptemplate.training_time_list.copy()
    prescoring_log = ptemplate.prescoring_log.copy()
    regression_log = ptemplate.regression_log.copy()
    best_prescoring_log = ptemplate.best_prescoring_log.copy()
    best_regression_log = ptemplate.best_regression_log.copy()

    eprint(training_time_list)
    time_sum, bug_reports_number_sum, file_number_sum = map(
        sum, zip(*training_time_list)
    )

    eprint("time_sum", time_sum)
    eprint("bug_reports_number_sum", bug_reports_number_sum)
    eprint("file_number_sum", file_number_sum)

    mean_time_bug_report_training = time_sum / bug_reports_number_sum
    mean_time_file_training = time_sum / file_number_sum

    eprint("mean_time_bug_report_training", mean_time_bug_report_training)
    eprint("mean_time_file_training", mean_time_file_training)

    results_timestamp = time.strftime("%Y%m%d%H%M%S")

    training_time = {
        "time_sum": time_sum,
        "bug_reports_number_sum": bug_reports_number_sum,
        "file_number_sum": file_number_sum,
        "mean_time_bug_report_training": mean_time_bug_report_training,
        "mean_time_file_training": mean_time_file_training,
    }
    with open(
        f"{file_prefix}_{ptemplate.name}_training_time_{results_timestamp}.json", "w"
    ) as time_file:
        json.dump(training_time, time_file)

    prescoring_log = ptemplate.prescoring_log.copy()
    with open(
        f"{file_prefix}_{ptemplate.name}_prescoring_log_{results_timestamp}.json", "w"
    ) as prescoring_log_file:
        json.dump(prescoring_log, prescoring_log_file)

    regression_log = ptemplate.regression_log.copy()
    with open(
        f"{file_prefix}_{ptemplate.name}_regression_log_{results_timestamp}.json", "w"
    ) as regression_log_file:
        json.dump(regression_log, regression_log_file)

    best_prescoring_log = ptemplate.best_prescoring_log.copy()
    with open(
        f"{file_prefix}_{ptemplate.name}_best_prescoring_log_{results_timestamp}.json",
        "w",
    ) as best_prescoring_log_file:
        json.dump(best_prescoring_log, best_prescoring_log_file)

    best_regression_log = ptemplate.best_regression_log.copy()
    with open(
        f"{file_prefix}_{ptemplate.name}_best_regression_log_{results_timestamp}.json",
        "w",
    ) as best_regression_log_file:
        json.dump(best_regression_log, best_regression_log_file)

    return {
        "name": ptemplate.name,
        "results": calculate_metric_results(all_results_df),
    }


# region 特征权重计算
def _weights_normalize(weights: np.ndarray):
    """
    权重归一化函数

    参数：
        weights: 原始权重向量

    返回：
        weights: L1归一化后的权重向量（总和为1）

    说明：
        - 当权重总和>0时执行归一化
        - 处理全零权重时保持原值
    """
    weights_sum = weights.sum()
    if weights_sum > 0:
        weights /= weights_sum

    return weights


def weights_chi2(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    基于卡方检验的特征权重计算

    参数：
        df (pd.DataFrame): 包含特征和标签的数据
        columns (list): 特征列名列表

    返回：
        np.ndarray: 归一化的卡方统计量作为特征权重

    实现：
        使用sklearn.feature_selection.chi2计算各特征与目标变量的卡方统计量
    """
    weights = chi2(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def weights_mutual_info_classif(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """基于互信息的特征权重计算，适用于连续特征"""
    weights = mutual_info_classif(
        df[columns], df["used_in_fix"], discrete_features=False
    )
    weights = weights

    return _weights_normalize(weights)


def weights_FastICA(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """使用独立成分分析(ICA)的首个成分作为特征权重"""
    m = FastICA(n_components=1)
    m.fit(df[columns])
    weights = m.components_[0]

    return _weights_normalize(weights)


def weights_variance(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """基于特征方差的权重计算，负方差置零处理"""
    fs = VarianceThreshold()
    fs.fit(df[columns])
    weights = fs.variances_
    weights[weights < 0] = 0

    return _weights_normalize(weights)


def weights_const(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """恒定权重（0.5），用作基准方法"""
    return np.ones(df[columns].shape[1]) * 0.5


def weights_ExtraTreesClassifier(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """基于极端随机树分类器的特征重要性"""
    tree = ExtraTreesClassifier(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_GradientBoostingClassifier(
    df: pd.DataFrame, columns: List[str]
) -> np.ndarray:
    """梯度提升回归树特征重要性"""
    tree = GradientBoostingRegressor(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_AdaBoostClassifier(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """AdaBoost分类器特征重要性"""
    tree = AdaBoostClassifier(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_kruskal_classif(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    Kruskal-Wallis检验统计量作为权重
    适用于非正态分布的组间差异检验
    """
    weights = kruskal_classif(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def kruskal_classif(X, y):
    """
    执行Kruskal-Wallis H检验
    返回各特征的检验统计量绝对值和p值
    """
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = kruskal(*args)
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_ttest_ind_classif(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    独立样本t检验统计量作为权重
    假设方差不相等（Welch's t-test）
    """
    weights = ttest_ind_classif(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def ttest_ind_classif(X, y):
    """执行Welch's t-test，返回绝对t值和p值"""
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = ttest_ind(*args, equal_var=False)
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_levene_median(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    基于中位数Levene检验的权重计算
    用于检测组间方差差异
    """
    weights = levene_median(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def levene_median(X, y):
    """执行基于中位数的Levene方差齐性检验"""
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = levene(args[0], args[1], center="median")
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_mean_var(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    均值-方差比率权重：
    (修复样本的变异系数) / (非修复样本的变异系数)
    变异系数 = 标准差 / 均值
    """
    weights_var = np.var(df[df["used_in_fix"] == 1][columns], axis=0)
    weights_mean = np.mean(df[df["used_in_fix"] == 1][columns], axis=0)
    weights_var1 = np.var(df[df["used_in_fix"] == 0][columns], axis=0)
    weights_var1_mean = np.mean(df[df["used_in_fix"] == 0][columns], axis=0)

    return (weights_var / weights_mean) / (weights_var1 / weights_var1_mean)


def weights_maximum_absolute_deviation(
    df: pd.DataFrame, columns: List[str]
) -> np.ndarray:
    """
    最大绝对偏差权重：
    计算修复样本各特征值与其最大值的平均绝对偏差
    """
    weights_max = np.max(df[df["used_in_fix"] == 1][columns], axis=0)
    weights_mad = np.mean(
        np.abs(df[df["used_in_fix"] == 1][columns] - weights_max), axis=0
    )

    return weights_mad


# endregion


def weights_on_df(
    method: Callable[[pd.DataFrame, List[str]], np.ndarray],
    df: pd.DataFrame,
    columns: List[str],
):
    """
    单权重方法计算包装函数

    参数：
        method: 权重计算方法
        df: 当前折训练数据
        columns: 特征列名列表

    返回：
        tuple: (方法名称, 权重向量)

    说明：
        - 用于并行计算任务包装
        - 调用具体权重计算方法并返回标准化结果
    """
    weights = method(df, columns)
    return method.__name__, weights


def eval_weights(
    m_name: str, weights: np.ndarray, df: pd.DataFrame, columns: List[str]
) -> Tuple[str, Tuple[np.ndarray, float]]:
    """
    权重评估函数（验证阶段）

    参数：
        m_name: 权重方法名称
        weights: 已计算的权重向量
        df: 验证集数据
        columns: 特征列名列表

    返回：
        tuple: (方法名称, (权重向量, MAP得分))

    流程：
        1. 计算验证集预测得分：X * weights
        2. 调用evaluate_fold计算MAP指标
    """
    Y = np.dot(df[columns], weights)
    return m_name, (weights, evaluate_fold(df, Y))


def fold_check(
    method: Callable[[pd.DataFrame, List[str]], np.ndarray],
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[str, Tuple[np.ndarray, float]]:
    """
    单折权重评估函数（非交叉验证模式）

    参数：
        method (function): 权重计算方法
        df (pd.DataFrame): 完整训练数据
        columns (list): 特征列名列表

    返回：
        tuple: (方法名称, (权重向量, MAP得分))

    说明：
        - 当use_prescoring_cross_validation=False时使用
        - 直接在整个数据集上计算和评估
    """
    weights = method(df, columns)
    Y = np.dot(df[columns], weights)
    return method.__name__, (weights, evaluate_fold(df, Y))


def evaluate_fold(df: pd.DataFrame, Y: np.ndarray) -> float:
    """
    评估预测结果的MAP指标

    参数：
        df: 待评估数据集（需包含used_in_fix列）
        Y: 预测得分向量

    返回：
        m_a_p: 平均精度均值（Mean Average Precision）

    流程：
        1. 构建结果数据框（含预测得分）
        2. 确定最小修复得分阈值（实际修复样本的最低得分）
        3. 生成候选集（预测得分≥阈值的样本）
        4. 调用calculate_metric_results计算指标
    """
    r = df[["used_in_fix"]].copy(deep=False)
    r["result"] = Y
    _, m_a_p, _, _ = calculate_metric_results(r)
    return m_a_p


def normal_score(
    df: pd.DataFrame, columns: List[str], weights: np.ndarray
) -> np.ndarray:
    """
    计算特征线性组合得分（基本评分函数）

    参数：
        df: 包含特征的数据框
        columns: 要使用的特征列名列表
        weights: 特征权重向量，维度应与columns长度一致

    返回：
        np.ndarray: 每个样本的得分向量，通过特征与权重的点积计算

    说明：
        这是最基本的评分方法，通过特征向量与权重向量的点积计算样本得分。
        计算公式: score = X * weights，其中X为样本的特征矩阵
    """
    # 计算特征矩阵与权重向量的点积
    score = np.dot(df[columns], weights)
    return score


def size_selectf_only_fixes_p(
    df: pd.DataFrame, score: np.ndarray, perc: float, smallest=True, largest=False
) -> pd.Series:
    """
    动态特征筛选策略：基于修复样本百分比扩展候选集

    本函数实现了一种混合筛选策略，首先保留所有实际修复的文件（正样本），
    然后根据得分阈值扩展一定比例的非修复文件（负样本）用于训练。
    可以选择包含得分最低或最高的负样本，以增强模型的鲁棒性。

    逻辑：
        - 强制包含所有实际修复样本（used_in_fix=1）
        - 按百分比扩展得分接近的负样本（未修复但得分接近）
        - 过滤掉得分小于等于0的样本

    参数：
        df: 包含used_in_fix列的数据框
        score: 特征得分向量，与df行数相同
        perc: 扩展百分比（相对于修复样本数）
        smallest: 是否包含得分最低的负样本，默认True
        largest: 是否包含得分最高的负样本，默认False

    返回：
        pd.Series: 布尔序列，True表示选中的样本

    示例：
        >>> # 选择所有修复样本及额外5%的低分非修复样本
        >>> selected = size_selectf_only_fixes_p(df, scores, 0.05)
        >>> train_set = df[selected]
    """
    # 标识所有修复样本（正样本）
    used_in_fix = df["used_in_fix"] == 1
    # 初始选择集合为所有修复样本
    ret = used_in_fix

    # 创建临时数据框，用于筛选操作
    G = df[["used_in_fix"]].copy(deep=False)
    G["score"] = score

    # 只考虑得分大于0的样本
    t = G[G["score"] > 0]["score"]

    if smallest:
        # 选择得分最低的一部分负样本
        # 数量等于修复样本数量乘以perc
        tm = t.nsmallest(int(perc * used_in_fix.sum())).max()
        # 添加这些低分样本到结果集
        ret |= G["score"] <= tm

    if largest:
        # 选择得分最高的一部分负样本
        # 数量等于修复样本数量乘以perc
        tm = t.nlargest(int(perc * used_in_fix.sum())).min()
        # 添加这些高分样本到结果集
        ret |= G["score"] >= tm

    # 最终过滤：确保所有选中样本得分都大于0
    ret &= G["score"] > 0

    return ret


def size_selectf_only_fixes_p_perc_05(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """
    筛选策略：保留所有修复样本外加5%的低分非修复样本

    这是size_selectf_only_fixes_p的便捷封装，固定perc=0.05

    参数：
        df: 包含used_in_fix列的数据框
        score: 特征得分向量

    返回：
        pd.Series: 布尔序列，True表示选中的样本
    """
    return size_selectf_only_fixes_p(df, score, perc=0.05)


def size_selectf_only_fixes_p_perc_10(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """
    筛选策略：保留所有修复样本外加10%的低分非修复样本

    这是size_selectf_only_fixes_p的便捷封装，固定perc=0.10

    参数：
        df: 包含used_in_fix列的数据框
        score: 特征得分向量

    返回：
        pd.Series: 布尔序列，True表示选中的样本
    """
    return size_selectf_only_fixes_p(df, score, perc=0.10)


def size_selectf_only_fixes_p_perc_15(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """
    筛选策略：保留所有修复样本外加15%的低分非修复样本

    这是size_selectf_only_fixes_p的便捷封装，固定perc=0.15

    参数：
        df: 包含used_in_fix列的数据框
        score: 特征得分向量

    返回：
        pd.Series: 布尔序列，True表示选中的样本
    """
    return size_selectf_only_fixes_p(df, score, perc=0.15)


def size_selectf_only_fixes_p_perc_20(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """
    筛选策略：保留所有修复样本外加20%的低分非修复样本

    这是size_selectf_only_fixes_p的便捷封装，固定perc=0.20

    参数：
        df: 包含used_in_fix列的数据框
        score: 特征得分向量

    返回：
        pd.Series: 布尔序列，True表示选中的样本
    """
    return size_selectf_only_fixes_p(df, score, perc=0.20)


def size_selectf_only_fixes_p_perc_25(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """
    筛选策略：保留所有修复样本外加25%的低分非修复样本

    这是size_selectf_only_fixes_p的便捷封装，固定perc=0.25

    参数：
        df: 包含used_in_fix列的数据框
        score: 特征得分向量

    返回：
        pd.Series: 布尔序列，True表示选中的样本
    """
    return size_selectf_only_fixes_p(df, score, perc=0.25)


def size_selectf_only_fixes_p_perc_30(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """
    筛选策略：保留所有修复样本外加30%的低分非修复样本

    这是size_selectf_only_fixes_p的便捷封装，固定perc=0.30

    参数：
        df: 包含used_in_fix列的数据框
        score: 特征得分向量

    返回：
        pd.Series: 布尔序列，True表示选中的样本
    """
    return size_selectf_only_fixes_p(df, score, perc=0.30)


def main():
    # parser = argparse.ArgumentParser(description="Train Adaptive Process")
    # parser.add_argument("file_prefix", type=str, help="Feature files prefix")
    # parser.add_argument("--max", action="store_true", help="Include feature 37")
    # parser.add_argument("--mean", action="store_true", help="Include feature 38")
    # args = parser.parse_args()

    # # 根据参数决定是否添加特征37和38
    # if args.max:
    #     node_feature_columns.append("f37")
    # if args.mean:
    #     node_feature_columns.append("f38")

    # file_prefix = args.file_prefix

    file_prefix = "aspectj"

    (
        fold_number,
        fold_testing,
        fold_training,
        _,
        _,
    ) = load(f"../joblib_memmap_{file_prefix}_graph/data_memmap", mmap_mode="r")

    models = [Adaptive_Process()]
    results = []
    for m in models:
        results.append(
            process(
                m,
                fold_number,
                fold_testing,
                fold_training,
                file_prefix,
            )
        )

    results = [r for r in results if r is not None]
    eprint("Results")
    for result in results:
        print("name ", result["name"])
        print_metrics(*result["results"])


if __name__ == "__main__":
    main()
