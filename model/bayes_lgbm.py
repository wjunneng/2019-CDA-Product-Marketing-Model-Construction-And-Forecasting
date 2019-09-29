# -*- coding: utf-8 -*-

import lightgbm as lgb
from bayes_opt import BayesianOptimization
from demo.preprocess import Preprocess
from configuration.config import DefaultConfig

import warnings

warnings.filterwarnings('ignore')

"""
贝叶斯优化
封装成函数
方便配合特征选择算法一起使用
"""


def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, n_estimators=10000,
                            learning_rate=0.05, categorical_feats="auto"):
    """

    :param X: 训练数据
    :param y: 训练数据y
    :param init_round: 随机探索步骤，可以使探索空间多样化
    :param opt_round: 迭代次数
    :param n_folds: 交叉验证
    :param n_estimators: 生成的最大树的数目，也是最大的迭代次数。
    :param learning_rate: 学习率
    :return: 一个全量的参数
    """
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, categorical_feature=categorical_feats, free_raw_data=False)

    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain,
                 min_child_weight):
        params = dict({'application': 'binary',  # regression 回归    binary 二分类   multiclass 多分类
                       'num_iterations': n_estimators,
                       'learning_rate': learning_rate,
                       'early_stopping_round': 1024,  # 如果一次验证数据的一个度量在最近的early_stopping_round 回合中没有提高，模型将停止训练
                       'metric': 'auc'})
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, stratified=True, verbose_eval=200,
                           metrics=['auc'])
        return max(cv_result['auc-mean'])

    # range
    LGB_Bayes = BayesianOptimization(lgb_eval, pbounds={'num_leaves': (25, 100),
                                                        'feature_fraction': (0.3, 0.9),
                                                        'bagging_fraction': (0.5, 1),
                                                        'max_depth': (5, 15),
                                                        'lambda_l1': (0, 5),
                                                        'lambda_l2': (0, 3),
                                                        'min_split_gain': (0.001, 0.1),
                                                        'min_child_weight': (5, 50)},
                                     random_state=0)
    # optimize
    # init_points ==  How many steps of random exploration you want to perform.
    # Random exploration can help by diversifying the exploration space.
    # n_iter == 迭代次数
    LGB_Bayes.maximize(init_points=init_round, n_iter=opt_round)

    append_params = {'application': 'binary',  # regression 回归    binary 二分类   multiclass 多分类
                     'num_iterations': n_estimators,
                     'learning_rate': learning_rate,
                     'early_stopping_round': 100,  # 如果一次验证数据的一个度量在最近的early_stopping_round 回合中没有提高，模型将停止训练
                     'metric': 'auc'}

    # return best parameters
    return LGB_Bayes, append_params


if __name__ == '__main__':
    X, df_test = Preprocess(df_training_path=DefaultConfig.df_training_path,
                            df_test_path=DefaultConfig.df_test_path).main()

    y = X[DefaultConfig.label_column].astype(int)

    del X[DefaultConfig.label_column]

    categorical_feats = [i for i in list(X.columns) if
                         i not in ['ID', 'Product using score', 'age', 'Point balance', 'Estimated salary']]

    opt_params, append_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=5,
                                                        n_estimators=2019,
                                                        learning_rate=0.05,
                                                        categorical_feats=categorical_feats)

    max_score = opt_params.max["target"]
    max_par = opt_params.max['params']

    max_par.update(append_params)
    max_par["num_leaves"] = int(round(max_par["num_leaves"]))
    max_par["max_depth"] = int(round(max_par["max_depth"]))

    print("------------------------------------------")
    print(max_score)
    print(max_par)
