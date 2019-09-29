import gc
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

from configuration.config import DefaultConfig
from demo.preprocess import Preprocess


class LightGbm(object):
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    def f1_score(self, y_hat, data):
        """
        F1-score
        :param y_hat:
        :param data:
        :return:
        """
        y_true = list(data.get_label())
        y_hat = np.round(y_hat)

        return 'f1', f1_score(y_true, y_hat), True

    def main(self, **params):
        """
        lgb 模型
        :param new_train:
        :param y:
        :param new_test:
        :param columns:
        :param params:
        :return:
        """
        # y_train
        y_train = list(self.X_train.loc[:, DefaultConfig.label_column])
        # 特征重要性
        feature_importance = None
        # 线下验证
        oof = np.zeros((self.X_train.shape[0]))
        # 线上结论
        prediction = np.zeros((self.X_test.shape[0]))

        seeds = [42, 2019, 223344, 2019 * 2 + 1024, 332232111]
        num_model_seed = 1

        print(DefaultConfig.select_model + ' start training...')

        for model_seed in range(num_model_seed):
            params = {'bagging_fraction': 0.5121462324340804, 'feature_fraction': 0.5174384402885819,
                      'lambda_l1': 4.969972779088037, 'lambda_l2': 0.9329835645863005, 'max_depth': 5,
                      'min_child_weight': 5.594628962551301, 'min_split_gain': 0.04198971046219193, 'num_leaves': 25,
                      'application': 'binary', 'num_iterations': 2019, 'learning_rate': 0.05,
                      'early_stopping_round': 100}

            print('模型 ', model_seed + 1, ' 开始训练')

            oof_lgb = np.zeros((self.X_train.shape[0]))

            # 获取验证集
            df_training, df_validation, df_test = Preprocess.get_validation_data(df_training=self.X_train,
                                                                                 df_test=self.X_test)
            # 0/1数目
            print('df_traing[label].value_counts:')
            print(df_training[DefaultConfig.label_column].value_counts())

            print('df_validation[label].value_counts:')
            print(df_validation[DefaultConfig.label_column].value_counts())

            # 分割
            train_x, test_x, train_y, test_y = df_training.drop(labels=DefaultConfig.label_column, axis=1), \
                                               df_validation.drop(labels=DefaultConfig.label_column, axis=1), \
                                               df_training.loc[:, DefaultConfig.label_column], \
                                               df_validation.loc[:, DefaultConfig.label_column]

            train_data = lgb.Dataset(train_x, label=train_y)
            validation_data = lgb.Dataset(test_x, label=test_y)

            gc.collect()
            bst = lgb.train(params, train_data, valid_sets=[validation_data], verbose_eval=1000, feval=self.f1_score)

            oof_lgb[test_x.index] += bst.predict(test_x)

            prediction_lgb = bst.predict(self.X_test.drop(DefaultConfig.label_column, axis=1))

            gc.collect()

            # 存放特征重要性
            feature_importance_df = pd.DataFrame()
            feature_importance_df["feature"] = list(bst.feature_name())
            feature_importance_df["importance"] = bst.feature_importance(importance_type='split',
                                                                         iteration=bst.best_iteration)

            oof += oof_lgb / num_model_seed
            prediction += prediction_lgb / num_model_seed
            print('logloss', log_loss(pd.get_dummies(y_train).values, oof_lgb))
            # 线下auc评分
            print('the roc_auc_score for train:', roc_auc_score(y_train, oof_lgb))

            if feature_importance is None:
                feature_importance = feature_importance_df
            else:
                feature_importance += feature_importance_df

        print('logloss', log_loss(pd.get_dummies(y_train).values, oof))
        print('ac', roc_auc_score(y_train, oof))

        if feature_importance is not None:
            feature_importance['importance'] /= num_model_seed
            plt.figure(figsize=(8, 8))

            print(list(dict(
                feature_importance.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False)).keys()))
            # 5折数据取平均值
            feature_importance.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).plot.barh()

            plt.show()

        return prediction

