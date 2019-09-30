import pandas as pd
import os

import numpy as np
from utils.fill_values import FillValues
from configuration.config import DefaultConfig
from utils.utils import Utils
from collections import Counter
import sys


class Preprocess(object):
    """
    预处理类
    """

    def __init__(self, df_test_path, df_training_path):
        self.df_test_path = df_test_path
        self.df_training_path = df_training_path

    def get_df_test(self, **params) -> (pd.DataFrame, list):
        """
        获取df_test
        :param params:
        :return:
        """
        df = pd.read_csv(filepath_or_buffer=self.df_test_path, encoding='utf-8')

        ids = list(df['ID'])

        df.sort_values(by='ID', inplace=True)

        df.rename(columns=dict(zip(list(df.columns), [i.strip("'").strip(' ') for i in list(df.columns)])),
                  inplace=True)

        df.reset_index(inplace=True, drop=True)

        return df, ids

    def drop_row(self, df, **params) -> pd.DataFrame:
        """
        删除特定行
        :param df:
        :param params:
        :return:
        """
        df_ = df.copy()
        df_.replace('?', np.nan, inplace=True)
        # 找到行
        row = np.where(pd.isna(df_))[0]
        # 过滤行
        row = dict(filter(lambda x: x[1] >= 6, dict(Counter(row)).items()))
        # 过滤行
        df = df[~df.index.isin(row.keys())]

        return df

    def get_df_training(self, **params) -> pd.DataFrame:
        """
        获取df_training
        :param params:
        :return:
        """
        df = pd.read_csv(filepath_or_buffer=self.df_training_path, encoding='utf-8')

        df = self.drop_row(df)

        df['ID'] = df['ID'].astype(int)

        df.sort_values(by='ID', inplace=True)

        df.rename(columns=dict(zip(list(df.columns), [i.strip("'").strip(' ') for i in list(df.columns)])),
                  inplace=True)

        df.reset_index(inplace=True, drop=True)

        return df

    def deal_gender(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        三、处理gender
        :param df:
        :param params:
        :return:
        """
        column = "gender"
        df[column] = df[column].apply(lambda x: 0 if x is np.nan else x)
        df[column] = df[column].apply(lambda x: 1 if x == "Male" else x)
        df[column] = df[column].apply(lambda x: 2 if x == "Female" else x)

        return df

    def deal_User_area(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        二、处理User_area
        :param df:
        :param params:
        :return:
        """
        column = "User area"

        df[column] = df[column].apply(lambda x: 0 if x is np.nan else x)
        df[column] = df[column].apply(lambda x: 1 if x == "Taipei" else x)
        df[column] = df[column].apply(lambda x: 2 if x == "Taichung" else x)
        df[column] = df[column].apply(lambda x: 3 if x == "Tainan" else x)

        return df

    def get_quantile(self, df, column, **params):
        """
        获取分位数
        :param df:
        :param params:
        :return:
        """
        import numpy as np

        df.replace(np.nan, -1, inplace=True)

        df_ = df[df[column] != -1]

        df_[column] = df_[column].astype(int)

        quantile = dict(df_[column].describe())
        # min/25%/50%/75%/max
        return [quantile['25%'], quantile['50%'], quantile['75%']]

    def pre_deal(self, df, df_type=False, **params) -> pd.DataFrame:
        """
        预处理
        :param df:
        :param params:
        :return:
        """
        print('pre_deal start...')

        # 0、处理age
        quantile = []
        if df_type:
            df_training_0_tmp = df[df[DefaultConfig.label_column] == 0]
            df_training_1_tmp = df[df[DefaultConfig.label_column] == 1]
            df_test_tmp = df[(df[DefaultConfig.label_column] != 0) & (df[DefaultConfig.label_column] != 1)]

            for df_tmp in [df_training_0_tmp, df_training_1_tmp, df_test_tmp]:
                quantile.extend(self.get_quantile(df=df_tmp, column="age"))

        # 类别标签
        label_value = df[DefaultConfig.label_column]
        del df[DefaultConfig.label_column]

        # 一、类别编码
        df = self.deal_User_area(df)
        df = self.deal_gender(df)

        # 二、处理缺失值 均值
        missing_value_columns = ['Product using score', 'age', 'Point balance', 'Estimated salary']
        df[missing_value_columns] = df[missing_value_columns].astype(float)
        df = FillValues(df=df, columns=missing_value_columns).simplefill(fillmethod='mean')
        df[['Product using score', 'age']] = df[['Product using score', 'age']].astype(int)
        # df[['Point balance', 'Estimated salary']] = df[['Point balance', 'Estimated salary']].astype(float)

        # 三、处理缺失值 众数
        missing_value_columns = ['Cumulative using time', 'Product service usage',
                                 'Pay a monthly fee by credit card', 'Active user']
        df[missing_value_columns] = df[missing_value_columns].astype(float)
        df = FillValues(df=df, columns=missing_value_columns).simplefill(fillmethod='mode')
        df[missing_value_columns] = df[missing_value_columns].astype(int)

        df[DefaultConfig.label_column] = label_value

        # 添加新列
        if df_type and len(quantile) != 0:
            quantile.extend([sys.maxsize, -1])
            quantile.sort()

            for i in range(len(quantile)):
                df['age_discrete'] = df['age'].copy()
                df['age_discrete'] = pd.cut(x=df['age_discrete'], bins=quantile, labels=list(range(len(quantile) - 1)))

            df['age_discrete'] = df['age_discrete'].astype(int)
        print('pre_deal end...')

        return df

    def main(self, save=True):
        """
        数据预处理
        :return:
        """
        df_training_cache_path = DefaultConfig.df_training_cache_path
        df_test_cache_path = DefaultConfig.df_test_cache_path
        df_cache_path = DefaultConfig.df_cache_path

        if os.path.exists(df_training_cache_path) and os.path.exists(
                df_test_cache_path) and os.path.exists(df_cache_path) and DefaultConfig.df_no_replace:
            df_training = Utils.reduce_mem_usage(pd.read_hdf(path_or_buf=df_training_cache_path, key='train', mode='r'))
            df_test = Utils.reduce_mem_usage(pd.read_hdf(path_or_buf=df_test_cache_path, key='test', mode='r'))
            df = Utils.reduce_mem_usage(pd.read_hdf(path_or_buf=df_cache_path, key='df', mode='r'))
        else:
            # 获取原数据
            df_test, _ = self.get_df_test()
            df_training = self.get_df_training()

            # 先将'?'替换为nan
            df_training.replace('?', np.nan, inplace=True)
            df_test.replace('?', np.nan, inplace=True)

            if DefaultConfig.select_model is 'lgbm':
                # ###############################################################
                # df_test
                df_test = self.pre_deal(df_test)
                # df_training
                df_training = self.pre_deal(df_training)

                # 合并
                df = pd.concat([df_training, df_test], ignore_index=True, axis=0)
                # ###############################################################
                # df_training_0 = df_training[df_training[DefaultConfig.label_column] == 0]
                # df_training_1 = df_training[df_training[DefaultConfig.label_column] == 1]
                #
                # df_training_0 = self.pre_deal(df_training_0)
                # df_training_1 = self.pre_deal(df_training_1)
                #
                # df_training = pd.concat([df_training_0, df_training_1], axis=0, ignore_index=True)
                #
                # df_training.sort_values(by='ID', inplace=True)
                #
                # df_training.reset_index(drop=True, inplace=True)
                # # 合并
                # df = pd.concat([df_training, df_test], ignore_index=True, axis=0)
                #
                # # df
                # df = self.pre_deal(df)

            elif DefaultConfig.select_model is 'cbt':
                # ################################################################ 先合并，再处理
                # 合并
                df = pd.concat([df_training, df_test], ignore_index=True, axis=0)

                # df
                df = self.pre_deal(df)
                # ################################################################# 先处理，再合并
                # df_training_0 = df_training[df_training[DefaultConfig.label_column] == 0]
                # df_training_1 = df_training[df_training[DefaultConfig.label_column] == 1]
                #
                # df_training_0 = self.pre_deal(df_training_0)
                # df_training_1 = self.pre_deal(df_training_1)
                #
                # df_training = pd.concat([df_training_0, df_training_1], axis=0, ignore_index=True)
                #
                # df_training.sort_values(by='ID', inplace=True)
                #
                # df_training.reset_index(drop=True, inplace=True)
                # # 合并
                # df = pd.concat([df_training, df_test], ignore_index=True, axis=0)
                #
                # # df
                # df = self.pre_deal(df)

            # 整数变量需要转化为整数
            df[DefaultConfig.int_columns] = df[DefaultConfig.int_columns].astype(int)

            count = df_training.shape[0]
            df_training = df.loc[:count - 1, :]
            df_test = df.loc[count:, :]
            df_test.reset_index(inplace=True, drop=True)
            # ##############################################################################################################

            if save:
                df_training.to_hdf(path_or_buf=df_training_cache_path, key='train')
                df_test.to_hdf(path_or_buf=df_test_cache_path, key='test')

        return df_training, df_test

    @staticmethod
    def get_validation_data(df_training, df_test, type, **params):
        """
        获取验证集
        :param df_training:
        :param df_test:
        :param params:
        :return:
        """
        df_validation = pd.DataFrame()

        if type is 'before':
            for row in range(df_test.shape[0]):
                df_validation = pd.concat(
                    [df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] - 1)]])

        elif type is 'after':
            for row in range(df_test.shape[0]):
                df_validation = pd.concat(
                    [df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] + 1)]])

        elif type is 'before_after':
            for row in range(df_test.shape[0]):
                df_validation = pd.concat(
                    [df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] - 1)]])
                df_validation = pd.concat(
                    [df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] + 1)]])

            df_validation.drop_duplicates(inplace=True, keep='first')

        ids = list(df_training['ID'])
        for id in list(df_validation['ID']):
            ids.remove(id)

        df_training = df_training[df_training['ID'].isin(ids)]

        return df_training, df_validation, df_test

    def generate_submition(self, prediction, X_test, validation_type, submit_or_not=True, **params):
        """
        生成提交结果
        :param prediction:
        :param X_test:
        :param params:
        :return:
        """
        import numpy as np

        predict_column = DefaultConfig.predict_column
        sub = pd.DataFrame(data=X_test['ID'].astype(int), columns=['ID'])

        print('before sum(prediction=1): ', len([i for i in prediction if i >= 0.5]))
        print('before sum(prediction=0): ', len([i for i in prediction if i < 0.5]))
        print('var(prediction): ', np.var(np.array(prediction)))
        if submit_or_not:
            result = []
            for i in prediction:
                if i >= 0.5 + 0.0 * np.var(np.array(prediction)):
                    result.append(1)
                else:
                    result.append(0)
            sub[predict_column] = result
            # 设置id为固定模式
            _, ids = self.get_df_test()
            # 设置成“category”数据类型
            sub['ID'] = sub['ID'].astype('category')
            # inplace = True，使 recorder_categories生效
            sub['ID'].cat.reorder_categories(ids, inplace=True)
            # inplace = True，使 df生效
            sub.sort_values('ID', inplace=True)
            print('after sum(prediction=1): ', sub[sub[predict_column] == 1].shape[0])
            # 规则
            sub.loc[sub[sub['ID'].isin(DefaultConfig.rule_ID_1)].index, predict_column] = 1
            print('rule_1 after sum(prediction=1): ', sub[sub[predict_column] == 1].shape[0])
            sub.loc[sub[sub['ID'].isin(DefaultConfig.rule_ID_0)].index, predict_column] = 0
            print('rule_0 after sum(prediction=1): ', sub[sub[predict_column] == 1].shape[0])

            print(list(sub[sub[predict_column] == 1].index))
            sub.to_csv(
                path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_rate_submit.csv',
                index=False, encoding='utf-8')

        sub = pd.DataFrame(data=X_test['ID'].astype(int), columns=['ID'])
        sub[predict_column] = prediction

        # 设置id为固定模式
        _, ids = self.get_df_test()
        # 设置成“category”数据类型
        sub['ID'] = sub['ID'].astype('category')
        # inplace = True，使 recorder_categories生效
        sub['ID'].cat.reorder_categories(ids, inplace=True)
        # inplace = True，使 df生效
        sub.sort_values('ID', inplace=True)

        sub.to_csv(
            path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_submit.csv',
            index=False, encoding='utf-8')
        return sub

    @staticmethod
    def merge(modeltypes, **params):
        """
        merge
        :param params:
        :return:
        """
        data = []
        predict_column = DefaultConfig.predict_column

        for modeltype in modeltypes:
            if modeltype is 'cbt':
                print('cbt')
                data.append(pd.read_csv(DefaultConfig.project_path + '/data/submit/' + modeltype + '_submit.csv',
                                        encoding='utf-8'))
            else:
                print('lgb')
                data.append(pd.read_csv(DefaultConfig.project_path + '/data/submit/' + modeltype + '_submit.csv',
                                        encoding='utf-8'))

        # data[0][predict_column] = data[0][predict_column] * 0 + data[1][predict_column] * 1.0

        result = []
        print(np.var(data[0][predict_column].values))
        for i in list(data[0][predict_column]):
            if i >= 0.5 + 0.1 * np.var(data[0][predict_column].values):
                result.append(1)
            else:
                result.append(0)

        data[0][predict_column] = result
        print('after sum(prediction=1): ', data[0][data[0][predict_column] == 1].shape[0])
        # 添加规则
        data[0].loc[
            data[0][data[0]['ID'].isin(DefaultConfig.rule_ID_1)].index, predict_column] = 1
        print('rule_1 after sum(prediction=1): ', data[0][data[0][predict_column] == 1].shape[0])
        data[0].loc[
            data[0][data[0]['ID'].isin(DefaultConfig.rule_ID_0)].index, predict_column] = 0
        print('rule_0 after sum(prediction=1): ', data[0][data[0][predict_column] == 1].shape[0])

        data[0].to_csv(path_or_buf=DefaultConfig.submition, index=False, encoding='utf-8')
