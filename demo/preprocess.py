import pandas as pd
import os

from utils.fill_values import FillValues
from utils.encoder import Encoder
from configuration.config import DefaultConfig
from utils.utils import Utils
from utils.convert import Convert


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

    def get_df_training(self, **params) -> pd.DataFrame:
        """
        获取df_training
        :param params:
        :return:
        """
        df = pd.read_csv(filepath_or_buffer=self.df_training_path, encoding='utf-8')

        df.sort_values(by='ID', inplace=True)

        df.rename(columns=dict(zip(list(df.columns), [i.strip("'").strip(' ') for i in list(df.columns)])),
                  inplace=True)

        df.reset_index(inplace=True, drop=True)

        return df

    def deal_Product_using_score(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        一、处理Product_using_score
        :param df:
        :param params:
        :return:
        """
        column = ["Product using score"]

        # 均值填充缺失值
        df = FillValues(df=df, columns=column).simplefill(fillmethod='mean')
        # 转换成整型
        df[column] = df[column].astype(int)

        return df

    def deal_User_area(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        二、处理User_area
        :param df:
        :param params:
        :return:
        """
        # column = ["User area"]
        #
        # # 类别编码
        # df = Encoder(df=df, columns=column).label_encoder()
        # # 转换成整型
        # df[column] = df[column].astype(int)
        #
        # return df

        import numpy as np

        df["User area"] = df["User area"].apply(lambda x: 0 if x is np.nan else x)
        df["User area"] = df["User area"].apply(lambda x: 1 if x == "Taipei" else x)
        df["User area"] = df["User area"].apply(lambda x: 2 if x == "Taichung" else x)
        df["User area"] = df["User area"].apply(lambda x: 3 if x == "Tainan" else x)

        return df

    def deal_gender(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        三、处理gender
        :param df:
        :param params:
        :return:
        """
        # column = ["gender"]
        #
        # # 类别编码
        # df = Encoder(df=df, columns=column).label_encoder()
        # # 转换成整型
        # df[column] = df[column].astype(int)
        #
        # return df

        import numpy as np

        df['gender'] = df['gender'].apply(lambda x: 0 if x is np.nan else x)
        df['gender'] = df['gender'].apply(lambda x: 1 if x == "Male" else x)
        df['gender'] = df['gender'].apply(lambda x: 2 if x == "Female" else x)

        return df

    def deal_age(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        四、处理age
        :param df:
        :param params:
        :return:
        """
        column = ["age"]

        # 均值填充缺失值
        df = FillValues(df=df, columns=column).simplefill(fillmethod='mean')
        # 转换成整型
        df[column] = df[column].astype(int)

        return df

    def deal_Cumulative_using_time(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        五、处理Cumulative_using_time
        :param df:
        :param params:
        :return:
        """
        column = ["Cumulative using time"]

        # 众数填充缺失值
        df = FillValues(df=df, columns=column).simplefill(fillmethod='mode')
        # 转换成整型
        df[column] = df[column].astype(int)

        return df

    def deal_Point_balance(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        六、处理Point_balance
        :param df:
        :param params:
        :return:
        """
        column = ["Point balance"]

        # 均值填充缺失值
        df = FillValues(df=df, columns=column).simplefill(fillmethod='mean')

        return df

    def deal_Product_service_usage(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        七、处理Product_service_usage
        :param df:
        :param params:
        :return:
        """
        column = ["Product service usage"]

        # 众数填充缺失值
        df = FillValues(df=df, columns=column).simplefill(fillmethod='mode')
        # 转换成整型
        df[column] = df[column].astype(int)

        return df

    def deal_Pay_a_monthly_fee_by_credit_card(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        八、处理Pay_a_monthly_fee_by_credit_card
        :param df:
        :param params:
        :return:
        """
        column = ["Pay a monthly fee by credit card"]

        # 众数填充缺失值
        df = FillValues(df=df, columns=column).simplefill(fillmethod='mode')
        # 转换成整型
        df[column] = df[column].astype(int)

        return df

    def deal_Active_user(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        九、处理Active_user
        :param df:
        :param params:
        :return:
        """
        column = ["Active user"]

        # 众数填充缺失值
        df = FillValues(df=df, columns=column).simplefill(fillmethod='mode')
        # 转换成整型
        df[column] = df[column].astype(int)

        return df

    def deal_Estimated_salary(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        十、处理Estimated_salary
        :param df:
        :param params:
        :return:
        """
        column = ["Estimated salary"]

        # 均值填充缺失值
        df = FillValues(df=df, columns=column).simplefill(fillmethod='mean')

        return df

    def main(self, save=True):
        """
        数据预处理
        :return:
        """
        import numpy as np

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

            # 处理步骤
            deal_step = [self.deal_Product_using_score, self.deal_User_area, self.deal_gender, self.deal_age,
                         self.deal_Cumulative_using_time, self.deal_Point_balance, self.deal_Product_service_usage,
                         self.deal_Pay_a_monthly_fee_by_credit_card, self.deal_Active_user, self.deal_Estimated_salary]

            # 先将'?'替换为nan
            df_training.replace('?', np.nan, inplace=True)
            df_test.replace('?', np.nan, inplace=True)

            if DefaultConfig.select_model is 'lgb':
                for step in deal_step:
                    # df_test
                    df_test = step(df_test)
                    # df_training
                    df_training = step(df_training)

                # 合并
                df = pd.concat([df_training, df_test], ignore_index=True, axis=0)

            elif DefaultConfig.select_model is 'cbt':
                # 合并
                df = pd.concat([df_training, df_test], ignore_index=True, axis=0)

                for step in deal_step:
                    # df
                    df = step(df)

            # 类别列
            for c_col in ['age']:
                # 数值列
                for n_col in ["Estimated salary", "Point balance"]:
                    df[n_col + '_groupby_' + c_col + '_mean_ratio'] = df[n_col] / (
                        df[c_col].map(df[n_col].groupby(df[c_col]).mean()))

            # ################################################################ 填充缺失值
            # # knn
            # df = FillValues(df=df, columns=list(df.columns)).knn_missing_value()
            # # MICE
            # df = FillValues(df=df, columns=list(df.columns)).mice_missing_value()
            # # IterForest
            # df = FillValues(df=df, columns=list(df.columns)).iterforest_missing_value()
            # # EM
            # df = FillValues(df=df, columns=list(df.columns)).em_missing_value()

            # 整数变量需要转化为整数
            # df[DefaultConfig.int_columns] = df[DefaultConfig.int_columns].astype(int)
            # df[DefaultConfig.float_columns] = df[DefaultConfig.float_columns].astype(float)

            # ################################################################# 变换
            # yeo_johnson变换
            # df = Convert(df, DefaultConfig.float_columns).yeo_johnson()
            # log1p变换
            # df = Convert(df, DefaultConfig.float_columns).log1p()

            # 整数变量需要转化为整数
            df[DefaultConfig.int_columns] = df[DefaultConfig.int_columns].astype(int)
            # df[DefaultConfig.float_columns] = df[DefaultConfig.float_columns].astype(float)

            count = df_training.shape[0]
            df_training = df.loc[:count - 1, :]
            df_test = df.loc[count:, :]
            df_test.reset_index(inplace=True, drop=True)
            # ##############################################################################################################

            # 效果不好待优化
            # df = pd.concat([df_training, df_test], axis=0, ignore_index=True)
            # df, new_columns = one_hot_encoder(df, categorical_columns=DefaultConfig.categorical_columns)
            # print('before: (%d, %d)' % (df_training.shape[0], df_test.shape[0]))
            # print('before: (%d)' % (len(df_training.columns)))
            # count = df_training.shape[0]
            # print('count', count)
            # df_training = df.loc[:count - 1, :]
            # df_test = df.loc[count:, :]
            # df_test.reset_index(inplace=True, drop=True)
            # print('after: (%d, %d)' % (df_training.shape[0], df_test.shape[0]))
            # print('after: (%d)' % (len(df_training.columns)))

            if save:
                df_training.to_hdf(path_or_buf=df_training_cache_path, key='train')
                df_test.to_hdf(path_or_buf=df_test_cache_path, key='test')

        return df_training, df_test

    @staticmethod
    def get_validation_data(df_training, df_test, **params):
        """
        获取验证集
        :param df_training:
        :param df_test:
        :param params:
        :return:
        """
        df_validation = pd.DataFrame()

        if DefaultConfig.before_after is 'before':
            for row in range(df_test.shape[0]):
                df_validation = pd.concat(
                    [df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] - 1)]])

        elif DefaultConfig.before_after is 'after':
            for row in range(df_test.shape[0]):
                df_validation = pd.concat(
                    [df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] + 1)]])

        elif DefaultConfig.before_after is 'before_after':
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
            sub['Predicted_Results'] = result
            # 设置id为固定模式
            _, ids = self.get_df_test()
            # 设置成“category”数据类型
            sub['ID'] = sub['ID'].astype('category')
            # inplace = True，使 recorder_categories生效
            sub['ID'].cat.reorder_categories(ids, inplace=True)
            # inplace = True，使 df生效
            sub.sort_values('ID', inplace=True)
            print('after sum(prediction=1): ', len([i for i in result if i == 1]))
            print('before sum(prediction=0): ', len([i for i in result if i == 0]))
            sub.to_csv(path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_' +
                                   validation_type + '_rate_submit.csv', index=False, encoding='utf-8')

        sub = pd.DataFrame(data=X_test['ID'].astype(int), columns=['ID'])
        sub['Predicted_Results'] = prediction

        # 设置id为固定模式
        _, ids = self.get_df_test()
        # 设置成“category”数据类型
        sub['ID'] = sub['ID'].astype('category')
        # inplace = True，使 recorder_categories生效
        sub['ID'].cat.reorder_categories(ids, inplace=True)
        # inplace = True，使 df生效
        sub.sort_values('ID', inplace=True)

        sub.to_csv(path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_' +
                               validation_type + '_submit.csv', index=False, encoding='utf-8')
        return sub

    @staticmethod
    def merge(**params):
        """
        merge
        :param params:
        :return:
        """
        if DefaultConfig.merge_type is 'before_after':
            before = pd.read_csv(DefaultConfig.lgb_before_submit, encoding='utf-8')
            after = pd.read_csv(DefaultConfig.lgb_after_submit, encoding='utf-8')

            before['Predicted_Results'] = (before['Predicted_Results'] + after['Predicted_Results']) / 2

        elif DefaultConfig.merge_type is 'lgb_cbt':
            lgb = pd.read_csv(DefaultConfig.lgb_before_submit, encoding='utf-8')
            cbt = pd.read_csv(DefaultConfig.cbt_before_submit, encoding='utf-8')

            lgb['Predicted_Results'] = (lgb['Predicted_Results'] * 0.05 + cbt['Predicted_Results'] * 0.95)

        elif DefaultConfig.merge_type is 'lgb_xgb':
            lgb_before = pd.read_csv(DefaultConfig.lgb_before_submit, encoding='utf-8')
            xgb_before_after = pd.read_csv(DefaultConfig.xgb_before_after_submit, encoding='utf-8')

            lgb_before['Predicted_Results'] = 0.9 * lgb_before['Predicted_Results'] + 0.1 * xgb_before_after[
                'Predicted_Results']

        elif DefaultConfig.merge_type is 'lgb_cbt_xgb':
            lgb_before = pd.read_csv(DefaultConfig.lgb_before_submit, encoding='utf-8')
            cbt_after = pd.read_csv(DefaultConfig.cbt_after_submit, encoding='utf-8')
            xgb_before_after = pd.read_csv(DefaultConfig.xgb_before_after_submit, encoding='utf-8')

            lgb_before['Predicted_Results'] = 0.4 * lgb_before['Predicted_Results'] + 0.5 * cbt_after[
                'Predicted_Results'] + \
                                              0.1 * xgb_before_after['Predicted_Results']

        result = []
        for i in list(lgb['Predicted_Results']):
            if i >= 0.5:
                result.append(1)
            else:
                result.append(0)

        print('sum(1): ', sum(result))
        lgb['Predicted_Results'] = result

        lgb.to_csv(path_or_buf=DefaultConfig.submition, index=False, encoding='utf-8')
