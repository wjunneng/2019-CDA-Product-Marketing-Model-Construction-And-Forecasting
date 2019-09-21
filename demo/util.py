from config import DefaultConfig

import pandas as pd


def get_df_test(**params) -> (pd.DataFrame, list):
    """
    获取df_test
    :param params:
    :return:
    """
    df = pd.read_csv(filepath_or_buffer=DefaultConfig.df_test_path, encoding='utf-8')

    ids = list(df['ID'])

    df.sort_values(by='ID', inplace=True)

    df.reset_index(inplace=True, drop=True)

    return df, ids


def get_df_training(**params) -> pd.DataFrame:
    """
    获取df_test
    :param params:
    :return:
    """
    df = pd.read_csv(filepath_or_buffer=DefaultConfig.df_training_path, encoding='utf-8')

    df.sort_values(by='ID', inplace=True)

    df.reset_index(inplace=True, drop=True)

    return df


def deal_Product_using_score(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    一、处理Product_using_score
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    # 均值填充
    df["'Product using score'"] = df["'Product using score'"].apply(lambda x: np.nan if x == '?' else x)
    df["'Product using score'"] = df["'Product using score'"].astype(float)
    df["'Product using score'"].fillna(df["'Product using score'"].mean(), inplace=True)
    df["'Product using score'"] = df["'Product using score'"].astype(int)

    return df


def deal_User_area(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    二、处理User_area
    :param df:
    :param params:
    :return:
    """
    df["'User area'"] = df["'User area'"].apply(lambda x: 0 if x == "?" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 1 if x == "Taipei" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 2 if x == "Taichung" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 3 if x == "Tainan" else x)

    return df


def deal_gender(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    三、处理gender
    :param df:
    :param params:
    :return:
    """
    df['gender'] = df['gender'].apply(lambda x: 0 if x == "?" else x)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == "Male" else x)
    df['gender'] = df['gender'].apply(lambda x: 2 if x == "Female" else x)

    return df


def deal_age(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    四、处理age
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    # 均值填充
    df["age"] = df["age"].apply(lambda x: np.nan if x == '?' else x)
    df["age"] = df["age"].astype(float)
    df["age"].fillna(df["age"].mean(), inplace=True)
    df["age"] = df["age"].astype(int)

    return df


def deal_Cumulative_using_time(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    五、处理Cumulative_using_time
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    # 众数填充
    df["'Cumulative using time'"] = df["'Cumulative using time'"].apply(lambda x: np.nan if x == '?' else x)
    df["'Cumulative using time'"] = df["'Cumulative using time'"].astype(float)
    df["'Cumulative using time'"].fillna(df["'Cumulative using time'"].mode()[0], inplace=True)
    df["'Cumulative using time'"] = df["'Cumulative using time'"].astype(int)

    return df


def deal_Point_balance(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    六、处理Point_balance
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    # 均值填充
    df["'Point balance'"] = df["'Point balance'"].apply(lambda x: np.nan if x == '?' else x)
    df["'Point balance'"] = df["'Point balance'"].astype(float)
    df["'Point balance'"].fillna(df["'Point balance'"].mean(), inplace=True)

    return df


def deal_Product_service_usage(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    七、处理Product_service_usage
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    # 众数填充
    df["'Product service usage'"] = df["'Product service usage'"].apply(lambda x: np.nan if x == '?' else x)
    df["'Product service usage'"] = df["'Product service usage'"].astype(float)
    df["'Product service usage'"].fillna(df["'Product service usage'"].mode()[0], inplace=True)
    df["'Product service usage'"] = df["'Product service usage'"].astype(int)

    return df


def deal_Pay_a_monthly_fee_by_credit_card(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    八、处理Pay_a_monthly_fee_by_credit_card
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    # 众数填充
    df["'Pay a monthly fee by credit card'"] = df["'Pay a monthly fee by credit card'"].apply(
        lambda x: np.nan if x == '?' else x)
    df["'Pay a monthly fee by credit card'"] = df["'Pay a monthly fee by credit card'"].astype(float)
    df["'Pay a monthly fee by credit card'"].fillna(df["'Pay a monthly fee by credit card'"].mode()[0], inplace=True)
    df["'Pay a monthly fee by credit card'"] = df["'Pay a monthly fee by credit card'"].astype(int)

    return df


def deal_Active_user(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    九、处理Active_user
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    # 众数填充
    df["'Active user'"] = df["'Active user'"].apply(lambda x: np.nan if x == '?' else x)
    df["'Active user'"] = df["'Active user'"].astype(float)
    df["'Active user'"].fillna(df["'Active user'"].mode()[0], inplace=True)
    df["'Active user'"] = df["'Active user'"].astype(int)

    return df


def deal_Estimated_salary(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    十、处理Estimated_salary
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    # 均值填充
    df["' Estimated salary'"] = df["' Estimated salary'"].apply(lambda x: np.nan if x == '?' else x)
    df["' Estimated salary'"] = df["' Estimated salary'"].astype(float)
    df["' Estimated salary'"].fillna(df["' Estimated salary'"].mean(), inplace=True)

    return df


def get_validation_data(df_training, df_test, type='after', **params):
    """
    获取验证集
    :param df_training:
    :param df_test:
    :param params:
    :return:
    """
    df_validation = pd.DataFrame()

    if type is 'after':
        for row in range(df_test.shape[0]):
            df_validation = pd.concat([df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] + 1)]])

    elif type is 'before':
        for row in range(df_test.shape[0]):
            df_validation = pd.concat([df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] - 1)]])

    ids = list(df_training['ID'])
    for id in list(df_validation['ID']):
        ids.remove(id)

    df_training = df_training[df_training['ID'].isin(ids)]

    return df_training, df_validation, df_test


def preprocess(save=True, **params):
    """
    数据预处理
    :param params:
    :return:
    """
    import os

    df_training_path = DefaultConfig.df_training_cache_path
    df_test_path = DefaultConfig.df_test_cache_path

    if os.path.exists(df_training_path) and os.path.exists(df_test_path) and DefaultConfig.df_no_replace:
        df_training = reduce_mem_usage(pd.read_hdf(path_or_buf=df_training_path, key='df_training', mode='r'))
        df_test = reduce_mem_usage(pd.read_hdf(path_or_buf=df_test_path, key='df_test', mode='r'))
    else:
        # 获取原数据
        df_test, _ = get_df_test()
        df_training = get_df_training()

        # 一、处理Product_using_score   产品使用分数
        df_test = deal_Product_using_score(df_test)
        df_training = deal_Product_using_score(df_training)

        # 二、处理User_area     用户地区
        df_test = deal_User_area(df_test)
        df_training = deal_User_area(df_training)

        # 三、处理gender    性别
        df_test = deal_gender(df_test)
        df_training = deal_gender(df_training)

        # 四、处理age  年龄
        df_test = deal_age(df_test)
        df_training = deal_age(df_training)

        # 五、处理Cumulative_using_time     使用累计时间
        df_test = deal_Cumulative_using_time(df_test)
        df_training = deal_Cumulative_using_time(df_training)

        # 六、处理Point_balance     点数余额
        df_test = deal_Point_balance(df_test)
        df_training = deal_Point_balance(df_training)

        # 七、处理Product_service_usage 产品服务实用量
        df_test = deal_Product_service_usage(df_test)
        df_training = deal_Product_service_usage(df_training)

        # 八、处理Pay_a_monthly_fee_by_credit_card  是否使用信用卡付月费
        df_test = deal_Pay_a_monthly_fee_by_credit_card(df_test)
        df_training = deal_Pay_a_monthly_fee_by_credit_card(df_training)

        # 九、处理Active_user   是否为活跃用户
        df_test = deal_Active_user(df_test)
        df_training = deal_Active_user(df_training)

        # 十、处理Estimated_salary  估计薪资
        df_test = deal_Estimated_salary(df_test)
        df_training = deal_Estimated_salary(df_training)

        if save:
            df_training.to_hdf(path_or_buf=df_training_path, key='df_training')
            df_test.to_hdf(path_or_buf=df_test_path, key='df_test')

    return df_training, df_test


def reduce_mem_usage(df, verbose=True):
    """
    减少内存消耗
    :param df:
    :param verbose:
    :return:
    """
    import numpy as np

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def lgb_model(X_train, X_test, **params):
    """
    lgb 模型
    :param new_train:
    :param y:
    :param new_test:
    :param columns:
    :param params:
    :return:
    """
    import gc
    import numpy as np
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score, f1_score
    import matplotlib.pyplot as plt

    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat)
        return 'f1', f1_score(y_true, y_hat), True

    # y_train
    y_train = X_train.loc[:, DefaultConfig.label_column]
    # 特征重要性
    feature_importance = None
    # 线下验证
    oof = np.zeros((X_train.shape[0]))
    # 线上结论
    prediction = np.zeros((X_test.shape[0]))

    seeds = [42, 2019, 223344, 2019 * 2 + 1024, 332232111]
    num_model_seed = 1

    print(DefaultConfig.select_model + ' start training...')

    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'num_leaves': 100,
        'verbose': -1,
        'max_depth': -1
    }

    # 寻找最优的num_leaves
    # min_merror = np.inf
    # for num_leaves in [150, 200, 250, 300, 500, 1000]:
    #     params["num_leaves"] = num_leaves
    #
    #     cv_results = lgb.cv(params=params,
    #                         train_set=lgb.Dataset(X_train, label=y_train),
    #                         num_boost_round=2000,
    #                         stratified=False,
    #                         nfold=5,
    #                         verbose_eval=50,
    #                         seed=23,
    #                         early_stopping_rounds=20)
    #
    #     mean_error = min(cv_results['multi_logloss-mean'])
    #
    #     if mean_error < min_merror:
    #         min_merror = mean_error
    #         params["num_leaves"] = num_leaves
    #
    # print('num_leaves: ', num_leaves)

    for model_seed in range(num_model_seed):
        print('模型 ', model_seed + 1, ' 开始训练')
        oof_lgb = np.zeros((X_train.shape[0]))
        prediction_lgb = np.zeros((X_test.shape[0]))

        # 存放特征重要性
        feature_importance_df = pd.DataFrame()
        # splits_type
        splits_type = ['before']
        for index, value in enumerate(splits_type):
            print('第 ' + str(index) + ' 折')

            # 获取验证集
            df_training, df_validation, df_test = get_validation_data(df_training=X_train, df_test=X_test, type=value)

            # 分割
            train_x, test_x, train_y, test_y = df_training.drop(labels=DefaultConfig.label_column, axis=1), \
                                               df_validation.drop(labels=DefaultConfig.label_column, axis=1), \
                                               df_training.loc[:, DefaultConfig.label_column], \
                                               df_validation.loc[:, DefaultConfig.label_column]

            train_data = lgb.Dataset(train_x, label=train_y)
            validation_data = lgb.Dataset(test_x, label=test_y)

            gc.collect()
            bst = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=10000, verbose_eval=1000,
                            early_stopping_rounds=2019, feval=lgb_f1_score)
            oof_lgb[test_x.index] += bst.predict(test_x)
            prediction_lgb += bst.predict(X_test.drop(DefaultConfig.label_column, axis=1)) / len(
                splits_type)
            gc.collect()

        oof += oof_lgb / num_model_seed
        prediction += prediction_lgb / num_model_seed
        print('logloss', log_loss(pd.get_dummies(y_train).values, oof_lgb))
        # 线下auc评分
        print('the roc_auc_score for train:', roc_auc_score(y_train, oof_lgb))

        if feature_importance is None:
            feature_importance = None
        else:
            feature_importance += feature_importance_df
            feature_importance['importance'] /= num_model_seed

    print('logloss', log_loss(pd.get_dummies(y_train).values, oof))
    print('ac', roc_auc_score(y_train, oof))

    if feature_importance is not None:
        feature_importance.to_hdf(path_or_buf=DefaultConfig.lgb_feature_cache_path, key='lgb')
        # 读取feature_importance_df
        feature_importance_df = reduce_mem_usage(
            pd.read_hdf(path_or_buf=DefaultConfig.lgb_feature_cache_path, key='lgb', mode='r'))

        plt.figure(figsize=(8, 8))
        # 按照flod分组
        group = feature_importance_df.groupby(by=['fold'])

        result = []
        for key, value in group:
            value = value[['feature', 'importance']]

            result.append(value)

        result = pd.concat(result)
        print(result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(40))
        # 5折数据取平均值
        result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(40).plot.barh()
        plt.show()

    sub = pd.DataFrame(data=X_test['ID'].astype(int), columns=['ID'])

    result = []
    for i in prediction:
        if i >= 0.5:
            result.append(1)
        else:
            result.append(0)

    sub['Predicted_Results'] = result

    # 设置id为固定模式
    _, ids = get_df_test()
    # 设置成“category”数据类型
    sub['ID'] = sub['ID'].astype('category')
    # inplace = True，使 recorder_categories生效
    sub['ID'].cat.reorder_categories(ids, inplace=True)
    # inplace = True，使 df生效
    sub.sort_values('ID', inplace=True)

    sub.to_csv(path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_submit.csv',
               index=False, encoding='utf-8')
    return sub

#
# if __name__ == '__main__':
#     df_training, df_validation, df_test = preprocess()
