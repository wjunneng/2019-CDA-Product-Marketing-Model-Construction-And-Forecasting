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
    df["'Product using score'"].replace('?', np.nan, inplace=True)
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
    import numpy as np

    df["'User area'"].replace('?', np.nan, inplace=True)
    df["'User area'"] = df["'User area'"].apply(lambda x: 0 if x is np.nan else x)
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
    import numpy as np

    df['gender'].replace('?', np.nan, inplace=True)
    df['gender'] = df['gender'].apply(lambda x: 0 if x is np.nan else x)
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
    df["age"].replace('?', np.nan, inplace=True)
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
    df["'Cumulative using time'"].replace('?', np.nan, inplace=True)
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
    df["'Point balance'"].replace('?', np.nan, inplace=True)
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
    df["'Product service usage'"].replace('?', np.nan, inplace=True)
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
    df["'Pay a monthly fee by credit card'"].replace('?', np.nan, inplace=True)
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
    df["'Active user'"].replace('?', np.nan, inplace=True)
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
    df["' Estimated salary'"].replace('?', np.nan, inplace=True)
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

    if type is 'before':
        for row in range(df_test.shape[0]):
            df_validation = pd.concat([df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] - 1)]])

    elif type is 'after':
        for row in range(df_test.shape[0]):
            df_validation = pd.concat([df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] + 1)]])

    elif type is 'before_after':
        for row in range(df_test.shape[0]):
            df_validation = pd.concat([df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] - 1)]])
            df_validation = pd.concat([df_validation, df_training[df_training['ID'] == (df_test.ix[row, 'ID'] + 1)]])

        df_validation.drop_duplicates(inplace=True, keep='first')

    ids = list(df_training['ID'])
    for id in list(df_validation['ID']):
        ids.remove(id)

    df_training = df_training[df_training['ID'].isin(ids)]

    return df_training, df_validation, df_test


def one_hot_encoder(df, categorical_columns, nan_as_category=True):
    """
    ont hot 编码
    :param df:
    :param nan_as_category:
    :return:
    """
    original_columns = list(df.columns)
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def add_virtual_features(df, **params):
    """
    添加虚拟变量
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    df.replace('?', np.nan, inplace=True)

    for column in list(DefaultConfig.columns):
        if column in list(df.columns) and df[column].isnull().sum() != 0:
            tmp = column + ' Virtual'
            df[tmp] = df[column].copy()
            df.loc[(df[tmp].notnull()), tmp] = 0
            df.loc[(df[tmp].isnull()), tmp] = 1

    return df


def KNN_missing_value(df, **params):
    """
    KNN算法可用于在多维空间中将点与其最近的k个邻居进行匹配。 它可以用于连续, 离散, 序数和分类的数据, 这对处理所有类型的缺失数据特别有用。
    使用KNN作为缺失值的假设是, 基于其他变量, 点值可以近似于与其最接近的点的值。
    :param df:
    :param params:
    :return:
    """
    import numpy as np
    from ycimpute.imputer import knnimput

    label_feature = df[DefaultConfig.label_column]
    del df[DefaultConfig.label_column]

    df['gender'] = df['gender'].apply(lambda x: 0 if x == "Male" else x)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == "Female" else x)

    df["'User area'"] = df["'User area'"].apply(lambda x: 1 if x == "Taipei" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 2 if x == "Taichung" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 3 if x == "Tainan" else x)

    df.replace('?', np.nan, inplace=True)
    df = df.astype(np.float)
    values = knnimput.KNN(k=16).complete(df.values)
    df = pd.DataFrame(data=values, index=None, columns=df.columns)
    df[DefaultConfig.label_column] = label_feature
    return df


def Mice_missing_value(df, **params):
    """
    算法基于线性回归。mice能填充如连续型,二值型,离散型等混
    合数据并保持数据一致性
    :param df:
    :param params:
    :return:
    """
    import numpy as np
    from ycimpute.imputer import mice

    label_feature = df[DefaultConfig.label_column]
    del df[DefaultConfig.label_column]

    df['gender'] = df['gender'].apply(lambda x: 0 if x == "Male" else x)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == "Female" else x)

    df["'User area'"] = df["'User area'"].apply(lambda x: 1 if x == "Taipei" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 2 if x == "Taichung" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 3 if x == "Tainan" else x)

    df.replace('?', np.nan, inplace=True)
    df = df.astype(np.float)
    values = mice.MICE().complete(df.values)
    df = pd.DataFrame(data=values, index=None, columns=df.columns)
    df[DefaultConfig.label_column] = label_feature
    return df


def IterForest_missing_value(df, **params):
    """
    是基于随机森林的一种算法,能够很好的填充连续
    型,离散型相混合的缺失数据。尤其对于各变量之间出存在相关
    性的数据集表现不错
    """
    import numpy as np
    from ycimpute.imputer import iterforest

    label_feature = df[DefaultConfig.label_column]
    del df[DefaultConfig.label_column]

    df['gender'] = df['gender'].apply(lambda x: 0 if x == "Male" else x)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == "Female" else x)

    df["'User area'"] = df["'User area'"].apply(lambda x: 1 if x == "Taipei" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 2 if x == "Taichung" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 3 if x == "Tainan" else x)

    df.replace('?', np.nan, inplace=True)
    df = df.astype(np.float)
    values = iterforest.IterImput().complete(df.values)
    df = pd.DataFrame(data=values, index=None, columns=df.columns)
    df[DefaultConfig.label_column] = label_feature
    return df


def EM_missing_value(df, **params):
    """
    EM基于高斯分布能处理混合数据,但在连续型数据上表现的相对较好。在多种数
    据缺失机制下,EM相对于其他方法有着较好的表现。
    """
    import numpy as np
    from ycimpute.imputer import EM

    label_feature = df[DefaultConfig.label_column]
    del df[DefaultConfig.label_column]

    df['gender'] = df['gender'].apply(lambda x: 0 if x == "Male" else x)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == "Female" else x)

    df["'User area'"] = df["'User area'"].apply(lambda x: 1 if x == "Taipei" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 2 if x == "Taichung" else x)
    df["'User area'"] = df["'User area'"].apply(lambda x: 3 if x == "Tainan" else x)

    df["'Point balance'"].replace('0', np.nan, inplace=True)
    df.replace('?', np.nan, inplace=True)
    df = df.astype(np.float)
    values = EM().complete(df.values)
    df = pd.DataFrame(data=values, index=None, columns=df.columns)
    df[DefaultConfig.label_column] = label_feature
    return df


def smote(X_train, save=True, **params):
    """
    过采样+欠采样
    :param X_train:
    :param y_train:
    :param params:
    :return:
    """
    from collections import Counter
    from imblearn.over_sampling import SMOTE

    y_train = X_train[DefaultConfig.label_column]
    del X_train[DefaultConfig.label_column]

    # smote 算法
    smote = SMOTE(ratio={0: 4200, 1: 1000}, n_jobs=10, kind='svm')
    train_X, train_y = smote.fit_sample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(train_y))
    X_train = pd.DataFrame(data=train_X, columns=X_train.columns)
    X_train[DefaultConfig.label_column] = train_y

    return X_train


def preprocess(save=True, **params):
    """
    数据预处理
    :param params:
    :return:
    """
    import os
    import numpy as np
    from sklearn.preprocessing import FunctionTransformer

    df_training_path = DefaultConfig.df_training_cache_path
    df_test_path = DefaultConfig.df_test_cache_path

    if os.path.exists(df_training_path) and os.path.exists(df_test_path) and DefaultConfig.df_no_replace:
        df_training = reduce_mem_usage(pd.read_hdf(path_or_buf=df_training_path, key='df_training', mode='r'))
        df_test = reduce_mem_usage(pd.read_hdf(path_or_buf=df_test_path, key='df_test', mode='r'))
    else:
        # 获取原数据
        df_test, _ = get_df_test()
        df_training = get_df_training()

        # ################################################ 添加虚拟变量
        # 添加虚拟变量
        # df_training = add_virtual_features(df_training)
        # df_test = add_virtual_features(df_test)

        # ################################################# 常规处理
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

        # # 六、处理Point_balance     点数余额
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

        # # 十、处理Estimated_salary  估计薪资
        df_test = deal_Estimated_salary(df_test)
        df_training = deal_Estimated_salary(df_training)

        # ##############################################################################################################
        df = pd.concat([df_training, df_test], ignore_index=True, axis=0)

        # # 一、处理Product_using_score   产品使用分数
        # df = deal_Product_using_score(df)
        #
        # # 二、处理User_area     用户地区
        # df = deal_User_area(df)
        #
        # # 三、处理gender    性别
        # df = deal_gender(df)
        #
        # # 四、处理age  年龄
        # df = deal_age(df)
        #
        # # 五、处理Cumulative_using_time     使用累计时间
        # df = deal_Cumulative_using_time(df)
        #
        # # # 六、处理Point_balance     点数余额
        # df = deal_Point_balance(df)
        #
        # # 七、处理Product_service_usage 产品服务实用量
        # df = deal_Product_service_usage(df)
        #
        # # 八、处理Pay_a_monthly_fee_by_credit_card  是否使用信用卡付月费
        # df = deal_Pay_a_monthly_fee_by_credit_card(df)
        #
        # # 九、处理Active_user   是否为活跃用户
        # df = deal_Active_user(df)
        #
        # # # 十、处理Estimated_salary  估计薪资
        # df = deal_Estimated_salary(df)

        # 3.数值列
        # 类别列
        for c_col in ['age']:
            # 数值列
            for n_col in ["' Estimated salary'", "'Point balance'"]:
                df[n_col + '_groupby_' + c_col + '_mean_ratio'] = df[n_col] / (
                    df[c_col].map(df[n_col].groupby(df[c_col]).mean()))

        # ################################################################ KNN
        # df = KNN_missing_value(df)
        # ################################################################ MICE
        # df = Mice_missing_value(df)
        # ################################################################ IterForest
        # df = IterForest_missing_value(df)
        # ################################################################ EM
        # df = EM_missing_value(df)

        # ########################################### 要进行yeo-johnson变换的特征列
        # print('进行yeo-johnson变换的特征列：')
        # print(DefaultConfig.float_columns)
        # pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
        # df[DefaultConfig.float_columns] = pt.fit_transform(df[DefaultConfig.float_columns])

        ########################################### 要进行对数变换的特征列
        # print('进行对数变换的特征列：')
        # print(DefaultConfig.float_columns)
        # transformer = FunctionTransformer(np.log1p)
        # df[DefaultConfig.float_columns] = transformer.fit_transform(df[DefaultConfig.float_columns].values)

        # 整数变量需要转化为整数
        # df[DefaultConfig.int_columns] = df[DefaultConfig.int_columns].astype(int)

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


def get_validation(X_train, X_valid, y_train, y_valid, categorical_columns, random_state=42, **params):
    """
    返回验证集
    :param X_train:
    :param y_train:
    :param params:
    :return:
    """
    import pandas as pd
    import lightgbm as lgb

    X_train[DefaultConfig.label_column] = list(y_train)
    X_valid[DefaultConfig.label_column] = list(y_valid)
    X_train['Is_Test'] = 0
    X_valid['Is_Test'] = 1

    # 将 Train 和 Test 合成一个数据集。Quality_label是数据本来的Y，所以剔除。
    df_adv = pd.concat([X_train, X_valid])

    adv_data = lgb.Dataset(data=df_adv.drop('Is_Test', axis=1), label=df_adv.loc[:, 'Is_Test'])

    # 定义模型参数
    params = {
        'boosting_type': 'gbdt',
        'colsample_bytree': 1,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_samples': 100,
        'min_child_weight': 1,
        'num_leaves': 36,
        'objective': 'binary',
        'random_state': random_state,
        'subsample': 1.0,
        'subsample_freq': 0,
        'metric': 'auc',
        'num_threads': 10,
        'boost_from_average': False,
        'verbose': -1
    }

    # 交叉验证
    adv_cv_results = lgb.cv(
        params,
        adv_data,
        num_boost_round=1000,
        nfold=5,
        verbose_eval=False,
        categorical_feature=categorical_columns,
        early_stopping_rounds=200,
        seed=random_state
    )

    print('交叉验证中最优的AUC为 {:.5f}，对应的标准差为{:.5f}.'.format(
        adv_cv_results['auc-mean'][-1], adv_cv_results['auc-stdv'][-1]))

    print('模型最优的迭代次数为{}.'.format(len(adv_cv_results['auc-mean'])))

    params['n_estimators'] = len(adv_cv_results['auc-mean'])

    model_adv = lgb.LGBMClassifier(**params)
    model_adv.fit(df_adv.drop('Is_Test', axis=1), df_adv.loc[:, 'Is_Test'])

    preds_adv = model_adv.predict_proba(df_adv.drop('Is_Test', axis=1))[:, 1]

    del X_train['Is_Test']
    del X_valid['Is_Test']

    # #################################### 效果不好
    # X_train['weight'] = preds_adv[:len(X_train)]
    # X_valid['weight'] = preds_adv[len(X_train):]
    #
    # X = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
    # X.sort_values(by='weight', inplace=True, ascending=True)
    # X.reset_index(drop=True, inplace=True)
    #
    # return X.drop('weight', axis=1).loc[len(X_valid):, :], X.drop('weight', axis=1).loc[:len(X_valid), :], X.loc[len(
    #     X_valid):, 'weight'], X.loc[:len(X_valid), 'weight']

    # #################################### 不排序
    return X_train, X_valid, preds_adv[:len(X_train)], preds_adv[len(X_train):]


def lgb_model(X_train, X_test, validation_type, **params):
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
    from boruta import BorutaPy

    def lgb_f1_score(y_hat, data):
        y_true = list(data.get_label())
        y_hat = np.round(y_hat)
        return 'f1', f1_score(y_true, y_hat), True

    # y_train
    y_train = list(X_train.loc[:, DefaultConfig.label_column])
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
        'learning_rate': 0.005,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'num_leaves': 100,
        'verbose': -1,
        'max_depth': -1
    }

    for model_seed in range(num_model_seed):
        print('模型 ', model_seed + 1, ' 开始训练')
        oof_lgb = np.zeros((X_train.shape[0]))
        prediction_lgb = np.zeros((X_test.shape[0]))

        # 存放特征重要性
        feature_importance_df = pd.DataFrame()
        for index, value in enumerate(validation_type):
            print('第 ' + str(index) + ' 折')

            # 获取验证集
            df_training, df_validation, df_test = get_validation_data(df_training=X_train, df_test=X_test, type=value)
            # 0/1数目
            print('df_traing[label].value_counts:')
            print(df_training[DefaultConfig.label_column].value_counts())

            print('df_validation[label].value_counts:')
            print(df_validation[DefaultConfig.label_column].value_counts())

            # # 采样
            # df_training = smote(df_training)
            # df_training[DefaultConfig.int_columns] = df_training[DefaultConfig.int_columns].astype(int)

            # 分割
            train_x, test_x, train_y, test_y = df_training.drop(labels=DefaultConfig.label_column, axis=1), \
                                               df_validation.drop(labels=DefaultConfig.label_column, axis=1), \
                                               df_training.loc[:, DefaultConfig.label_column], \
                                               df_validation.loc[:, DefaultConfig.label_column]

            # train_data, validation_data, train_data_weight, validation_data_weight = get_validation(train_x, test_x,
            #                                                                                         train_y, test_y,
            #                                                                                         DefaultConfig.categorical_columns,
            #                                                                                         seeds[model_seed])
            #
            # train_data = lgb.Dataset(train_data.drop(DefaultConfig.label_column, axis=1),
            #                          label=train_data.loc[:, DefaultConfig.label_column],
            #                          weight=train_data_weight)
            # validation_data = lgb.Dataset(validation_data.drop(DefaultConfig.label_column, axis=1),
            #                               label=validation_data.loc[:, DefaultConfig.label_column],
            #                               weight=validation_data_weight)

            train_data = lgb.Dataset(train_x, label=train_y)
            validation_data = lgb.Dataset(test_x, label=test_y)

            gc.collect()
            bst = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=10000, verbose_eval=1000,
                            early_stopping_rounds=2019, feval=lgb_f1_score)

            oof_lgb[test_x.index] += bst.predict(test_x)
            prediction_lgb += bst.predict(X_test.drop(DefaultConfig.label_column, axis=1)) / len(validation_type)
            gc.collect()

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = list(bst.feature_name())
            fold_importance_df["importance"] = bst.feature_importance(importance_type='split',
                                                                      iteration=bst.best_iteration)
            fold_importance_df["fold"] = index + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        oof += oof_lgb / num_model_seed
        prediction += prediction_lgb / num_model_seed
        print('logloss', log_loss(pd.get_dummies(y_train).values, oof_lgb))
        # 线下auc评分
        print('the roc_auc_score for train:', roc_auc_score(y_train, oof_lgb))

        if feature_importance is None:
            feature_importance = feature_importance_df
        else:
            feature_importance += feature_importance_df

    if feature_importance is not None:
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
        print(list(dict(result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False)).keys()))
        # 5折数据取平均值
        result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(40).plot.barh()
        plt.show()

    return prediction


def lgb_classifier_model(X_train, X_test, validation_type, **params):
    """
    LGBMClassifier
    :param X_train:
    :param X_test:
    :param validation_type:
    :param params:
    :return:
    """
    import lightgbm as lgb
    from boruta import BorutaPy

    gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             metric='F1',
                             verbose=0,
                             learning_rate=0.01,
                             num_leaves=100,
                             feature_fraction=0.8,
                             bagging_fraction=0.9,
                             bagging_freq=8,
                             lambda_l1=0.6,
                             lambda_l2=0,
                             random_state=42)
    # 有了gridsearch我们便不需要fit函数
    # 获取验证集
    df_training, df_validation, df_test = get_validation_data(df_training=X_train, df_test=X_test,
                                                              type=DefaultConfig.before_after)
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

    train_y = train_y.astype(int)
    test_y = test_y.astype(int)
    gbm.fit(X=train_x, y=train_y, eval_set=(test_x, test_y))
    # Boruta特征选择器
    feat_selector = BorutaPy(gbm, n_estimators=500, verbose=2, random_state=42)

    train = pd.concat([df_training, df_validation], axis=0, ignore_index=True)
    train.sort_values(by='ID', inplace=True)
    train.reset_index(inplace=True, drop=True)

    # 寻找所有的特征
    feat_selector.fit(train.drop(labels=DefaultConfig.label_column, axis=1).values,
                      train.loc[:, DefaultConfig.label_column].astype(int).values)
    # 检查所有的特征
    print(feat_selector.support_)
    print(train.columns[feat_selector.support_])


def xgb_model(X_train, X_test, validation_type, **params):
    """
    xgb 模型
    :param new_train:
    :param y:
    :param new_test:
    :param columns:
    :param params:
    :return:
    """
    import gc
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import log_loss, roc_auc_score, f1_score
    import matplotlib.pyplot as plt

    def xgb_score(y, t):
        t = t.get_label()
        y_bin = [1. if y_cont >= 0.5 else 0. for y_cont in y]  # binaryzing your output
        return 'f1', f1_score(t, y_bin)

    # y_train
    y_train = list(X_train.loc[:, DefaultConfig.label_column])
    # 特征重要性
    feature_importance = None
    # 线下验证
    oof = np.zeros((X_train.shape[0]))
    # 线上结论
    prediction = np.zeros((X_test.shape[0]))

    seeds = [42, 2019, 223344, 2019 * 2 + 1024, 332232111]
    num_model_seed = 1

    print(DefaultConfig.select_model + ' start training...')

    params = {'max_depth': 7,
              'eta': 0.25,
              'silent': 1,
              'subsample': 1,
              'lambda': 50,
              'objective': 'binary:logistic'}

    for model_seed in range(num_model_seed):
        print('模型 ', model_seed + 1, ' 开始训练')
        oof_xgb = np.zeros((X_train.shape[0]))
        prediction_xgb = np.zeros((X_test.shape[0]))

        # 存放特征重要性
        feature_importance_df = pd.DataFrame()
        for index, value in enumerate(validation_type):
            print('第 ' + str(index) + ' 折')

            # 获取验证集
            df_training, df_validation, df_test = get_validation_data(df_training=X_train, df_test=X_test, type=value)
            # 0/1数目
            print('df_traing[label].value_counts:')
            print(df_training[DefaultConfig.label_column].value_counts())

            print('df_validation[label].value_counts:')
            print(df_validation[DefaultConfig.label_column].value_counts())

            # # 采样
            # df_training = smote(df_training)
            # df_training[DefaultConfig.int_columns] = df_training[DefaultConfig.int_columns].astype(int)

            # 分割
            train_x, test_x, train_y, test_y = df_training.drop(labels=DefaultConfig.label_column, axis=1), \
                                               df_validation.drop(labels=DefaultConfig.label_column, axis=1), \
                                               df_training.loc[:, DefaultConfig.label_column], \
                                               df_validation.loc[:, DefaultConfig.label_column]

            # train_data, validation_data, train_data_weight, validation_data_weight = get_validation(train_x, test_x,
            #                                                                                         train_y, test_y,
            #                                                                                         DefaultConfig.categorical_columns,
            #                                                                                         seeds[model_seed])
            #
            # train_data = lgb.Dataset(train_data.drop(DefaultConfig.label_column, axis=1),
            #                          label=train_data.loc[:, DefaultConfig.label_column],
            #                          weight=train_data_weight)
            # validation_data = lgb.Dataset(validation_data.drop(DefaultConfig.label_column, axis=1),
            #                               label=validation_data.loc[:, DefaultConfig.label_column],
            #                               weight=validation_data_weight)

            train_data = xgb.DMatrix(data=train_x, label=train_y, nthread=10)
            validation_data = xgb.DMatrix(data=test_x, label=test_y, nthread=10)

            gc.collect()
            bst = xgb.train(params=params, dtrain=train_data,
                            evals=[(train_data, 'train'), (validation_data, 'validation')],
                            num_boost_round=1500, early_stopping_rounds=1000, feval=xgb_score,
                            verbose_eval=False)
            oof_xgb[test_x.index] += bst.predict(xgb.DMatrix(test_x))
            prediction_xgb += bst.predict(
                xgb.DMatrix(X_test.drop(DefaultConfig.label_column, axis=1) / len(validation_type)))
            gc.collect()

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = list(bst.get_score(importance_type='gain').keys())
            fold_importance_df["importance"] = list(bst.get_score(importance_type='gain').values())
            fold_importance_df["fold"] = index + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        oof += oof_xgb / num_model_seed
        prediction += prediction_xgb / num_model_seed
        print('logloss', log_loss(pd.get_dummies(y_train).values, oof_xgb))
        # 线下auc评分
        print('the roc_auc_score for train:', roc_auc_score(y_train, oof_xgb))

        if feature_importance is None:
            feature_importance = feature_importance_df
        else:
            feature_importance += feature_importance_df

    print(feature_importance['importance'])
    if feature_importance is not None:
        feature_importance['importance'] /= num_model_seed

    print('logloss', log_loss(pd.get_dummies(y_train).values, oof))
    print('ac', roc_auc_score(y_train, oof))

    if feature_importance is not None:
        feature_importance.to_hdf(path_or_buf=DefaultConfig.xgb_feature_cache_path, key='xgb')
        # 读取feature_importance_df
        feature_importance_df = reduce_mem_usage(
            pd.read_hdf(path_or_buf=DefaultConfig.xgb_feature_cache_path, key='xgb', mode='r'))

        plt.figure(figsize=(8, 8))
        # 按照flod分组
        group = feature_importance_df.groupby(by=['fold'])

        result = []
        for key, value in group:
            value = value[['feature', 'importance']]

            result.append(value)

        result = pd.concat(result)
        print(list(dict(result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False)).keys()))
        # 5折数据取平均值
        result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(40).plot.barh()
        plt.show()

    return prediction


def cbt_model(X_train, X_test, validation_type, **params):
    """
    cbt 模型
    :param new_train:
    :param y:
    :param new_test:
    :param columns:
    :param params:
    :return:
    """
    import gc
    import numpy as np

    import pandas as pd
    import catboost as cbt
    from sklearn.metrics import log_loss, roc_auc_score, f1_score
    import matplotlib.pyplot as plt

    # y_train
    y_train = list(X_train.loc[:, DefaultConfig.label_column])
    # 特征重要性
    feature_importance = None
    # 线下验证
    oof = np.zeros((X_train.shape[0]))
    # 线上结论
    prediction = np.zeros((X_test.shape[0]))

    seeds = [42, 2019, 223344, 2019 * 2 + 1024, 332232111]
    num_model_seed = 1

    print(DefaultConfig.select_model + ' start training...')

    for model_seed in range(num_model_seed):
        print('模型 ', model_seed + 1, ' 开始训练')
        oof_cbt = np.zeros((X_train.shape[0]))
        prediction_cbt = np.zeros((X_test.shape[0]))

        # 存放特征重要性
        feature_importance_df = pd.DataFrame()
        for index, value in enumerate(validation_type):
            print('第 ' + str(index) + ' 折')

            # 获取验证集
            df_training, df_validation, df_test = get_validation_data(df_training=X_train, df_test=X_test, type=value)
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

            gc.collect()
            bst = cbt.CatBoostClassifier(iterations=3000, learning_rate=0.005, verbose=300, early_stopping_rounds=1666,
                                         task_type='GPU', use_best_model=True)
            bst.fit(train_x, train_y,
                    eval_set=(test_x, test_y))

            oof_cbt[test_x.index] += bst.predict_proba(test_x)[:, 1]
            prediction_cbt += bst.predict_proba(X_test.drop(DefaultConfig.label_column, axis=1))[:, 1] / len(
                validation_type)
            gc.collect()

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = list(bst.feature_names_)
            fold_importance_df["importance"] = bst.get_feature_importance()
            fold_importance_df["fold"] = index + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        oof += oof_cbt / num_model_seed
        prediction += prediction_cbt / num_model_seed
        print('logloss', log_loss(pd.get_dummies(y_train).values, oof_cbt))
        # 线下auc评分
        print('the roc_auc_score for train:', roc_auc_score(y_train, oof_cbt))

        if feature_importance is None:
            feature_importance = feature_importance_df
        else:
            feature_importance += feature_importance_df

    feature_importance['importance'] /= num_model_seed
    print('logloss', log_loss(pd.get_dummies(y_train).values, oof))
    print('ac', roc_auc_score(y_train, oof))

    if feature_importance is not None:
        feature_importance.to_hdf(path_or_buf=DefaultConfig.cbt_feature_cache_path, key='cbt')
        # 读取feature_importance_df
        feature_importance_df = reduce_mem_usage(
            pd.read_hdf(path_or_buf=DefaultConfig.cbt_feature_cache_path, key='cbt', mode='r'))

        plt.figure(figsize=(8, 8))
        # 按照flod分组
        group = feature_importance_df.groupby(by=['fold'])

        result = []
        for key, value in group:
            value = value[['feature', 'importance']]

            result.append(value)

        result = pd.concat(result)
        print(list(dict(result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False)).keys()))
        # 5折数据取平均值
        result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(40).plot.barh()
        plt.show()

    return prediction


def generate_submition(prediction, X_test, validation_type, submit_or_not=True, **params):
    """
    生成集过
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
        _, ids = get_df_test()
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
    _, ids = get_df_test()
    # 设置成“category”数据类型
    sub['ID'] = sub['ID'].astype('category')
    # inplace = True，使 recorder_categories生效
    sub['ID'].cat.reorder_categories(ids, inplace=True)
    # inplace = True，使 df生效
    sub.sort_values('ID', inplace=True)

    sub.to_csv(path_or_buf=DefaultConfig.project_path + '/data/submit/' + DefaultConfig.select_model + '_' +
                           validation_type + '_submit.csv', index=False, encoding='utf-8')
    return sub


def merge(type, **params):
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
        lgb_before = pd.read_csv(DefaultConfig.lgb_before_submit, encoding='utf-8')
        cbt_after = pd.read_csv(DefaultConfig.cbt_after_submit, encoding='utf-8')

        lgb_before['Predicted_Results'] = (lgb_before['Predicted_Results']*0.5 + cbt_after['Predicted_Results']*0.5)

    elif DefaultConfig.merge_type is 'lgb_xgb':
        lgb_before = pd.read_csv(DefaultConfig.lgb_before_submit, encoding='utf-8')
        xgb_before_after = pd.read_csv(DefaultConfig.xgb_before_after_submit, encoding='utf-8')

        lgb_before['Predicted_Results'] = 0.9 * lgb_before['Predicted_Results'] + 0.1 * xgb_before_after[
            'Predicted_Results']

    elif DefaultConfig.merge_type is 'lgb_cbt_xgb':
        lgb_before = pd.read_csv(DefaultConfig.lgb_before_submit, encoding='utf-8')
        cbt_after = pd.read_csv(DefaultConfig.cbt_after_submit, encoding='utf-8')
        xgb_before_after = pd.read_csv(DefaultConfig.xgb_before_after_submit, encoding='utf-8')

        lgb_before['Predicted_Results'] = 0.4 * lgb_before['Predicted_Results'] + 0.5 * cbt_after['Predicted_Results'] + \
                                          0.1 * xgb_before_after['Predicted_Results']

    result = []
    for i in list(lgb_before['Predicted_Results']):
        if i >= 0.5:
            result.append(1)
        else:
            result.append(0)

    print('sum(1): ', sum(result))
    lgb_before['Predicted_Results'] = result

    lgb_before.to_csv(path_or_buf=DefaultConfig.submition, index=False, encoding='utf-8')

# if __name__ == '__main__':
#     df_training, df_validation, df_test = preprocess()
