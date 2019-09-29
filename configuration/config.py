# -*- coding: utf-8 -*-
"""
    配置文件
"""
import os


class DefaultConfig(object):
    """
    参数配置
    """

    def __init__(self):
        pass

    # 项目路径
    project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])

    # df_test 路径
    df_test_path = project_path + '/data/original/df_test.csv'
    # df_training 路径
    df_training_path = project_path + '/data/original/df_training.csv'

    # label_column
    label_column = "Purchase or not"
    # predict_column
    predict_column = "Predicted_Results"
    # submition
    submition = project_path + '/data/submit/submition.csv'

    # ###################################################### cache
    df_no_replace = False
    df_training_cache_path = project_path + '/data/cache/df_training.h5'
    df_test_cache_path = project_path + '/data/cache/df_test.h5'
    df_cache_path = project_path + '/data/cache/df.h5'

    lgb_feature_cache_path = project_path + '/data/cache/lgb_feature.h5'
    xgb_feature_cache_path = project_path + '/data/cache/xgb_feature.h5'
    cbt_feature_cache_path = project_path + '/data/cache/cbt_feature.h5'
    # ######################################################

    # select_model
    # select_model = 'lgbm'
    # select_model = 'lgbm_classifier'

    select_model = 'cbt'
    # select_model = 'xgb'
    # select_model = 'merge'

    modeltypes = ['lgbm', 'cbt']

    # 浮点型特征
    float_columns = ["Point balance", "Estimated salary"]

    # 整数特征
    int_columns = ["ID", "Product using score", "User area", "gender", "age", "Cumulative using time",
                   "Product service usage", "Pay a monthly fee by credit card", "Active user"]

    if select_model is 'lgbm':
        columns = list(
            ['age', 'Product using score', 'Point balance', 'Estimated salary', 'ID', 'Product service usage',
             'User area', 'gender', 'Cumulative using time', 'Active user', 'Pay a monthly fee by credit card'])
        columns.append('Purchase or not')

        # before_after
        before_after = ['before']

    elif select_model is 'cbt':
        columns = list(
            ["age",
             "Product service usage",
             "Point balance",
             "User area",
             "gender",
             "Estimated salary",
             "Cumulative using time",
             "Active user",
             "ID",
             "Product using score",
             "Pay a monthly fee by credit card"]

        )
        columns.append('Purchase or not')
        # before_after
        before_after = ['before']

    elif select_model is 'xgb':

        # before_after
        before_after = 'before_after'

    # rule_ID_1
    rule_ID_1 = [8924, 1877, 2463, 9256, 3366, 2500, 8042, 1470, 8851, 5286, 2542, 5138, 5495, 6254]
    rule_ID_0 = [372, 2459, 3603, 3367, 253, 4257, 2003, 5491]
