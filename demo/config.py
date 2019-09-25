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

    # select_model
    # select_model = 'lgb'
    # select_model = 'cbt'
    # select_model = 'xgb'
    select_model = 'merge'

    # merge_type
    # merge_type = 'lgb_cbt_xgb'
    merge_type = 'lgb_cbt'
    # merge_type = 'lgb_xgb'

    # label_column
    label_column = "'Purchase or not'"

    # cache_path
    df_no_replace = False
    df_training_cache_path = project_path + '/data/cache/df_training.h5'
    df_test_cache_path = project_path + '/data/cache/df_test.h5'

    lgb_feature_cache_path = project_path + '/data/cache/lgb_feature.h5'
    xgb_feature_cache_path = project_path + '/data/cache/xgb_feature.h5'
    cbt_feature_cache_path = project_path + '/data/cache/cbt_feature.h5'

    # 类别特征
    categorical_columns = ["'User area'", "gender", "'Cumulative using time'", "'Product service usage'",
                           "'Pay a monthly fee by credit card'", "'Active user'"]
    # 整数特征
    int_columns = ["ID", "'Product using score'", "'User area'", "gender", "age", "'Cumulative using time'",
                   "'Product service usage'", "'Pay a monthly fee by credit card'", "'Active user'"]
    # 浮点型特征
    float_columns = ["'Point balance'", "' Estimated salary'"]

    if select_model is 'lgb':
        # mean/mode 0.837
        columns = ["ID", "' Estimated salary'_groupby_age_mean_ratio", "' Estimated salary'",
                   "'Point balance'_groupby_age_mean_ratio", "'Product using score'",
                   "'Point balance'", "age", "'Cumulative using time'", "'User area'", "gender",
                   "'Product service usage'", "'Active user'", "'Pay a monthly fee by credit card'"]

        columns.append("'Purchase or not'")

        # before_after
        before_after = 'before'

    elif select_model is 'cbt':
        # 0.837
        columns = ["age",
                   "'Product service usage'",
                   "'Point balance'",
                   "'User area'",
                   "gender",
                   "' Estimated salary'",
                   "'Cumulative using time'",
                   "'Active user'",
                   "ID",
                   "'Product using score'",
                   "'Pay a monthly fee by credit card'"]

        columns.append("'Purchase or not'")

        before_after = 'before'

    elif select_model is 'xgb':
        columns = ["'Product service usage'", "'Active user'", 'age', "'User area'", "'Point balance'", 'gender',
                   "' Estimated salary'", 'ID', "'Product using score'", "'Pay a monthly fee by credit card'",
                   "'Cumulative using time'"]

        columns.append("'Purchase or not'")

        # before_after
        before_after = 'before_after'

    # lgb before submit
    lgb_before_submit = project_path + '/data/submit/lgb_before_submit.csv'
    # lgb after submit
    lgb_after_submit = project_path + '/data/submit/lgb_after_submit.csv'
    # lgb before_after submit
    lgb_before_after_submit = project_path + '/data/submit/lgb_before_after_submit.csv'

    # cbt before submit
    cbt_before_submit = project_path + '/data/submit/cbt_before_submit.csv'
    # cbt after submit
    cbt_after_submit = project_path + '/data/submit/cbt_after_submit.csv'
    # cbt before_after submit
    cbt_before_after_submit = project_path + '/data/submit/cbt_before_after_submit.csv'

    # xgb before submit
    xgb_before_submit = project_path + '/data/submit/xgb_before_submit.csv'
    # xgb after submit
    xgb_after_submit = project_path + '/data/submit/xgb_after_submit.csv'
    # xgb before_after submit
    xgb_before_after_submit = project_path + '/data/submit/xgb_before_after_submit.csv'

    # lgb submit
    lgb_submit = project_path + '/data/submit/lgb_submit.csv'
    # submition
    submition = project_path + '/data/submit/submition.csv'

