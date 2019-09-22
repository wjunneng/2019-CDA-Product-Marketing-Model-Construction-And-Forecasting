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
    select_model = 'lgb'
    # select_model = 'cbt'

    # label_column
    label_column = "'Purchase or not'"

    # cache_path
    df_no_replace = False
    df_training_cache_path = project_path + '/data/cache/df_training.h5'
    df_test_cache_path = project_path + '/data/cache/df_test.h5'

    lgb_feature_cache_path = project_path + '/data/cache/lgb_feature.h5'
    xgb_feature_cache_path = project_path + '/data/cache/xgb_feature.h5'
    cat_feature_cache_path = project_path + '/data/cache/cat_feature.h5'

    # 类别特征
    categorical_columns = ["'User area'", "gender", "'Cumulative using time'", "'Product service usage'",
                           "'Pay a monthly fee by credit card'", "'Active user'"]

    if select_model is 'lgb':
        # mean/mode
        # columns = ["ID", "'Product using score'", "'Point balance'", "' Estimated salary'", "age", "'User area'",
        #            "gender",
        #            "'Active user'", "'Product service usage'", "'Pay a monthly fee by credit card'",
        #            "'Cumulative using time'", "'Purchase or not'"]

        # EM
        columns = ["age", "'Product using score'", "ID", "' Estimated salary'", "'Point balance'",
                   "'Cumulative using time'",
                   "'User area'", "'Product service usage'", "'Active user'", "'Pay a monthly fee by credit card'",
                   "gender", "'Purchase or not'"]

        # columns = ["ID", "'Point balance'", "'Product using score'", "' Estimated salary'", "age",
        #            "'Cumulative using time'", "'User area'", "gender", "'Product service usage'", "'Active user'",
        #            "'Product service usage' Virtual", "'Active user' Virtual", "' Estimated salary' Virtual",
        #            "'Cumulative using time' Virtual", "age Virtual", "'Pay a monthly fee by credit card' Virtual",
        #            "'Point balance' Virtual", "'Product using score' Virtual", "'Pay a monthly fee by credit card'",
        #            "'User area' Virtual", "gender Virtual", "'Purchase or not'"]

    elif select_model is 'cbt':
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
                   "'Pay a monthly fee by credit card'",
                   "'Purchase or not'"]

    # lgb before submit
    lgb_before_submit = project_path + '/data/submit/lgb_before_submit.csv'
    # lgb after submit
    lgb_after_submit = project_path + '/data/submit/lgb_after_submit.csv'
    # cbt before submit
    cbt_before_submit = project_path + '/data/submit/cbt_before_submit.csv'
    # cbt after submit
    cbt_after_submit = project_path + '/data/submit/cbt_after_submit.csv'

    # lgb submit
    lgb_submit = project_path + '/data/submit/lgb_submit.csv'
    # lgb_cbt_submit
    lgb_cbt_submit = project_path + '/data/submit/lgb_cbt_submit.csv'
