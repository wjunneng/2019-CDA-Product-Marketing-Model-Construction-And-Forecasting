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

    # label_column
    label_column = "'Purchase or not'"

    # cache_path
    df_no_replace = True
    df_training_cache_path = project_path + '/data/cache/df_training.h5'
    df_test_cache_path = project_path + '/data/cache/df_test.h5'

    lgb_feature_cache_path = project_path + '/data/cache/lgb_feature.h5'
    xgb_feature_cache_path = project_path + '/data/cache/xgb_feature.h5'
    cat_feature_cache_path = project_path + '/data/cache/cat_feature.h5'



