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
