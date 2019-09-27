import warnings

from configuration.config import DefaultConfig
from demo.preprocess import Preprocess
from model.lightgbm import LightGbm
from model.catboost import CatBoost

warnings.filterwarnings('ignore')


def main():
    """
    主函数
    :return:
    """
    import time

    start = time.clock()

    preprocess = Preprocess(df_training_path=DefaultConfig.df_training_path,
                            df_test_path=DefaultConfig.df_test_path)

    if DefaultConfig.select_model is 'merge':
        # merge
        preprocess.merge()
        print('\nmerge 耗时： %s \n' % str(time.clock() - start))

        return

    # 加载数据
    df_training, df_test = preprocess.main()
    print('\n加载数据 耗时： %s \n' % str(time.clock() - start))

    print(list(df_training.columns))
    df_training = df_training[DefaultConfig.columns]
    df_test = df_test[DefaultConfig.columns]

    if DefaultConfig.select_model is 'lgb':
        # 获取验证集数据
        prediction = LightGbm(df_training, df_test).main()
        print('\n模型训练+预测 耗时： %s \n' % str(time.clock() - start))
    elif DefaultConfig.select_model is 'cbt':
        # 获取验证集数据
        prediction = CatBoost(df_training, df_test).main()
        print('\n模型训练+预测 耗时： %s \n' % str(time.clock() - start))
    # elif DefaultConfig.select_model is 'xgb':
    #     # 获取验证集数据
    #     prediction = xgb_model(df_training, df_test, validation_type=[DefaultConfig.before_after])
    #     print('\n模型训练+预测 耗时： %s \n' % str(time.clock() - start))

    # 生成提交结果
    preprocess.generate_submition(prediction, df_test, validation_type=DefaultConfig.before_after)
    print('\n生成提交结果 耗时： %s \n' % str(time.clock() - start))


if __name__ == '__main__':
    # 主函数
    main()
