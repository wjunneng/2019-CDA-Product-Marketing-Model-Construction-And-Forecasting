import warnings
from util import *

warnings.filterwarnings('ignore')


def main():
    """
    主函数
    :return:
    """
    import time

    start = time.clock()

    # 加载数据
    df_training, df_test = preprocess()
    print('\n加载数据 耗时： %s \n' % str(time.clock() - start))

    # 获取验证集数据
    lgb_model(df_training, df_test)
    print('\n模型训练+预测 耗时： %s \n' % str(time.clock() - start))


if __name__ == '__main__':
    # 主函数
    main()
