import pandas as pd
import numpy as np
from boruta import BorutaPy


class FeatureSelection(object):
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train

    def boruta(self, model) -> pd.DataFrame:
        """
        boruta 模型选择
        :param model:
        :return:
        """
        # Boruta特征选择器
        feat_selector = BorutaPy(model, n_estimators=500, verbose=2, random_state=None)

        # 寻找所有的特征
        feat_selector.fit(self.X_train.values, self.y_train.values)

        # 检查所有的特征
        print(feat_selector.support_)
        print(self.X_train.columns[feat_selector.support_])

        return self.X_train.columns[feat_selector.support_]
