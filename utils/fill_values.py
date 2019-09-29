import pandas as pd
from ycimpute.imputer import knnimput, mice, iterforest, EM, SimpleFill


class FillValues(object):
    def __init__(self, df: pd.DataFrame, columns: list):
        self.df = df
        self.columns = columns

    def knn_missing_value(self, **params) -> pd.DataFrame:
        """
        KNN算法可用于在多维空间中将点与其最近的k个邻居进行匹配。 它可以用于连续, 离散, 序数和分类的数据, 这对处理所有类型的缺失数据特别有用。
        使用KNN作为缺失值的假设是, 基于其他变量, 点值可以近似于与其最接近的点的值。
        :param params:
        :return:
        """
        before = self.df[self.columns].values
        after = knnimput.KNN(k=16).complete(before)
        self.df[self.columns] = after

        return self.df

    def mice_missing_value(self, **params) -> pd.DataFrame:
        """
        算法基于线性回归。mice能填充如连续型,二值型,离散型等混
        合数据并保持数据一致性
        :param params:
        :return:
        """
        before = self.df[self.columns].values
        after = mice.MICE().complete(before)
        self.df[self.columns] = after

        return self.df

    def iterforest_missing_value(self, **params) -> pd.DataFrame:
        """
        是基于随机森林的一种算法,能够很好的填充连续
        型,离散型相混合的缺失数据。尤其对于各变量之间出存在相关
        性的数据集表现不错
        :param params:
        :return:
        """
        before = self.df[self.columns].values
        after = iterforest.IterImput().complete(before)
        self.df[self.columns] = after

        return self.df

    def em_missing_value(self, **params) -> pd.DataFrame:
        """
        EM基于高斯分布能处理混合数据,但在连续型数据上表现的相对较好。在多种数
        据缺失机制下,EM相对于其他方法有着较好的表现。
        :param params:
        :return:
        """
        before = self.df[self.columns].values
        after = EM(max_iter=1000).complete(before)
        self.df[self.columns] = after

        return self.df

    def simplefill(self, fillmethod: str, **params) -> pd.DataFrame:
        """
        'mean', 'zero', 'median', 'min', 'random'
        :param fillmethod:
        :param params:
        :return:
        """
        if fillmethod is 'mode':
            # 众数填充
            for column in self.columns:
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)

            return self.df

        elif fillmethod is 'mean':
            # 均值填充
            for column in self.columns:
                self.df[column].fillna(self.df[column].mean(), inplace=True)

            return self.df

        # before = self.df[self.columns].values
        # after = SimpleFill(fill_method=fillmethod).complete(before.reshape(1, -1))
        # self.df[self.columns] = after

        return self.df
