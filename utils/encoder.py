import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Encoder(object):
    def __init__(self, df: pd.DataFrame, columns: list):
        self.df = df
        self.columns = columns

    def label_encoder(self, **params) -> pd.DataFrame:
        """
        标签编码
        :param params:
        :return:
        """
        before = self.df[self.columns]

        before = before.astype(str)

        after = LabelEncoder().fit_transform(before)

        # 重新构造
        self.df[self.columns] = pd.DataFrame(data=after, columns=self.columns, index=None)

        return self.df
