import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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
        assert len(self.columns) == 1

        before = self.df[self.columns[0]].values

        before = before.astype(str)

        after = LabelEncoder().fit_transform(before)

        # 重新构造
        self.df[self.columns[0]] = after

        return self.df

    def one_hot_encoder(self, **params) -> pd.DataFrame:
        """
        one-hot 编码
        :param params:
        :return:
        """
        for column in self.columns:
            before = self.df[column]

            before = before.astype(str)

            after = OneHotEncoder().fit_transform(before.values.reshape(-1, 1))

            print([column + '_' + str(i) for i in range(after.shape[1])])

            tmp = pd.DataFrame(data=after.toarray(), columns=[column + '_' + str(i) for i in range(after.shape[1])])

            # 重新构造
            self.df = pd.concat([self.df, tmp], axis=1)

        return self.df
