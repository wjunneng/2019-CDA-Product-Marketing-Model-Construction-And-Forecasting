import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from demo.preprocess import Preprocess
from configuration.config import DefaultConfig


class Step_1(object):
    def __init__(self, X, TARGET):
        self.X = X
        self.TARGET = TARGET

    def missing_values_table(self, df):
        """
        # 分析缺失值函数
        """
        # Total missing values
        mis_val = df.isnull().sum()
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(columns={0: '缺失的个数', 1: '缺失百分比'})
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '缺失百分比', ascending=False).round(1)
        # Print some summary information
        print("数据一共 " + str(df.shape[1]) + " 维.\n" +
              "有 " + str(mis_val_table_ren_columns.shape[0]) +
              " 维有缺失值.")
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

    def main(self):
        # ---------------------------------------- step 1 观察数据 ----------------------------------------#
        print('数据 shape: ', self.X.shape)
        for column in list(self.X.columns):
            print('column: ', column)
            print(len(self.X[column].value_counts()))

        # 分析 TARGET 分布
        # TODO
        """
        判断是否 imbalanced class problem
        """
        print("---------------正负样本分布--------------------")
        distrubtion_0_1 = self.X[self.TARGET].value_counts()
        print(distrubtion_0_1)
        self.X[self.TARGET].astype(int).plot.hist()
        plt.show()

        print("---------------缺失值分布--------------------")
        missing_values = self.missing_values_table(self.X)
        print("缺失率最高的列")
        print(missing_values.head(20))

        # 分析数据类型 Column Types
        print("---------------特征的数据类型--------------------")
        ColumnTypes = self.X.dtypes.value_counts()
        print("tips:object为字符类型:")
        print(ColumnTypes)

        # ---------------------------------------- step 2 探索数据 ----------------------------------------#
        # 相关性分析
        print("--------------相关性分析--------------------")
        # Find correlations with the self.TARGET and sort
        correlations = self.X.corr()[self.TARGET].sort_values()

        # Display correlations
        print('正相关性最高的特征：')
        print(correlations.tail(3))
        print('负相关性最高的特征：')
        print(correlations.head(3))

        # # 画出相关性最高的特征与样本分布的关系
        # import_fea = correlations.abs().sort_values().tail(11).index.values[:-1]
        # for i, source in enumerate(import_fea):
        #     # create a new subplot for each source
        #     # plot repaid loans
        #     # 空值不能画图
        #     self.X.fillna(0)
        #     sns.kdeplot(self.X.loc[self.X[self.TARGET] == 0, source], label='self.TARGET == 0')
        #     # plot loans that were not repaid
        #     sns.kdeplot(self.X.loc[self.X[self.TARGET] == 1, source], label='self.TARGET == 1')
        #     # Label the plots
        #     plt.title('Distribution of %s by self.TARGET Value' % source)
        #     plt.xlabel('%s' % source)
        #     plt.ylabel('Density')
        #     plt.show()


if __name__ == '__main__':
    X = Preprocess(df_training_path=DefaultConfig.df_training_path,
                   df_test_path=DefaultConfig.df_test_path).get_df_training()
    X.replace('?', np.nan, inplace=True)
    for column in list(X.columns):
        try:
            X[column] = X[column].astype(float)
            print(column)
        except:
            continue
    Step_1(X=X, TARGET=DefaultConfig.label_column).main()
