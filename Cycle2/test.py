import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
from numpy.random import *
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

def solve1():
    # csv取り込み
    df = pd.read_csv("./tallWeight.csv", skiprows=1, names=('tall', 'weight'))

    # 統計量を確認
    #print(df.describe())

    num = list(range(0,200))

    # 相関係数行列を求める
    df_corr = df.corr()
    #print(df_corr)

    t1, p1 = stats.ttest_rel(df['tall'], df['weight'])
    #print('p値: ', p1)

    #plt.scatter(df.weight, df.tall)
    #plt.xlabel('weight')
    #plt.ylabel('tall')
    #plt.grid(True)

    # 線形回帰
    model1 = LinearRegression()

    #print(df.shape)
    #print(df.weight.shape)
    # 学習
    model1.fit(df, df.weight)

    plt.scatter(df.weight, df.tall)
    plt.plot(df, model1.predict(df), c='red')
    plt.xlabel('weight')
    plt.ylabel('tall')
    plt.grid(True)

    print('係数: ', model1.coef_)
    print('切片: ', model1.intercept_)