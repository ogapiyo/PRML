import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
from numpy.random import *
import pandas as pd

def solve_2dim_equations(a, b, c, d, i, j):

    m = np.array([[a, b], [c, d]])
    n = np.array([i, j])
    x = np.dot(np.linalg.inv(m), n)
    print(x)

def solve1():
    # 0から100まで、0.01間隔で入ったリストXを作る！
    X = np.arange(0, 100, 0.1)
    # 確率密度関数にX,平均50、標準偏差20を代入
    Y = norm.pdf(X, 50, 20)

    # x,yを引数にして、関数の色をr(red)に指定！カラーコードでも大丈夫です！
    plt.plot(X, Y, color='r')
    plt.show()

def solve2():
    #正規分布で、平均172.14, 標準偏差5.57**2に一致する1000個のデータをランダムに取得
    rdData = normal(172.14, 5.57**2, 1000)
    #DataFrame
    df = pd.DataFrame(rdData)
    #csv出力
    df.to_csv("test.csv")

def solve3():
    #csv取り込み
    df = pd.read_csv("./test.csv", header=None, names=('A', 'B'))
    #散布図
    df.plot.scatter(x='A', y='B')
    #print(df)