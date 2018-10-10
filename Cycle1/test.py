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
    #正規分布で、平均172.14, 標準偏差5.57に一致する1000個のデータをランダムに取得
    rdData = normal(172.14, 5.57, 1000)
    #DataFrame
    df = pd.DataFrame(rdData)
    #csv出力
    df.to_csv("test.csv")

def solve3():
    #csv取り込み
    df = pd.read_csv("./test.csv", header=None, names=('A', 'B'))
    print(df)
    #散布図
    df.plot.scatter(x='A', y='B')
    print(df)

def testSolve1():
    # 0から0.99まで、0.01間隔で入ったリストXを作る
    X = np.arange(0, 1, 0.01)

    #データ作成
    array = [[0] * 2 for i in range(100)]
    i = 0
    for var_x in X:
        print(normal(math.sin(2 * math.pi * var_x), 0.3))
        array[i][0] = var_x
        array[i][1] = normal(math.sin(2 * math.pi * var_x), 0.3)
        i += 1

    #データフレーム作成
    df = pd.DataFrame(array, columns=['X', 'Y'])

    #ランダムにシャッフル
    df_random = df.sample(frac=1)

    #CSV出力
    #columns=[取り出すカラム], index=[行番号の有無], header=[ヘッダーの有無]
    df_random.to_csv("testSolve.csv", columns='Y', index=False, header=False)

    #散布図出力
    df.plot.scatter(x='X', y='Y')
    print(df)

def testSolve2():
    array = [[0]*2 for i in range(100)]

    #100回データ作成
    for i in range(1, 101):
        #0~1の乱数を1つ生成
        h = rand()

        f = 1 #f(0) = 1
        for n in range(1, 120):
            f = f + (f* (1 - f)) - h

        if f < 0:
            f = 0

        #配列にセット
        array[i-1][0] = h
        array[i-1][1] = f

    #DataFrame生成
    df = pd.DataFrame(array, columns=['h', 'f'])

    #ソート
    df = df.sort_values('h')
    print(df)

    # 散布図出力
    df.plot.scatter(x='h', y='f')
    print(df)


