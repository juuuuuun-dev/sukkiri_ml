import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./sukkiri-ml-codes/datafiles/cinema.csv')
print(df.isnull().any(axis=0))

# 穴埋め
df2 = df.fillna(df.mean())
# print(df2.isnull().any(axis=0))
# print(df2['SNS2'])
# 外れ値
# 平均から突出したようなデータが含まれるとscoreが挙がりにくいので外れ値の処理を行う。

plt.plot(df2['SNS2'], df2['sales'])
plt.savefig('image1.png')


# 列に対して比較演算
test = pd.DataFrame(
    {
        'Acolumn':[1,2,8],
        'Bcolumn':[8,5,6],
        'Ccolumn':[4,4,4],
    }
)

# index, a value, b valueの順で表示
print(test[test['Acolumn'] < 2])
print(test['Acolumn'] < 2)

# 外れ値の削除
no = df2[(df2['SNS2'] > 1000) & (df2['sales'] < 8500)].index
print(no)

df3 = df2.drop(no, axis=0)
df3.plot(kind='scatter',x='SNS2',y='sales')
plt.savefig('image2.png')

# 特徴量取り出し 1
col = ['SNS1', 'SNS2', 'actor', 'original']
x = df3[col]
# 特徴量取り出し 2
x = df3.loc[:, 'SNS1':'original']

t = df3['sales']

# 訓練用 テスト用に分ける ホールドアウト
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)
