import pandas as pd
import sklearn import tree

df = pd.read_csv('./sukkiri-ml-codes/datafiles/iris.csv')
# print(df.head(3))
print(df['種類'].unique())
print(df['種類'].value_counts())
# print(df.tail(3))
# any axis0で行(横) 1で列(縦)
# all(全てが) と any(いずれかが) 
# print(df.isnull().any(axis=0))
# print(df.sum())
print(df.isnull().sum())
# dropna nullのあるデータを削除 axis0で行(横) 1で列(縦)
df2 = df.dropna(how='any', axis=0)
print(df2.tail(3))

# fillna nullの穴埋め
df["花弁長さ"] = df["花弁長さ"].fillna(0)
print(df.tail(3))

# mean 平均値の計算
print(df.mean())

# 書き換え
df = pd.read_csv('./sukkiri-ml-codes/datafiles/iris.csv')
colmean = df.mean()
df2 = df.fillna(colmean)
print(df2.isnull().any(axis=0))

# 特徴量と正解データの取り出し
xcol = ['がく片長さ','がく片幅','花弁長さ','花弁幅']
x = df2[xcol]
t = df2['種類']

# 決定木 model
# random_state (random seed)乱数を生成する際に使用する数値。乱数の再現性を持たせるために必要

model = tree.DesisionTreeClassifier(max_depth=2, random_state=0)


