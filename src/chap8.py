import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("./sukkiri-ml-codes/datafiles/Boston.csv")
print(df.head(2))

print(df['CRIME'].value_counts())
crime = pd.get_dummies(df['CRIME'], drop_first=True)
df2 = pd.concat([df, crime], axis=1)
df2 = df2.drop(['CRIME'], axis=1)
print(df2.head(2))

# 検証データの作成。まず全体を分割
train_val, test = train_test_split(df2, test_size=0.2, random_state=0)

print(train_val.isnull().sum())

# 穴埋め fillna
train_val_mean = train_val.mean()
print(train_val_mean)
train_val2 = train_val.fillna(train_val_mean)

# plot
colname = train_val2.columns
for name in colname:
    plt.plot(train_val2[name], train_val2['PRICE'], linestyle="", marker="o")
    image_path = f"./img/chap8-{name}.png"
    plt.savefig(image_path)
    plt.cla()

# 特徴量の取捨選択
# 決定木ではモデルが自動で特徴量の取捨選択をするが
# 重回帰分析ではモデルは与えられたすべての特徴量を利用する
# そのため予測に影響の与えない列があると、モデル全体で予測性能の低下が起こる場合がある

# 相関関係のある特徴量
# 散布図で右肩上がり、または右肩下がりのデータは相関関係がある

# 外れ値
# すべての外れ値を取り除いてしまうと、テストデータで外れ値が含まれている場合に
# 予測性能が下がるのである程度は残しておく

out_line1 = train_val2[(train_val2['RM'] < 6) &
                       (train_val2['PRICE'] > 40)].index
out_line2 = train_val2[(train_val2['PTRATIO'] > 18) &
                       (train_val2['PRICE'] > 40)].index

print(out_line1, out_line2)

train_val3 = train_val2.drop([76], axis=0)

col = ["INDUS", "NOX", "RM", "PTRATIO", "LSTAT", "PRICE"]
train_val4 = train_val3[col]
print(train_val4.head(3))

# 相関係数
print(train_val4.corr()['PRICE'])
# ここでは正負(+-)に関係なく相関係数の強さを調べるため絶対値にしてからsort

train_cor = pd.Series(train_val4.corr()['PRICE'])
abs_cor = train_cor.map(abs)
print(abs_cor)
abs_cor = abs_cor.sort_values(ascending=False)
print(abs_cor)

# 相関係数のチェックは外れ値の削除後に行う
