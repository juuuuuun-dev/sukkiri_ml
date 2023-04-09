import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# 相関係数 正解データのPRICEとの相関係数を表示
print(train_val4.corr()['PRICE'])
# ここでは正負(+-)に関係なく相関係数の強さを調べるため絶対値にしてからsort
train_cor = pd.Series(train_val4.corr()['PRICE'])
abs_cor = train_cor.map(abs)
print(abs_cor)
abs_cor = abs_cor.sort_values(ascending=False)
print(abs_cor)

# 相関係数のチェックは外れ値の削除後に行う
# PRICEとの相関係数、上位3つを特徴量として使用
col = ["RM", "LSTAT", "PTRATIO"]
x = train_val4[col]
t = train_val4[["PRICE"]]

# 訓練データをさらに訓練データと検証データに分割
x_train, x_val, y_train, y_val = train_test_split(
    x, t, test_size=0.2, random_state=0)

print(x_train.mean())

# 各特徴量で平均値が大きく異なる場合は、特徴量を標準化して
# 各特徴量の平均と標準偏差を統一させることにより性能が上がる場合がある
sc_model_x = StandardScaler()
sc_model_x.fit(x_train)

sc_x = sc_model_x.transform(x_train)
print(sc_x)

tmp_df = pd.DataFrame(sc_x, columns=x_train.columns)
print(tmp_df.mean())
print(tmp_df.std())

# 正解データも標準化
sc_model_y = StandardScaler()
sc_model_y.fit(y_train)
sc_y = sc_model_y.transform(y_train)

# モデル作成と学習
model = LinearRegression()
model.fit(sc_x, sc_y)

# 検証データも標準化
sc_x_val = sc_model_x.transform(x_val)
sc_y_val = sc_model_y.transform(y_val)

print(model.score(sc_x_val, sc_y_val))


def learn(x, t):
    x_train, x_val, y_train, y_val = train_test_split(
        x, t, test_size=0.2, random_state=0)

    # 訓練データ 標準化
    sc_model_x = StandardScaler()
    sc_model_x.fit(x_train)
    sc_x_train = sc_model_x.transform(x_train)

    sc_model_y = StandardScaler()
    sc_model_y.fit(y_train)
    sc_y_train = sc_model_y.transform(y_train)

    # 訓練データ 学習
    model = LinearRegression()
    model.fit(sc_x_train, sc_y_train)

    # 検証データ 標準化
    sc_x_val = sc_model_x.transform(x_val)
    sc_y_val = sc_model_y.transform(y_val)

    # score
    train_score = model.score(sc_x_train, sc_y_train)
    val_score = model.score(sc_x_val, sc_y_val)

    return train_score, val_score, model


feature_col = ['RM', 'LSTAT', 'PTRATIO']
x = train_val3.loc[:, feature_col]
t = train_val3[['PRICE']]

s1, s2, model = learn(x, t)
print(s1, s2)

# 多項式特徴量
# 回帰分析では

x['RM2'] = x['RM'] ** 2
x['LSTAT2'] = x['LSTAT'] ** 2
x['PTRATIO2'] = x['PTRATIO'] ** 2

# print(x.head(2))

s1, s2, model = learn(x, t)
print(s1, s2)

x['RM * LSTAT'] = x['RM'] * x['LSTAT']

s1, s2, model = learn(x, t)
print(s1, s2)

# 再学習
sc_model_x2 = StandardScaler()
sc_model_x2.fit(x)
sc_x = sc_model_x2.transform(x)

sc_model_y2 = StandardScaler()
sc_model_y2.fit(t)
sc_y = sc_model_y2.transform(t)
model = LinearRegression()
model.fit(sc_x, sc_y)
print(model.score(sc_x, sc_y))

# テストデータの前処理
test2 = test.fillna(train_val.mean())
x_test = test2.loc[:, feature_col]
y_test = test2[['PRICE']]

x_test['RM2'] = x_test['RM'] ** 2
x_test['LSTAT2'] = x_test['LSTAT'] ** 2
x_test['PTRATIO2'] = x_test['PTRATIO'] ** 2
x_test['RM * LSTAT'] = x_test['RM'] * x_test['LSTAT']
# 標準化
sc_x_test = sc_model_x2.transform(x_test)
sc_y_test = sc_model_y2.transform(y_test)
print(model.score(sc_x_test, sc_y_test))
