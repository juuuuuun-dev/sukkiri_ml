import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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

# 重回帰分析モデル
model = LinearRegression()
model.fit(x_train, y_train)

# 予測する
new = [[150, 700, 300, 0]]
# 165, 350, 90
print(model.predict(new))

# 評価 平均二乗誤差 平均を2乗してから平均をだす
# 一般的に0.8以上で良い予測性能
print(model.score(x_test, y_test))
# score()は決定木の場合は正解率だが、回帰モデルの場合は平均二乗誤差を計算してくれる

# モデルの保存
with open('chimema.pkl', 'wb') as f:
    pickle.dump(model,f)

# predictの計算に使用した、各カラム係数 各カラムの数値に掛ける値
print(model.coef_)
# 定数項 次数が0 掛ける係数がない 足すだけ
print(model.intercept_)

# 次数　⇒　掛け合わせた文字の個数

tmp = pd.DataFrame(model.coef_)
tmp.index = x_train.columns
print(tmp)

# newで実際の計算 小数点2位を四捨五入
# new = [[150, 700, 300, 0]]
# (1.1 * 150) + (0.5 * 700) + (0.3 * 300) + (214 * 0 ) + 6253


