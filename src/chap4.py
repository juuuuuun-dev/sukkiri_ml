import pickle

import pandas as pd
from sklearn import tree

print("sukkiri ml")

data = {
    "matsuda": [160, 160],
    "asagi": [161, 175]
}

df = pd.DataFrame(data)
df.index = ["4月", "5月"]

df = pd.read_csv('./sukkiri-ml-codes/datafiles/KvsT.csv')
# print(df.head(3))
col = ["身長", "体重"]
# print(df[col])

# 特徴量
xcol = ["身長", "体重", "年代"]
x = df[xcol]
# print(x)
# 正解データ
t = df['派閥']
# modelの学習
model = tree.DecisionTreeClassifier(random_state = 1)
model.fit(x, t)

# 新しいデータで予測
taro = [[170, 70, 20]]
predict = model.predict(taro)
# print(predict)

# multi
matsuda = [172, 65, 20]
asagi = [158, 48, 20]
# print(model.predict([matsuda, asagi]))

# 正解率 1.0で100%
print(model.score(x, t))

# pickle モデルオブジェクトの直列化(バイト化,シリアライズ化)として保存し、再利用できるようにする
filename = 'KinokoTakenoko.pkl'
with open(filename, 'wb') as f:
    pickle.dump(model, f)

with open(filename, 'rb') as f:
    model2 = pickle.load(f)

suzuki = [180, 75, 30]
print(model2.predict([suzuki]))


#4.8
data = {
    "データベースの試験得点": [70, 72, 75, 80],
    "ネットワークの試験得点": [80, 85, 79, 92]
}
index = ["一郎", "二郎", "三郎", "太郎"]

df = pd.DataFrame(data)
df.index=index
print(df)

df = pd.read_csv('./sukkiri-ml-codes/datafiles/ex1.csv')
print(df.index)
print(df.columns)
print(df[["x0", "x2"]])