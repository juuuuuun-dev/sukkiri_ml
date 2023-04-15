from sklearn.covariance import MinCovDet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./sukkiri-ml-codes/datafiles/bike.tsv', sep="\t")
print(df.head(2))

weather = pd.read_csv(
    './sukkiri-ml-codes/datafiles/weather.csv',
    encoding=('shift-jis'))
print(weather.head(3))

temp = pd.read_json('./sukkiri-ml-codes/datafiles/temp.json')
temp = temp.T
print(temp.head(3))

# 結合 inner join
df2 = df.merge(weather, how='inner', on='weather_id')
print(df2.head(3))

# left join
df3 = df2.merge(temp, how='left', on='dteday')
print(df3.head(3))
print(df3[df3['dteday'] == '2011-07-20'])

plt.plot(df3[['temp', 'hum']])
image_path = f"./img/chap10-temp.png"
plt.savefig(image_path)
plt.cla()

plt.plot(df3['atemp'].loc[220:240])
# image_path = f"./img/chap10-atemp.png"
# plt.savefig(image_path)
# plt.cla()

# 欠損値を前後の値から線形保管 floatに変換してから
df3['atemp'] = df3['atemp'].astype(float)
df3['atemp'] = df3['atemp'].interpolate()
plt.plot(df3['atemp'].loc[220:240])
image_path = f"./img/chap10-atemp.png"
plt.savefig(image_path)
plt.cla()

# がく片長さに欠損値がある。がく片長さをpredictで予想して欠損値を埋める
iris_df = pd.read_csv('./sukkiri-ml-codes/datafiles/iris.csv')
non_df = iris_df.dropna()
print(non_df.head(2))
x = non_df.loc[:, 'がく片幅':'花弁幅']
t = non_df['がく片長さ']
model = LinearRegression()
model.fit(x, t)

condition = iris_df['がく片長さ'].isnull()
non_data = iris_df.loc[condition]
print(non_data)
x = non_data.loc[:, 'がく片幅':'花弁幅']
pred = model.predict(x)
print(pred)
iris_df.loc[condition, 'がく片長さ'] = pred
print(iris_df[137:138])


# マハラノビス距離 外れ値の計算
# マハラノビス距離とはデータ分布の特徴を考慮した中心からの距離
# 特徴を考慮しないとユーグリット距離

df4 = df3.loc[:, 'atemp':'windspeed']
df4 = df4.dropna()
print(df4)
mcd = MinCovDet(random_state=0, support_fraction=0.7)
mcd.fit(df4)
distance = mcd.mahalanobis(df4)
print(distance)

# 箱ひげ図 真ん中の箱は IQR/四分位範囲 25%で四分割の真ん中50%
# 上下のヒゲは残り25%
# このままだと外れ値が大きいので箱が見えない
plt.boxplot(distance)
image_path = f"./img/chap10-box.png"
plt.savefig(image_path)
plt.cla()

# シリーズに変換しdescribe()で列データの平均や最大値など計算できる
distance = pd.Series(distance)
print(distance)
tmp = distance.describe()
print(tmp)

# 外れ値の閾値
# 大 75%の数値 + 1.5 * IQR
# 小 25%の数値 - 1.5 * IQR
# IRQの計算
irq = tmp['75%'] - tmp['25%']
print(irq)
jougen = (1.5 * irq) + tmp['75%']
kagen = tmp['25%'] - (1.5 * irq)
print(jougen)
print(kagen)

# 外れ値
outline = distance[(distance > jougen) | (distance < kagen)].index
inline = distance.drop(distance.index[outline])
print(outline)
print(inline)
plt.boxplot(inline)
image_path = f"./img/chap10-box2.png"
plt.savefig(image_path)
plt.cla()
