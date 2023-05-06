# 回帰 予測性能評価
# RMSE ２乗平均平方根誤差
# 平均二乗誤差の平方根をとったもの
# 外れ値があると影響を受けやすい

from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

df = pd.read_csv("./sukkiri-ml-codes/datafiles/cinema.csv")
df = df.fillna(df.mean())
# print(df.head(10))

x = df.loc[:, 'SNS1':'original']
t = df['sales']
model = LinearRegression()
model.fit(x, t)

pred = model.predict(x)
print(pred)
print(t)
# 平均二乗誤差 値が小さいほど予測性能がいい
mse = mean_squared_error(pred, t)
print(mse)
# RMSE (2乗した数値なので平方根をとる)
print(math.sqrt(mse))
