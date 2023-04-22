from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./sukkiri-ml-codes/datafiles/Boston.csv")
df = df.fillna(df.mean())
df = df.drop([76], axis=0)

x_col = ['RM', 'PTRATIO', 'LSTAT']
t = df[['PRICE']]
x = df.loc[:, x_col]

print(t)

# 標準化
sc = StandardScaler()
sc_x = sc.fit_transform(x)
sc2 = StandardScaler()
sc_t = sc2.fit_transform(t)

print(sc_x.shape)
# 特徴量エンジニアリング すべて項目に対して追加してくれる PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
pf_x = pf.fit_transform(sc_x)
print(pf_x.shape)
print(pf.get_feature_names_out())

x_train, x_test, y_train, y_test = train_test_split(
    pf_x, sc_t, test_size=0.3, random_state=0)

# 線形回帰
model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
weight = model.coef_
# s = pd.Series(weight, index=pf.get_feature_names_out())
print(weight)

# リッジ回帰
ridgeModel = Ridge(alpha=10)
ridgeModel.fit(x_train, y_train)
print(ridgeModel.score(x_train, y_train))
print(ridgeModel.score(x_test, y_test))


# ラッソ回帰 不要な特徴量を削除した上で回帰式を作成するモデル
model = Lasso(alpha=0.1)
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
weight = model.coef_
s = pd.Series(weight, index=pf.get_feature_names_out())
print(s)
