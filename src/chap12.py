from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./sukkiri-ml-codes/datafiles/iris.csv")
print(df.head(2))

# ロジスティック回帰
df_mean = df.mean()
train2 = df.fillna(df_mean)

x = train2.loc[:, :'花弁幅']
t = train2['種類']

sc = StandardScaler()
new = sc.fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(
    new, t, test_size=0.2, random_state=0)

# Cが小さいほど過学習を防げる
model = LogisticRegression(
    random_state=0,
    C=0.1,
    multi_class='auto',
    solver='lbfgs')

model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_val, y_val))
print(model.coef_)

x_new = [[1, 2, 3, 4]]
# 確率が一番大きい答えを返す
print(model.predict(x_new))

# 各答えの確率を返す
# setosa, versicolor, virginica
print(model.predict_proba(x_new))
