from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# ランダムフォレスト (アンサンブル学習)
# アンサンブル学習には大きく分けてバギングとブースティングという手法がある
# ランダムフォレストは広義のバギングの一つ
# バギングとは訓練データをブートスタップサンプリング(復元抽出/一度抽出したものを戻して、再度抽出対象にする)
# ランダムフォレストは特徴量の列もランダムに選択するバギングを少し拡張したもの
# バギングで利用されているのは、ほぼほぼランダムフォレスト

df = pd.read_csv("./sukkiri-ml-codes/datafiles/Survived.csv")
print(df['Age'].isnull().sum())

jo1 = df['Pclass'] == 1
jo2 = df['Survived'] == 0
jo3 = df['Age'].isnull()

df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 43

jo2 = df['Survived'] == 1
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 35

jo1 = df['Pclass'] == 2
jo2 = df['Survived'] == 0
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 26

jo2 = df['Survived'] == 1
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 20

jo1 = df['Pclass'] == 3
jo2 = df['Survived'] == 0
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 43

jo2 = df['Survived'] == 1
df.loc[(jo1) & (jo2) & (jo3), 'Age'] = 35

print(df['Age'].isnull().sum())

col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
x = df[col]
t = df['Survived']

dummy = pd.get_dummies(df['Sex'], drop_first=True)

x = pd.concat([x, dummy], axis=1)
print(x.head(2))

x_train, x_test, y_train, y_test = train_test_split(
    x, t, test_size=0.2, random_state=0)
# n_estimators 決定木の数
model = RandomForestClassifier(n_estimators=200, random_state=0)

model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

model2 = tree.DecisionTreeClassifier(random_state=0)
model2.fit(x_train, y_train)
print(model2.score(x_train, y_train))
print(model2.score(x_test, y_test))


# アンサンブル学習の二つ目、アダブースト
# 逐次的に各モデルが一つづつ学習していき、前モデルの情報を踏まえて再学習していく
# 最後のモデルの予測性能がよくなるわけではく、予測するさいは各モデルの正解率に重みがついた票で多数決を行い、予測する


x_train, x_test, y_train, y_test = train_test_split(
    x, t, test_size=0.2, random_state=0)

# アダブーストのベースモデル
base_model = DecisionTreeClassifier(random_state=0, max_depth=5)
# ベースモデルから500個逐次的作成
ada_model = AdaBoostClassifier(
    n_estimators=500,
    random_state=0,
    base_estimator=base_model)
ada_model.fit(x_train, y_train)

print(ada_model.score(x_train, y_train))
print(ada_model.score(x_test, y_test))
