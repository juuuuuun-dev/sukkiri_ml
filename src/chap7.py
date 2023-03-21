import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pprint
import pickle

df = pd.read_csv('./sukkiri-ml-codes/datafiles/Survived.csv')
print(df.head(2))
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())

# 決定木では多少の外れ値はそれほど影響をうけない

col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
x = df[col]
t = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(
    x, t, test_size=0.2, random_state=0)
print(x_train.shape)

# model
# 不均衡データの場合はclass_weight='balanced'を設定
model = tree.DecisionTreeClassifier(
    max_depth=5,
    random_state=0,
    class_weight='balanced')
model.fit(x_train, y_train)
print(model.score(X=x_test, y=y_test))

# max_depthが深いと過学習が起こる。
# 過学習が起こると特に重要ではない条件が判定されてしまうのでscoreが下がる
# 過学習を回避するためには
# - データ数を増やす
# - データの前処理方法を変える
# - モデルの学習時の設定を変える
# - モデル自体を変える

# 前処理を変える
df2 = pd.read_csv('./sukkiri-ml-codes/datafiles/Survived.csv')

print(df2.groupby('Pclass').mean()['Age'])

# ピボットテーブル(クロス集計)はデフォルトで平均を返す
ptable = pd.pivot_table(df2, index='Survived', columns='Pclass', values='Age')
print(ptable)

# 最大値
ptable = pd.pivot_table(
    df2,
    index='Survived',
    columns='Pclass',
    values='Age',
    aggfunc='max')
print(ptable)

# 穴埋めを各グループ化して行う
is_null = df2['Age'].isnull()

# pivotで計算された pclas=1 survived=0 のAge平均を代入
df2.loc[(df2['Pclass'] == 1) & (
    df2['Survived'] == 0) & (is_null), 'Age'] = 43
# p=1, s=1
df2.loc[(df2['Pclass'] == 1) & (
    df2['Survived'] == 1) & (is_null), 'Age'] = 35

# p=2, s=0
df2.loc[(df2['Pclass'] == 2) & (
    df2['Survived'] == 0) & (is_null), 'Age'] = 33
# p=2, s=1
df2.loc[(df2['Pclass'] == 2) & (
    df2['Survived'] == 1) & (is_null), 'Age'] = 25

# p=3, s=0
df2.loc[(df2['Pclass'] == 3) & (
    df2['Survived'] == 0) & (is_null), 'Age'] = 26
# p=3, s=1
df2.loc[(df2['Pclass'] == 3) & (
    df2['Survived'] == 1) & (is_null), 'Age'] = 20

col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
x = df2[col]
t = df2['Survived']


def learn(x, t, depth=3):
    x_train, x_test, y_train, y_test = train_test_split(
        x, t, test_size=0.2, random_state=0)
    model = tree.DecisionTreeClassifier(
        max_depth=depth,
        random_state=0,
        class_weight='balanced')

    model.fit(x_train, y_train)
    score = model.score(X=x_train, y=y_train)
    score2 = model.score(X=x_test, y=y_test)
    return round(score, 3), round(score2, 3), model


for j in range(1, 15):
    s1, s2, m = learn(x, t, depth=j)
    sentence = '深さ{}: 訓練データの精度{}: テストデータの精度{}'
    # print(sentence.format(j, s1, s2))

sex = df2.groupby('Sex').mean(numeric_only=True)
# pprint.pprint(vars(nump))

plt.bar(sex.index.to_numpy(), sex['Survived'].to_numpy())
plt.savefig('./img/chap7-20.png')

col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']
x = df2[col]
t = df2['Survived']

# train_score, test_score, model = learn(x, t)

# 特徴量に文字列の値は設定できない
male = pd.get_dummies(df2['Sex'])
print(male)

emb = pd.get_dummies(df2['Embarked'], drop_first=False)
print(emb)

print(x)
x_temp = pd.concat([x, male], axis=1)
print(x_temp)

x_new = x_temp.drop('Sex', axis=1)

for j in range(1, 10):
    s1, s2, m = learn(x_new, t, depth=j)
    sentence = '深さ{}: 訓練データの精度{}: テストデータの精度{}'
    print(sentence.format(j, s1, s2))

# score85以上が目安なので
s1, s2, model = learn(x_new, t, depth=5)
with open('./pkl/survived.pkl', 'wb') as f:
    pickle.dump(model, f)

# 特徴量重要度の確認
f = model.feature_importances_
print(f)
fp = pd.DataFrame(f, index=x_new.columns)
print(fp)
