import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('./sukkiri-ml-codes/datafiles/Bank.csv')
print(df.head(2))

# ダミー化 文字列の値を数値に
# 複数あるのでカラムで抜き出す
str_col = [
    'job',
    'marital',
    'education',
    'default',
    'contact',
    'housing',
    'loan',
    'month',
]
str_df = df[str_col]
dummy_df = pd.get_dummies(str_df, drop_first=True)

# 文字列カラム削除してダミー追加
num_df = df.drop(str_col, axis=1)
df2 = pd.concat([num_df, dummy_df, str_df], axis=1)


print(df2.head(2))

# データ分割
train_val, test = train_test_split(df2, test_size=0.2, random_state=0)

# 穴埋めは分割後
print(train_val.isnull().sum())
train_val2 = train_val.fillna(train_val.mean())

# 不均衡の確認 答えの差が大きい
# 正解は均等ほど望ましい
# class_weight="balanced"を指定
print(train_val2['y'].value_counts())

# 正解データの用意と不要なカラムを削除
t = train_val2['y']
x = train_val2.drop(str_col, axis=1)
x = x.drop(['id', 'y', 'day'], axis=1)


# x_train, x_val, y_train, y_val = train_test_split(
#     x, t, test_size=0.2, random_state=0)

# model = tree.DecisionTreeClassifier(random_state=0, max_depth=)


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


# 一旦 精度の確認と特徴量の重要度を確認
for j in range(1, 10):
    s1, s2, m = learn(x, t, depth=j)
    sentence = '深さ{}: 訓練データの精度{}: テストデータの精度{}'
    print(sentence.format(j, s1, s2))
    a = pd.Series(
        m.feature_importances_,
        index=x.columns).sort_values(
        ascending=False)
    print(a[0:9])

# durationが最重要であることがわかる
# d durationに関係しているカラムを確認
# この関係からdurationの穴埋めを最適化
for name in str_df .columns:
    print(train_val.groupby(name)['y'].mean())
    print("------")

print(
    pd.pivot_table(
        train_val,
        index="housing",
        columns='loan',
        values="duration"))

print(
    pd.pivot_table(
        train_val,
        index="housing",
        columns='contact',
        values="duration"))

print(
    pd.pivot_table(
        train_val,
        index="loan",
        columns='contact',
        values="duration"))


# loanとhousingでのpivotの平均からdurationを穴埋め 2*2
def nan_fill(train_val):
    isnull = train_val['duration'].isnull()
    train_val2 = train_val.copy()
    train_val2.loc[(isnull) & (train_val2['housing'] == 'yes')
                   & (train_val2['loan'] == 'yes'), 'duration'] = 439
    train_val2.loc[(isnull) & (train_val2['housing'] == 'yes')
                   & (train_val2['loan'] == 'no'), 'duration'] = 332
    train_val2.loc[(isnull) & (train_val2['housing'] == 'no')
                   & (train_val2['loan'] == 'yes'), 'duration'] = 299
    train_val2.loc[(isnull) & (train_val2['housing'] == 'no')
                   & (train_val2['loan'] == 'no'), 'duration'] = 237
    return train_val2


train_val2 = nan_fill(train_val)
print(train_val2.isnull().sum())
t = train_val2['y']
x = train_val2.drop(str_col, axis=1)
x = x.drop(['id', 'y', 'day'], axis=1)

for j in range(1, 10):
    s1, s2, m = learn(x, t, depth=j)
    sentence = '深さ{}: 訓練データの精度{}: テストデータの精度{}'
    print(sentence.format(j, s1, s2))
    a = pd.Series(
        m.feature_importances_,
        index=x.columns).sort_values(
        ascending=False)
    print(a[0:9])


# depth7でテストデータでscoreを出す
model_tree = tree.DecisionTreeClassifier(
    max_depth=7,
    random_state=0,
    class_weight="balanced")
model_tree.fit(x, t)

test2 = nan_fill(test)
t = test['y']
x = test2.drop(str_col, axis=1)
x = x.drop(['id', 'y', 'day'], axis=1)

# 0.816 章ではここで終わり
print(model_tree.score(x, t))
