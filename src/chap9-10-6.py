from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('./sukkiri-ml-codes/datafiles/Bank.csv')
print(df.shape)
df.head()
str_col_name = [
    'job',
    'default',
    'marital',
    'education',
    'housing',
    'loan',
    'contact',
    'month']
str_df = df[str_col_name]
str_df2 = pd.get_dummies(str_df, drop_first=True)

num_df = df.drop(str_col_name, axis=1)  # 数値列を抜き出す
# 結合(今後の集計の利便性も考慮してstr_dfも結合しておく)
df2 = pd.concat([num_df, str_df2, str_df], axis=1)

# 訓練&検証データとテストデータに分割
train_val, test = train_test_split(df2, test_size=0.1, random_state=9)
print(train_val.head())

is_nan = train_val.isnull().sum()
print(is_nan)
print(is_nan[is_nan > 0])

# 改善案12


# corr 特徴量の相関係数
corr_duration = train_val.corr()['duration']
print(corr_duration)
# 絶対値にして降順
corr_duration = corr_duration.map(abs).sort_values(ascending=False)
print(corr_duration)

# 外れ値をマハラノビス距離で検証
num_df = train_val.drop(str_col_name, axis=1)
num_df = num_df.drop('id', axis=1)
num_df2 = num_df.dropna()
mcd2_path = './pkl/chap9-10-6-mcd2.pkl'

# pkl
# mcd2 = MinCovDet(random_state=0, support_fraction=0.7)
# mcd2.fit(num_df2)

# with open(mcd2_path, mode='wb') as fp:
#     pickle.dump(mcd2, fp)

# pkl 復元
with open(mcd2_path, mode="rb") as fp:
    mcd_2 = pickle.load(fp)

# 箱髭で外れ値の確認
distance = mcd_2.mahalanobis(num_df2)
distance = pd.Series(distance)
plt.boxplot(distance)
image_path = f"./img/chap9-10-6-box.png"
plt.savefig(image_path)
plt.cla()

# 300000以上のがある
print(distance[0:3])
no = distance[distance > 300000].index
print(no)
print(distance[2561])

# 削除してもうもう一度モデル
no = num_df2.iloc[no[0]:(no[0] + 1), :].index
print(no)
train_val2 = train_val.drop(no)
print(train_val2)
corr_duration = train_val2.corr()['duration'].map(
    abs).sort_values(ascending=False)

num_df2 = train_val2.drop(str_col_name, axis=1)
num_df2 = num_df2.drop('id', axis=1)
num_df3 = num_df2.dropna()
mcd3_path = './pkl/chap9-10-6-mcd3.pkl'

# pkl
# mcd3 = MinCovDet(random_state=0, support_fraction=0.7)
# mcd3.fit(num_df3)

# with open(mcd3_path, mode='wb') as fp:
#     pickle.dump(mcd3, fp)

with open(mcd3_path, mode="rb") as fp:
    mcd3 = pickle.load(fp)

distance2 = mcd3.mahalanobis(num_df3)
distance2 = pd.Series(distance2)
plt.boxplot(distance2)
image_path = f"./img/chap9-10-6-box-2.png"
plt.savefig(image_path)
plt.cla()

# 次はdistance2の最大と最小の閾値を計算してdrop
tmp = distance2.describe()
print(tmp)
irq = tmp['75%'] - tmp['25%']
upper = (1.5 * irq) + tmp['75%']
lower = tmp['25%'] - (1.5 * irq)

outline = distance2[(distance2 > upper) | (distance2 < lower)].index
print(outline)
inline = distance2.drop(distance2.index[outline])
print(inline)

plt.boxplot(inline)
image_path = f"./img/chap9-10-6-box-2-inline.png"
plt.savefig(image_path)
plt.cla()

# 上限下限
# 0.7285823073004032 0.7116272405612037
train_val3 = train_val2.drop(train_val2.index[outline])
# train_val3 = train_val2  # 0.7271253237617876 0.7154980171805077
print(train_val3)
corr_duration3 = train_val3.corr()['duration'].map(
    abs).sort_values(ascending=False)

not_nan_df = train_val3.dropna()
temp_t = not_nan_df['duration']
temp_x = not_nan_df[['housing_yes', 'loan_yes',
                     'age', 'marital_single', 'job_student']]

model_liner = LinearRegression()
a, b, c, d = train_test_split(temp_x, temp_t, random_state=0, test_size=0.2)

model_liner.fit(a, c)
# 0.7285823073004032 0.7116272405612037
print(model_liner.score(a, c), model_liner.score(b, d))

# 線形回帰 欠損値を埋める
is_null = train_val2['duration'].isnull()

non_x = train_val2.loc[is_null, ['housing_yes',
                                 'loan_yes', 'age', 'marital_single', 'job_student']]
pred_d = model_liner.predict(non_x)
print(pred_d)
train_val2.loc[is_null, 'duration'] = pred_d
plt.hist(train_val2.loc[train_val['y'] == 0, "duration"])
image_path = f"./img/chap9-10-6-hist-0.png"
plt.savefig(image_path)
plt.cla()
plt.hist(train_val2.loc[train_val['y'] == 1, "duration"])
image_path = f"./img/chap9-10-6-hist-1.png"
plt.savefig(image_path)
plt.cla()


def learn(x, t, i):
    x_train, x_val, y_train, y_val = train_test_split(
        x, t, test_size=0.2, random_state=13)
    datas = [x_train, x_val, y_train, y_val]
    model = tree.DecisionTreeClassifier(
        random_state=i, max_depth=i, class_weight="balanced")
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)

    val_score = model.score(x_val, y_val)
    return train_score, val_score, model, datas


t = train_val2['y']
x = train_val2.drop(str_col_name, axis=1)
x = x.drop(['id', 'y', 'day'], axis=1)

for i in range(1, 15):
    s1, s2, model, datas = learn(x, t, i)
    print(i, s1, s2)

# depthは10に
# テストデータ
test2 = test.copy()
isnull = test2['duration'].isnull()
model_tree = tree.DecisionTreeClassifier(
    random_state=10, max_depth=10, class_weight="balanced")

if isnull.sum() > 0:
    temp_x = test2.loc[isnull, ['housing_yes', 'loan_yes',
                                'age', 'marital_single', 'job_student']]
    temp_y = test['duration']
    
    model_tree.fit(temp_x, temp_y)
    pred_d = model_tree.predict(temp_x)
    test2.loc[isnull, 'duration'] = pred_d
x_test = test2.drop(str_col_name, axis=1)
x_test = x_test.drop(['id', 'y', 'day'], axis=1)
y_test = test['y']

print(model.score(x_test, y_test))
