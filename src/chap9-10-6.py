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

# 改善案1 線形回帰で穴埋め
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

# 30000以上のがある
print(distance[0:3])
no = distance[distance > 30000].index
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
