# 教師無し学習
# 次元削減 傾向が似ている特徴量をまとめて新しい特徴量を作成し、特徴量を削減する
# 例) 数学,英語の点数 -> 二つの値から理数系点数を作成

# 主成分分析 次元削減のひとつ 線形代数

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
df = pd.read_csv("./sukkiri-ml-codes/datafiles/Boston.csv")
print(df.head(3))

df2 = df.fillna(df.mean())
dummy = pd.get_dummies(df2['CRIME'], drop_first=True)
df3 = df2.join(dummy)
df3 = df3.drop(['CRIME'], axis=1)
print(df3.head(3))

df4 = df3.astype('float')

sc = StandardScaler()
sc_df = sc.fit_transform(df4)

# 主成分分析
# n_components ベクトル数, whiten = 白色化
model = PCA(n_components=2, whiten=True)
model.fit(sc_df)
print(model.components_[0])
new = model.transform(sc_df)
new_df = pd.DataFrame(new)
print(new_df.head(3))

# 主成分負荷量 標準化済みデータと主成分得点の相関関係
new_df.columns = ['PC1', 'PC2']
df5 = pd.DataFrame(sc_df, columns=df4.columns)
df6 = pd.concat([df5, new_df], axis=1)
print(df6.head(3))

# 相関関係
df_corr = df6.corr()
df_corr.loc[:'very_low', 'PC1':]
print(df_corr)

pc_corr = df_corr.loc[:'very_low', 'PC1':]
print(pc_corr['PC1'].sort_values(ascending=False))
print(pc_corr['PC2'].sort_values(ascending=False))
