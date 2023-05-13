# K 分割交差検証
# ホールドアウト方の問題の解決
# 検証データと訓練データで外れ値などの偏りを防ぐ
# ロジックはデータをランダムにn分割する
# 分割したうちの一つのデータを検証データ、残りを訓練データとして学習し、予測性能を出す
# 終わったら検証データを変えて合計n回繰り返す
# 最終的にn回分の平均を出す
# nは3-5の場合が多い

# なおモデルによってmoduleを使い分ける必要がある
# 回帰 from sklearn.model_selection import KFold
# 分類 from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression

import pandas as pd
df = pd.read_csv("./sukkiri-ml-codes/datafiles/cinema.csv")

df = df.fillna(df.mean())
x = df.loc[:, 'SNS1': 'original']
t = df['sales']

kf = KFold(n_splits=3, shuffle=True, random_state=0)
model = LinearRegression()
result = cross_validate(
    model,
    x,
    t,
    cv=kf,
    scoring='r2',  # 決定係数
    return_train_score=True)
print(pd.DataFrame(result))

# 検証データの決定計数の平均を出す
s = sum(result['test_score']) / len(result['test_score'])
print(s)
