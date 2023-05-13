# 分類 予測性能
# 適合率 雨が降ると予測した件数のうち、実際に雨が降った件数の比率 (コスト重視)
# 再現率 実際に雨が降った件数のうち、雨が降ると予想した件数の比率 (リスク重視)
# 予測モデルによってどちらを重視すべきか判断する必要がある

from sklearn.metrics import classification_report
import pandas as pd
from sklearn import tree

df = pd.read_csv("./sukkiri-ml-codes/datafiles/Survived.csv")
df = df.fillna(df.mean())

x = df[['Pclass', 'Age']]
t = df['Survived']


model = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
model.fit(x, t)

pred = model.predict(x)
out_put = classification_report(y_pred=pred, y_true=t, output_dict=True)
# precision (適合率)
# recall (再現率)
# どちらも高いほうがいい
print(pd.DataFrame(out_put))
# print(type(out_put))
# f1-score 適合率と再現率の平均 適合率と再現率どちらも大事の場合に指標にする
