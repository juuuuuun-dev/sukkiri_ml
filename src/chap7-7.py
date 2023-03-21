import pandas as pd
from sklearn import tree

df = pd.read_csv('./sukkiri-ml-codes/datafiles/ex4.csv')
print(df.head(3))
print(df['sex'].mean())
print(df.groupby('class').mean()['score'])

ptable = pd.pivot_table(df, index='class', columns='sex', values='score')
print(ptable)
# 3, 1

x = df.loc[:, 'class':'score']
dept = pd.get_dummies(df['dept_id'], drop_first=True, prefix='dept')
x_temp = pd.concat([x, dept], axis=1)
print(x_temp)
x_new = x_temp.drop('dept_id', axis=1)
print(x_new)
