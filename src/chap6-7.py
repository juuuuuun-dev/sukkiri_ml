import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./sukkiri-ml-codes/datafiles/ex3.csv")
print(df.head(5))

print(df.isnull().any(axis=0))
df2 = df.fillna(df.median())


plt.plot(df2['x0'], df2['target'])
plt.savefig('./img/chap6-7.png')

for name in df.columns:
    if name == 'target':
        continue
    plt.plot(df2[name], df2['target'])
    plt.savefig(f'./img/chap6-7{name}.png')
    plt.clf()
    
no = df2[(df2['x2'] < -2) & (df2['target'] > 100)].index
print(no)
df3 = df2.drop(no, axis=0)
plt.plot(df3['x2'], df3['target'])
plt.savefig('./img/chap6-7x2-after.png')

x = df3.loc[:, 'x0':'x3']
t = df3['target']
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))