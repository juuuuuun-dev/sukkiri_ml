import pandas as pd

print("sukkiri ml")

data = {
    "matsuda": [160, 160],
    "asagi": [161, 175]
}

df = pd.DataFrame(data)
df.index = ["4月", "5月"]

df = pd.read_csv('./sukkiri-ml-codes/datafiles/KvsT.csv')
# print(df.head(3))
col = ["身長", "体重"]
print(df[col])