# 回帰 予測性能評価
# MAE 平均絶対誤差。差分の絶対値
# 数値が低いほど性能がよく、外れ値の影響を受けにくい


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math


yosoku = [2, 3, 5, 7, 11, 13]
target = [3, 5, 8, 11, 16, 19]

mse = mean_squared_error(yosoku, target)
print('rmse:{}'.format(math.sqrt(mse)))
print(f'mae:{format(mean_absolute_error(yosoku,target))}')

print("外れ値の混入")
yosoku = [2, 3, 5, 7, 11, 13, 46]
target = [3, 5, 8, 11, 16, 19, 23]

mse = mean_squared_error(yosoku, target)
print('rmse:{}'.format(math.sqrt(mse)))
print(f'mae:{format(mean_absolute_error(yosoku,target))}')
