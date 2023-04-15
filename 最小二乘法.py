import numpy as np

# 输入数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.2, 8.1, 10.1])

# 计算最小二乘法拟合直线的斜率和截距
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
xy_mean = np.mean(x * y)
x_square_mean = np.mean(x ** 2)
k = (xy_mean - x_mean * y_mean) / (x_square_mean - x_mean ** 2)
b = y_mean - k * x_mean

# 输出拟合直线的方程
print("拟合直线的方程为：y = {:.2f}x + {:.2f}".format(k, b))
