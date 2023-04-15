#1 导入相应模块
import pandas as pd   # pandas库
import numpy as np   # numpy库
import sklearn    # python的sklearn库
import matplotlib as mpl
import matplotlib.pyplot as plt # 画图
from sklearn.linear_model import LinearRegression   # 线性回归模型
from sklearn.model_selection import train_test_split  # 用于划分数据集
from sklearn.preprocessing import StandardScaler   # 数据标准化
from sklearn.metrics import mean_squared_error
'''设置后中文字符 防止乱码'''

# 设置字符集，防止中文乱码（图中有中文，需要加上这个）
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

# 2读取数据
path='1.txt'  # 下载数据的路径  1000行数据
df = pd.read_csv(path,sep=';')
# 打印了前五行数据
print('这是打印的前五行数据：')
print('头一行数据： 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率')

df.head()
print(df.head())
# 查看格式信息
df.info()
print(df.info())
# 3异常数据处理（异常数据过滤）
new_df = df.replace('?',np.nan) # 替换非法字符为np.nan
datas = new_df.dropna(axis=0,how='any') # 只要有一个数据为空，就进行行删除
print('这是删除非法字符后的数据：')
print(datas)
datas.describe() # 观察数据的多种统计指标
print('这是数据的多种统计指标：')
print(datas.describe())

'''时间格式处理'''
# 4创建一个时间字符串格式化
def data_format(dt):
    import time
    t = time.strptime(' '.join(dt),'%d/%m/%Y %H:%M:%S')
    return (t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)

# 获取X，Y变量，X为时间，Y为功率，并将时间转换成数值型的连续变量
names=['Date','Time','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
# names为数据集中的字段名
print('----------------------正在完成创建一个时间字符串格式化--------------\n---------------------------------请稍等！！-------------------')
X = datas[names[0:2]]  #或 X = df.iloc[:,0:2]
X = X.apply(lambda x :pd.Series(data_format(x)),axis=1)
Y = datas[names[2]]  #或 Y = df['Global_active_power']
print('-----------已经完成创建一个时间字符串格式化----------')
names=['Date','Time','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']



# 描述性统计分析
datas[['Global_active_power','Date']].describe()
print(datas[['Global_active_power','Date']].describe())


# 读取数据
data = pd.read_csv('data.csv')

# 统计每个值的出现次数
counts = data['Global_active_power'].value_counts()

# 计算每个值的百分比
percent = counts / counts.sum()

# 绘制柱图
percent.plot(kind='bar')

# 设置百分比显示
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

# 显示图形
plt.show()

print('----------------已经对数据集进行训练集、测试集划分-----------------')
# 5对数据集进行训练集、测试集划分  from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # 测试集为20%
print('----------------已经完成对数据集进行训练集、测试集划分-----------------')
print(X_train.shape,X_test.shape)
class PLS:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        n_targets = Y.shape[0]

        # Initialize matrices
        T = np.zeros((n_samples, self.n_components))
        U = np.zeros((n_samples, self.n_components))
        P = np.zeros((n_features, self.n_components))
        Q = np.zeros((n_targets, self.n_components))
        W = np.zeros((n_features, self.n_components))
        B = np.zeros((self.n_components, n_targets))

        # Center data
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y, axis=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        for i in range(self.n_components):
            # Calculate weights
            w = np.dot(X_centered.T, Y_centered) / np.dot(Y_centered.T, Y_centered)
            w = w / np.linalg.norm(w)
            t = np.dot(X_centered, w)
            q = np.dot(Y_centered.T, t) / np.dot(t.T, t)

            # Calculate loadings
            p = np.dot(X_centered.T, t) / np.dot(t.T, t)
            P[:, i] = p
            q = q / np.linalg.norm(q)
            Q[:, i] = q

            # Calculate regression coefficients
            b = np.dot(t.T, Y_centered) / np.dot(t.T, t)
            B[i, :] = b

            # Deflate X and Y
            X_centered = X_centered - np.outer(t, p)
            Y_centered = Y_centered - np.outer(t, b)

            # Store scores
            T[:, i] = t
            U[:, i] = np.dot(X, p)
            W[:, i] = w

        self.X_mean = X_mean
        self.Y_mean = Y_mean
        self.T = T
        self.U = U
        self.P = P
        self.Q = Q
        self.W = W
        self.B = B

    def predict(self, X):
        X_centered = X - self.X_mean
        T = np.dot(X_centered, self.P)
        Y_pred = np.dot(T, self.B) + self.Y_mean
        return Y_pred
'''# Fit PLS model'''
pls = PLS(n_components=3)
pls.fit(X_train, Y_train)
Y_pred = pls.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
print("Mean squared error:", mse)