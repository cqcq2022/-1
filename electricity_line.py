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


# 6 数据标准化   from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
# fit是做运算，计算标准化需要的均值和方差，transform是进行转化
X_train = ss.fit_transform(X_train) # 训练并转换
X_test = ss.transform(X_test) ## 直接使用在模型构建数据上进行一个数据标准化操作
print('----------------已经完成对数据集进行数据标准化fit是做运算，计算标准化需要的均值和方差，transform是进行转化 -----------------')

# 7模型训练（线性模型）
lr = LinearRegression()
lr.fit(X_train, Y_train) ## 训练模型
y_predict = lr.predict(X_test)  # 预测
print('-----------------已经完成模型训练（线性模型）---------------------')


'''错误原因

'''
# 8模型检验
print ("模型检验的准确率准确率:",lr.score(X_test, Y_test))
print('这个准确率着实有点低了话说！')
print('在模型检验的时候老出现“TypeError: only size-1 arrays can be converted to Python scalars的问题\n”解决办法：numpy类型的转换需要更改为astype')
# while True:
#     mse = np.average((np.array(y_predict).astype-np.array(Y_test).astype)**2)
#     rmse = np.sqrt(mse)
#     print ("MSE:" ,mse)
#     print ("RMSE:",rmse)
#     else:
#         print('TypeError: only size-1 arrays can be converted to Python scalars的问题暂时还没有结局，模型检验还没有成功')

'''模型的保存和加载'''
print('--------------------------------------正在运行模型的保存和加载------------------------------------')
import joblib
# 模型保存
joblib.dump(ss,'data_pi.model')  # 将标准化模型保存
joblib.dump(lr,'data_tp.model') # 将模型保存
# 加载模型
joblib.load('data_ss.model')
joblib.load('data_lr.model')
print('------------------------已经完成data_ss.model模型的保存和加载------------------------')


# 预测值与实际值画图比较
t = np.arange(len(X_test))
plt.figure(facecolor='w') # 建一个画布，facecolor是背景色
plt.plot(t,Y_test,'r-',linewidth=2,label=u'真实值')
plt.plot(t,y_predict,'g-',linewidth=2,label=u'预测值')
plt.legend(loc='lower right') # 显示图列，设置图列的位置
# best/upper right/upper left/lower left/lower right/right/center left/center right/lower center/upper center/center
plt.title(u'线性回归预测时间与功率的关系',fontsize=20)
# plt.grid(b=True)
plt.show()




# 数据集中取出功率与电流'Global_active_power','Global_reactive_power'和'Global_intensity'
X = datas[names[2:4]]
Y2 = datas[names[5]]
# 数据集划分
X2_train,X2_test,Y2_train,Y2_test = train_test_split(X, Y2, test_size=0.2, random_state=0)
# 数据标准化
scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)
X2_test = scaler2.transform(X2_test)
# 模型训练
lr2 = LinearRegression()
lr2.fit(X2_train, Y2_train)
# 预测
Y2_predict = lr2.predict(X2_test)

# 模型评估
print ("电流预测准确率: ", lr2.score(X2_test,Y2_test))
print ("电流参数:", lr2.coef_)
# 画图
t=np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, Y2_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc = 'lower right')
plt.title(u"线性回归预测功率与电流之间的关系", fontsize=20)
plt.show()
