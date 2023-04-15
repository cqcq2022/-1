#1 导入相应模块
import pandas as pd   # pandas库
import numpy as np   # numpy库
import sklearn    # python的sklearn库
import matplotlib as mpl
import matplotlib.pyplot as plt # 画图
from sklearn.linear_model import LinearRegression   # 线性回归模型
from sklearn.model_selection import train_test_split  # 用于划分数据集
from sklearn.preprocessing import StandardScaler   # 数据标准化

'''设置后中文字符 防止乱码'''

# 设置字符集，防止中文乱码（图中有中文，需要加上这个）
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
# # 读取txt文件
# data = pd.read_table('1.txt', sep=';')
#
# # 将数据保存为csv文件
# data.to_csv('data.csv', index=False)
df=pd.read_csv('data.csv')
print('这是打印的前五行数据：')
print('头一行数据： 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率')
new_df = df.replace('?',np.nan) # 替换非法字符为np.nan
datas = new_df.dropna(axis=0,how='any') # 只要有一个数据为空，就进行行删除
print('这是删除非法字符后的数据：')
print(datas)
print('这是数据的多种统计指标：',datas.describe())

names=['Date','Time','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']

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
Y = datas[names[4]]  #或 Y = df['Voltage']
print('-----------已经完成创建一个时间字符串格式化----------')
names=['Date','Time','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']

# 5对数据集进行训练集、测试集划分  from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # 测试集为20%

print('----------------已经完成对数据集进行训练集、测试集划分-----------------')

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
# 8模型检验
print ("模型检验的准确率准确率:",lr.score(X_test, Y_test))
#模型的检验
print('--------------------------------------正在运行模型的保存和加载------------------------------------')
import joblib
joblib.dump(ss,'data_biaozhun.model')  # 将标准化模型保存
joblib.dump(lr,'data_tv.model') # 将模型保存
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
plt.title(u'线性回归预测时间与电压的关系',fontsize=20)
# plt.grid(b=True)
plt.show()


# 数据集中取出用电器使用情况 与电流'Sub_metering_1','Sub_metering_2','Sub_metering_3'
X = datas[names[6:8]]
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
print ("用电器预测准确率: ", lr2.score(X2_test,Y2_test))
print ("用电器使用参数参数:", lr2.coef_)


# 画图
t=np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, Y2_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc = 'lower right')
plt.title(u"各用电器的使用情况与电流之间的关系", fontsize=20)
plt.show()