import warnings
warnings.filterwarnings("ignore")
# 屏蔽所有警告信息

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置seaborn样式
sns.set(style="white", context="notebook", palette="muted")

# 导入数据
train = pd.read_csv("数据分析/train.csv")
test = pd.read_csv("数据分析/test.csv")

# print("训练数据大小:",train.shape) #返回数据有多少行，多少列
# print("测试数据大小:",test.shape)
# print(train.head()) #显示前5行数据

# 数据合并（训练数据和测试数据合并）
# 目的通常是为了后续对全部数据（包括测试数据）进行统一的预处理（如特征工程、缺失值填充等），以确保训练和测试数据的处理逻辑完全一致。
full = pd.concat([train, test], ignore_index=True) # 使用pd.concat合并数据，ignore_index=True去掉test索引，然后拼接

#print(full.info())

# 对缺失值进行处理
# Survived，Age，Fare，Cabin，Embarked存在缺失值

# 看一下港口和生死之间的关系（有的港口穷，有的港口富有）
sns.barplot(data=train,x="Embarked",y="Survived")
# y轴显示的是平均值，比如C港口条形高度约为0.55，意味着从C港口登船的乘客中约55%幸存。
plt.show()


s = full.groupby("Embarked")["Survived"].value_counts().to_frame()
# full.groupby("Embarked")：按"Embarked"（登船港口）列对数据进行分组
# ["Survived"]：选择"Survived"（是否生存）列
# .value_counts()：统计每个分组中"Survived"列各个值（0=死亡，1=生存）的出现次数
# .to_frame()：将统计结果转换为DataFrame格式
##s2 = s/s.sum(level=0) # s.sum(level=0) 的含义，指定沿 第 0 层索引（Embarked） 进行聚合，即：把同一个港口下，存活=0 和存活=1 的数值加在一起
# s = 每个港口中存活/死亡的人数。s2 = 每个港口中存活/死亡的占比
##pd.merge(s,s2,left_index=True,right_index=True,suffixes=['_num',"_rate"])
print(s)