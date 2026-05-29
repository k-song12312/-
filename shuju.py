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
train = pd.read_csv("泰坦尼克数据/train.csv")
test = pd.read_csv("泰坦尼克数据/test.csv")

##print("训练数据大小:",train.shape) #返回数据有多少行，多少列
##print("测试数据大小:",test.shape)
##print(train.head()) #显示前5行数据

# 数据合并（训练数据和测试数据合并）
# 目的通常是为了后续对全部数据（包括测试数据）进行统一的预处理（如特征工程、缺失值填充等），以确保训练和测试数据的处理逻辑完全一致。
full = pd.concat([train, test], ignore_index=True) # 使用pd.concat合并数据，ignore_index=True去掉test索引，然后拼接

##print(full.info())

# 对缺失值进行处理
# Survived，Age，Fare，Cabin，Embarked存在缺失值

# 看一下港口和生死之间的关系（有的港口穷，有的港口富有）
##sns.barplot(data=train,x="Embarked",y="Survived")
# y轴显示的是平均值，比如C港口条形高度约为0.55，意味着从C港口登船的乘客中约55%幸存。
##plt.show()

# 统计每个港口中存活/死亡的人数
s = full.groupby("Embarked")["Survived"].value_counts().to_frame()
# full.groupby("Embarked")：按"Embarked"（登船港口）列对数据进行分组
# ["Survived"]：选择"Survived"（是否生存）列
# .value_counts()：统计每个分组中"Survived"列各个值（0=死亡，1=生存）的出现次数
# .to_frame()：将统计结果转换为DataFrame格式
# 黑线是误差棒（error bars）。seaborn.barplot 默认会显示 95% 置信区间的误差估计，用黑线表示。它展示的是：从该港口登船的幸存率平均值的波动范围，线越长说明数据越分散或样本越少，估计越不精确。

# 每个港口中存活/死亡的占比（结论C港口的存活率最高）
s2 = s/s.groupby(level=0).sum() # s.sum(level=0) 的含义，指定沿 第 0 层索引（Embarked） 进行聚合，即：把同一个港口下，存活=0 和存活=1 的数值加在一起
# s = 每个港口中存活/死亡的人数。s2 = 每个港口中存活/死亡的占比

# 将表s和s2合并
s3 = pd.merge(s,s2,left_index=True,right_index=True,suffixes=['_num',"_rate"])

# 绘制每个港口中每个船舱等级的乘客人数直方图
##sns.catplot(x="Pclass",col="Embarked",data=train,kind="count",height=3)
# 横坐标是pclass，纵坐标是Embarked，kind="count"表示统计绘制方法，height=3表示图高度为3
##plt.show()

# 绘制每个不同父母/子女人数的人的存活率柱状图
# 有一个的存活率第一高，有两个的第二高，，三个的置信区间太大
##sns.barplot(data=train,x="Parch",y="Survived")
##plt.show()

# 绘制每个不同兄弟姐妹/配偶人数的人的存活率柱状图
# 有一个的时候最高
##sns.barplot(data=train,x="SibSp",y="Survived")
##plt.show()

# 绘制每个船票等级的人的存活率柱状图
# 船票等级越高，生还率越高
##sns.barplot(data=train,x="Pclass",y="Survived")
##plt.show()

# 绘制不同性别的人的生存率柱状图
# 女性存活率远高于男性,女性0.7几，男性只有0.2
##sns.barplot(data=train,x="Sex",y="Survived")
##plt.show()

# 票价和死亡率的关系
ageFacet=sns.FacetGrid(train,hue="Survived",aspect=3)
ageFacet.map(sns.kdeplot,"Fare",shade=True)
ageFacet.set(xlim=(0,150))
ageFacet.add_legend()
plt.show()

##print(s)
##print("下一张表",s2)