import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

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
# 票价高生存率高
##ageFacet=sns.FacetGrid(train,hue="Survived",aspect=3)
##ageFacet.map(sns.kdeplot,"Fare",shade=True)
##ageFacet.set(xlim=(0,150))
##ageFacet.add_legend()
##plt.show()

# 查看票价分布
# fare的分布呈左偏的形态，其偏度skewness=4.37较大，说明数据偏移平均值较多，因此我们需要对数据进行对数化处理，防止数据权重分布不均匀。
##farePlot = sns.displot(full["Fare"][full['Fare'].notnull()],label = 'skewness:%.2f'%full["Fare"].skew())
##plt.show()

#对数化处理fare值
##full['Fare'] = full['Fare'].map(lambda x: np.log(x) if x > 0 else x)

#处理之后票价Fare分布
##farePlot = sns.histplot(full['Fare'][full['Fare'].notnull()],label = 'skewness:%.2f' %(full['Fare'].skew()))
##plt.title('skewness:%.2f' %(full['Fare'].skew()))
##plt.savefig('./10-Fare票价分布.png', dpi=200)
##plt.show()


# 数据预处理
# 填充缺失值
# 船舱缺失值改为U
full["Cabin"] = full["Cabin"].fillna("U")

# 港口缺失值填充
# 看一下港口有缺失值的行的数据
##print(full[full['Embarked'].isnull()])
##print(Embarked_null)
# 英国S港登船人数最多，所以把这两个缺失值改为英国S港
full["Embarked"] = full["Embarked"].fillna("S")

# 处理车费空值
# 看一下车费有缺失值的行的数据
##print(full[full['Fare'].isnull()])
# 查到他舱等级是3，年龄男性，年龄60.5(这个数据太少，排除)，S港登船，那么把和他一样的数据的人挑出来，取平均值填进去
price = full[(full['Pclass'] == 3) & (full['Embarked'] == 'S') &(full["Cabin"]=="U")]['Fare'].mean()
full["Fare"] = full["Fare"].fillna(price)
##print(full.info())


# 特征工程

full["Title"] = full["Name"].map(lambda x: x.split(",")[1].split(".")[0].strip())
# 查看Title数据分布
##print(full["Title"].value_counts())

# 将Title信息进行整合
TitleDict={}
TitleDict['Mr']='Mr'
TitleDict['Mlle']='Miss' # mlle是miss法语写法
TitleDict['Miss']='Miss'
TitleDict['Master']='Master'
TitleDict['Jonkheer']='Master' # 荷兰贵族头衔，master是对未成年男性的尊称
TitleDict['Mme']='Mrs'# mme是mrs法语写法
TitleDict['Ms']='Mrs'
TitleDict['Mrs']='Mrs'
TitleDict['Don']='Royalty'
TitleDict['Sir']='Royalty'
TitleDict['the Countess']='Royalty'
TitleDict['Dona']='Royalty' # Royalty是皇室成员
TitleDict['Lady']='Royalty'
TitleDict['Capt']='Officer' # officer是军官或专业人士
TitleDict['Col']='Officer'
TitleDict['Major']='Officer'
TitleDict['Dr']='Officer'
TitleDict['Rev']='Officer'

# 头衔数据更换
full["Title"] = full["Title"].map(TitleDict)# .map(TitleDict) 会遍历 Title 列中的每一个值，如果该值在 TitleDict 字典中存在，就替换为对应的值，如果不存在，就保持原样（或变成 NaN）。
##print(full["Title"].value_counts())

# 看头衔与存活的关系
# 男的和工作人员存活率低
##sns.barplot(data=full,x="Title",y="Survived")
##plt.show()


# 家庭成员数量与生还关系
# 家庭成员2-4生还率高
full['familyNum']=full['Parch']+full['SibSp'] + 1#查看familyNum与Survived
##sns.barplot(data=full,x='familyNum', y='Survived')
##plt.show()


# 把1放到一组，234放一组，剩下的放一组
# 我们按照家庭成员人数多少，将家庭规模分为小(O)、中(1)、大(2)三类:可以降低模型学习难度
def familysize(familyNum) :
    if familyNum== 1:
        return 0
    elif (familyNum>=2)&(familyNum<=4):
        return 1
    else:
        return 2
full['familySize']=full['familyNum'].map(familysize)
##print(full['familySize'].value_counts())

##sns.barplot(data=full,x='familySize',y='Survived')
##plt.show()
# 小家庭存活率高




# 船舱类型与生存关系
# 撞冰山的时候跟船舱的位置也有关系
full['Cabin'].unique()
##print(full['Cabin'].unique())

#提取Cabin字段首字母
full['Deck']=full['Cabin'].map(lambda x:x[0])#查看不同Deck类型乘客的生存率
##sns.barplot(data=full, x='Deck', y='Survived')
##plt.show()
# BDE生还率高，UT生还率低

# 同一票号的乘客数量可能不同，可能也与乘客生存率有关系
TickCountDict = full['Ticket'].value_counts()
##print(TickCountDict)



#将同票号乘客数量数据并入数据集中
full['Tickcom']=full['Ticket'].map(TickCountDict)
##print(full['Tickcom'].head())
#查看Tickcom与Survived之间关系
##sns.barplot(data=full,x='Tickcom',y='Survived')
##plt.show()

#按照Tickcom大小，将TickGroup分为三类
def TickCountGroup(num) :
    if (num>=2)&(num<=4):
        return 0
    elif (num==1) | ((num>=5)&(num<=8) ) :
        return 1
    else :
        return 2
#得到各位乘客TickGroup的类别
full['TickGroup']=full['Tickcom'].map(TickCountGroup)
#查看TickGroup与Survived之间关系
##sns.barplot(data=full,x='TickGroup',y='Survived')
##plt.show()
# 票里人数越多越不容易活



# 年龄数据处理
# 进行相关性分析
age_notnull = full[full['Age'].notnull()] # full['Age'] → 取出 Age 列 .notnull() → 返回一个布尔 Series，每个位置是 True 或 False。Age 有值 → True，Age 是缺失值（NaN）→ False
# 只选择数字类型的列
numeric_cols = age_notnull.select_dtypes(include=['number'])
print(numeric_cols.corr())
# 船舱等级似乎和年龄有些关系0.47


#筛选数据集
agePre = full[['Age', 'Parch', 'Pclass', 'SibSp', 'familyNum', 'Tickcom', 'Title']]
#进行one-hot编码
agePre = pd.get_dummies(agePre)
ageCorrDf = agePre.corr()
print(ageCorrDf['Age'].sort_values()) #  每个特征与 Age 的相关系数，按升序排列。






