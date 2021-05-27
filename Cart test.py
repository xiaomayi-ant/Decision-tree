from sklearn.datasets  import   load_wine
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pydotplus
import matplotlib
import os
import pandas as pd
import operator
import joblib
import  numpy as np

# 导入路径
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
#设置字体
matplotlib.rcParams['font.sans-serif']=['SimHei']  #用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False   #正常显示负号

#划分训练集和测试集   顺序：训练集，测试集，训练集标签（类别），测试集标签（类别） 同时确定最优剪枝数 超参数学习曲线
fn=r'C:\Users\ant.zheng\Desktop\dataset\Regression\regre.xlsx'
wine = pd.read_excel(fn,sheet_name='Sheet1')
# wine=wine[['费用','展示次数','点击次数','点击率','安装次数','每次安装费用','评价']]
wine=wine[['Spend','CTR','CVR','CPA','PCVR','PCPA','Impression','Clicks','Prime','Install','SCPA','type']]
wine = wine.fillna(0)
m, n = wine.shape
test=[]
newtest={}
newsortedtest={}
Xtrain,Xtest,Ytrain,Ytest=train_test_split(wine.iloc[:,:n-1],wine.iloc[:,n-1],test_size=0.3,random_state=1)
for i  in  range(10):
#建立模型
    clf=tree.DecisionTreeClassifier(criterion='gini',splitter='best',random_state=1,max_depth=i+1)  #实例化，全部选择默认参数
    max_depth=i+1
    clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)

    test.append(score)
    if max_depth not in newtest.keys():
        newtest[max_depth]=score
print(newtest)
sortedtest=sorted(newtest.items(),key=operator.itemgetter(1),reverse=True)
for  i  in  sortedtest:
    newsortedtest[i[0]]=i[1]
newsortedtest=sorted(newsortedtest.items(),key=operator.itemgetter(1),reverse=True)
a=newsortedtest[0][0]
print('最佳深度:',a)
plt.plot(range(1,11),test,color="red",label="max_depth")
plt.legend()
plt.show()

#-----------------------------------------------------------------------------------------
#建立模型
clf=tree.DecisionTreeClassifier(criterion='gini',splitter='random',random_state=1,max_depth=a)  #实例化，全部选择默认参数
clf.fit(Xtrain,Ytrain)

#决策树模型评分
from   sklearn.metrics import  accuracy_score
score=accuracy_score(Ytest,clf.predict(Xtest))
print("CART模型准确率为：%.f%%" %(score*100))


def gra(clf):
    #使用graphivz画出图形
    import graphviz
    f_n=wine.columns.values.tolist()
    feature_name=['Spend','CTR','CVR','CPA','PCVR','PCPA','Impression','Clicks','Prime','Install','SCPA']
    class_name=['good','bad']
    dot_data=tree.export_graphviz(clf,out_file=None,feature_names=feature_name,class_names=class_name,filled=True,rounded=True,special_characters=True)
    with open('dot_data.txt','w',encoding='utf-8') as f:
        f.writelines(dot_data)
    print([*zip(feature_name,clf.feature_importances_)])
    import codecs
    txt_dir = 'dot_data.txt'
    txt_dir_utf8 = 'dot_data_utf8.txt'

    with codecs.open(txt_dir, 'r',encoding='utf-8') as f, codecs.open(txt_dir_utf8, 'w', encoding='utf-8') as wf:
        for line in f:
            lines = line.strip().split('\t')
            # print(lines)
            if 'label' in lines[0]:
                newline = lines[0].replace('\n', '').replace(' ', '')
            else:
                newline = lines[0].replace('\n','').replace('helvetica', ' "Microsoft YaHei" ')
            wf.write(newline + '\t')
    return txt_dir_utf8

#使用graphivz画出图形
import graphviz
f_n=wine.columns.values.tolist()
a=len(f_n)
class_name=['good','bad']
dot_data=tree.export_graphviz(clf,out_file=None,feature_names=f_n[:a-1],class_names=class_name,filled=True,rounded=True,special_characters=True)
with open('dot_data.txt','w',encoding='utf-8') as f:
    f.writelines(dot_data)
print('各项特征及其重要性:')
print([*zip(f_n[:a-1],np.around(clf.feature_importances_,decimals=3))])   #feature_importances_  计算逻辑:各节点样本数*该节点gini值,累加,除以根节点总样本数,最后各个特征之间进行归一化

import codecs
txt_dir = 'dot_data.txt'
txt_dir_utf8 = 'dot_data_utf8.txt'

with codecs.open(txt_dir, 'r',encoding='utf-8') as f, codecs.open(txt_dir_utf8, 'w', encoding='utf-8') as wf:
    for line in f:
        lines = line.strip().split('\t')
        # print(lines)
        if 'label' in lines[0]:
            newline = lines[0].replace('\n', '').replace(' ', '')
        else:
            newline = lines[0].replace('\n','').replace('helvetica', ' "Microsoft YaHei" ')
        wf.write(newline + '\t')

import pydotplus
with open('dot_data_utf8.txt', encoding='utf-8') as f:
    dot_graph = f.read()
graph = pydotplus.graph_from_dot_data(dot_graph)
filename = r"C:\Users\ant.zheng\Desktop\Cart.pdf"
graph.write_pdf(filename)  # 生成png文件
print("完成")


