import pandas as pd # pandas read csv
import numpy as np  #For save data to csv
from sklearn import decomposition #For PCA
from sklearn.svm import SVC #For SVM
from numpy import * #For 读取行数和列数Shape

#读取训练集train_data
print("=================Read data======================")
train_data=pd.read_csv('1,2,3,4,5 - train.csv')
trainDataSet= train_data.values[:, :20] #获取训练集1-20列(boneID)
trainLabel= train_data.values[:, 20:] #获取训练集21列(label)
print("Train_data size: ", len(trainDataSet),"X",shape(trainDataSet)[1]) #输出trainDataSet行数X列数

#读取测试集test_data
test_data=pd.read_csv('1,2,3,4,5 - test.csv')
testDataSet= test_data.values[:, :20] #获取测试集1-20列(boneID)
testLabel= test_data.values[:, 20:] #获取测试集21列(label)
print("Test_data size: ", len(testDataSet),"X",shape(testDataSet)[1]) #输出testDataSet行数X列数
print("=================Read end======================")


#train data 和 test data PCA 降维
print("=================Start PCA======================")
pca = decomposition.PCA(n_components=15)  #降到15维
trainDataSet = pca.fit_transform(trainDataSet) #用trainDataSet来训练PCA模型，同时返回降维后的数据
testDataSet = pca.fit_transform(testDataSet) #用testDataSet来训练PCA模型，同时返回降维后的数据
#print(testDataSet[:3])
np.savetxt('trainDataSet.csv', trainDataSet, delimiter = ',')  #把降维后的数据存成csv
np.savetxt('testDataSet.csv', testDataSet, delimiter = ',')  #把降维后的数据存成csv
print("PCA train_data size: ", len(trainDataSet),"X",shape(trainDataSet)[1]) #输出降维后的trainDataSet行数X列数
print("PCA test_data size: ", len(testDataSet),"X",shape(testDataSet)[1]) #输出降维后的testDataSet行数X列数
print("=================PCA Finish======================")


#用SVM监督学习来训练trainData
print("=================Start SVM======================")
clf = SVC() #创建分类器对象
clf.fit(trainDataSet, trainLabel) #用训练数据拟合分类器模型
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print(clf.predict(testDataSet)) #用训练好的分类器去预测testDataSet数据的标签
print(clf.score(testDataSet,testLabel)) #Returns the mean accuracy on the given test data and labels.
#clf.score(Test samples(X),True labels for X)
print("=================SVM Finsih======================")












#把0去掉后(20维)0.602814698984
#(57维)0.724003127443
#降维后0.179827990618


# Reference:
# sklearn.svm.SVC — scikit-learn 0.18.1 documentation
# => http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# python clf.fit 什么意思_百度知道
# => https://zhidao.baidu.com/question/691186420174711084.html

# The Iris Dataset — scikit-learn 0.18.1 documentation
# => http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py

# python iris 数据集 - power0405hf的专栏 - 博客频道 - CSDN.NET
# => http://blog.csdn.net/power0405hf/article/details/50767700

# pandas教程：[5]读取csv数据
# => http://jingyan.baidu.com/article/ab69b270d9b9542ca7189f2b.html

# PCA example with Iris Data-set
# => http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py

# Python将数组（矩阵）存成csv文件，将csv文件读取为数组（矩阵）
# => http://blog.csdn.net/vernice/article/details/50683637

# scikit-learn中PCA的使用方法 - 好库文摘
# => http://doc.okbase.net/u012162613/archive/120946.html

# Python中怎样使用shape计算矩阵的行和列-爱编程
# => http://www.w2bc.com/Article/11904

# python统计多维数组的行数和列数 - 如果一切重来 - 博客频道 - CSDN.NET
# => http://blog.csdn.net/xiaoxiangzi222/article/details/55225874
