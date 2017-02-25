import pandas as pd # pandas read csv
import numpy as np  #For save data to csv
from sklearn import decomposition #For PCA
from sklearn.svm import SVC #For SVM
from numpy import * #For 读取行数和列数Shape
import matplotlib.pyplot as plt #For draw picture

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


#用SVM监督学习来训练trainData
print("=================Start SVM======================")
clf = SVC() #创建分类器对象
clf.fit(trainDataSet, trainLabel) #用训练数据拟合分类器模型
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
predict_label = clf.predict(testDataSet)
np.savetxt('predict_label.csv', predict_label, delimiter = ',')  #把predict后的label存成csv
print(predict_label) #用训练好的分类器去预测testDataSet数据的标签
print(clf.score(testDataSet,testLabel)) #Returns the mean accuracy on the given test data and labels.
#clf.score(Test samples(X),True labels for X)
print("=================SVM Finsih======================")

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
#Drawing picture
plt.plot(testLabel,'gs', label='Expected')
plt.plot(predict_label,'y+', label='Predict')
legend = ax.legend(loc='upper left', shadow=True, fontsize='large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
plt.xlabel('Data numbers')
plt.ylabel('Fingers numbers')
plt.show()



# reference
# Pyplot tutorial — Matplotlib 2.0.0 documentation
# http://matplotlib.org/users/pyplot_tutorial.html?winzoom=1

# api example code: legend_demo.py — Matplotlib 2.0.0 documentation
# http://matplotlib.org/examples/api/legend_demo.html

# Matplotlib Examples — Matplotlib 2.0.0 documentation
# http://matplotlib.org/examples/index.html