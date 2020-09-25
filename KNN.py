# -*- encoding: utf-8 -*-

'''
@File    : KNN.py
@Time    : 2020/9/16 17:09
@Author  : s_bei
@Version : 1.0
@Contact : 13191688291@163.com
@Licence : (C)CopyRight 2020-2022
@Desc    : None

'''
# import lib
import numpy as np
import pickle as pl

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pl.load(fo,encoding='bytes')
        #pickle.load用法：pickle.load(file,encoding='ASCII'),其中encoding代表了解码方式
        #pickle提供了数据序列方法，其中dump为序列化，load为反序列化
    return dict

class Knearestneihhbor:
    def __init__(self):
        pass
    def train(self,X,y):
        """X is N x D where each row is an example. Y is 1-demension of size  N"""
        self.Xtr = X
        #对图片进行矩阵化
        self.ytr = y
        #对标签进行矩阵化

    def predict(self,X,k):
        """X is N x D where each row is an example we wish to predict label for"""
        num_test = X.shape[0]
        Ypred = np.zeros(num_test,dtype=self.ytr.dtype)
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]),axis=1)
            #欧氏距离的计算
            Y_pred = np.zeros(k,dtype=self.ytr.dtype)
            for j in range(k):
                Y_pred[j] = self.ytr[np.argsort(distances)[j]]
            print(Y_pred)
            #输出k个距离最近类标签
            Ypred[i] = np.argmax(np.bincount(Y_pred))
            print(Ypred[i])
            #获取k个距离最近类中出现次数最多的标签作为预测标签


        return Ypred

top_num = 500

#导入训练集以及测试集
path = 'D:/test-data/cifar-10-batches-py'
train_data = unpickle(path + '/data_batch_5')
test_data = unpickle(path + '/test_batch')

#进行KNN算法对CIFAR-10数据集做测试
knn = Knearestneihhbor()
knn.train(train_data[b'data'][:top_num],np.array(train_data[b'labels'][:top_num]))
Yte_predict = knn.predict(test_data[b'data'][:top_num],5)
print('accuracy:%f' % (np.mean(Yte_predict == test_data[b'labels'][:top_num])))
print(Yte_predict)
