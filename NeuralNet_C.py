# -*- coding: utf-8 -*-
"""
Created on Tue May  5 08:46:36 2020

@author: Administrator
"""


import numpy as np
np.random.seed(1337)

from keras.datasets import mnist #keras自带的数据集，手写体
from keras.utils import np_utils   #用来做编码处理
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

#读取数据
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#数据预处理
X_train = X_train.reshape(X_train.shape[0],-1)/255
X_test = X_test.reshape(X_test.shape[0],-1)/255

y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)


#搭建神经网络
model = Sequential([
    Dense(32,input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
    ])

#设置优化器
rms = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
# SGD  
# Momentum  AdaGrad  RMSprop 
# 这些方法都在SGD的基础上进行了某种改进，加速了训练过程

#设置代价函数与优化方式
model.compile(
    loss='categorical_crossentropy',   #？
    optimizer=rms,
    metrics=['accuracy'])    #？



#training
#epochs整个数据集的训练多少遍

print('Training---------')
model.fit(X_train,y_train,epochs=2,batch_size=32)

#testing

print('Testing------------')
loss,accuracy = model.evaluate(X_test,y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)


















