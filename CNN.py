# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:30:09 2020

@author: Administrator
"""

import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam


#读取数据
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#数据预处理
X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)



#建立卷积神经网络模型
model = Sequential()
#第一个卷积层的输出是(32,28,28)
model.add(Conv2D(32,(5,5),activation='relu',
                 input_shape=(1,28,28),padding='same'))
#第一个池化层的输出是(32,14,14)
model.add(MaxPooling2D(pool_size=(2,2),
                       strides=(2,2),
                       padding='same'
                       ))
#第二个卷积层的输出是(64,14,14)
model.add(Conv2D(64,(5,5),activation='relu',
                 input_shape=(1,28,28),padding='same'))
#第二个池化层的输出是(64,7,7)
model.add(MaxPooling2D(pool_size=(2,2),
                       strides=(2,2),
                       padding='same'
                       ))
#将数据压缩为一层 64*7*7=3136
model.add(Flatten()) 

#第一个全连接层  
model.add(Dense(1024))
model.add(Activation('relu'))

#第二个全连接层
model.add(Dense(10))
model.add(Activation('softmax'))


#先建立优化器备用
adam = Adam(lr=1e-4)


#模型的损失函数及优化
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])



print('Training-----------')
model.fit(X_train,y_train,epochs=1,batch_size=32)

print('Testing-----------')
loss,accuracy = model.evaluate(X_test,y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)










