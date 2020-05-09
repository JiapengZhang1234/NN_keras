# -*- coding: utf-8 -*-
"""
Created on Tue May  5 07:56:18 2020

@author: Administrator
"""

# 利用keras搭建回归神经网络

import numpy as np
np.random.seed(1337)

# Sequential代表的是序列模型
from keras.models import Sequential
# Dense代表的是全连接层
from keras.layers import Dense
import matplotlib.pyplot as plt

# 创造数据
X = np.linspace(-1,1,200)
np.random.shuffle(X)
Y = 0.5*X + 2 + np.random.normal(0,0.05,(200,))


# 展示创造的数据
# plt.scatter(X,Y)
# plt.show()

# 将数据分为训练集与测试集
X_train,Y_train = X[:160],Y[:160]
X_test,Y_test = X[160:],Y[160:]


# 建立神经网络
model = Sequential()
model.add(Dense(output_dim=1,input_dim=1))
#model.add(Dense(output_dim=1))

# 选择代价函数以及优化方法
model.compile(loss='mse', optimizer='sgd')

# training
print('Training----------')
for step in range(301):
    # train_on_batch函数有一个cost的默认返回值
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost:',cost)

# testing
print('\nTesting----------')
# 利用测试集来评估模型
cost = model.evaluate(X_test,Y_test,batch_size=40)
print('test cost:',cost)
# 得到模型中某一层的权重
W,b = model.layers[0].get_weights()
print('Weights=',W,'\nBias=',b)



# 训练过程的可视化
Y_pred = model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()






