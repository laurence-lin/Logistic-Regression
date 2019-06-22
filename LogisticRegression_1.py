import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

'''
Logistic Regression Implementation
'''

file = pd.read_csv('dataset.txt')
data = np.array(file)
feature = data[:, 0:2]
label = data[:, 2]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def Normalize(x):
    '''
    Normalization: to 2D data feature
    '''
    num_feature = x.shape[1]
    
    for i in range(num_feature):
        Max = np.max(x[:, i])
        Min = np.min(x[:, i])
        x[:, i] = (x[:, i] - Min)/(Max - Min)
        
    return x

def Denormalize(x):
    '''
    Denormalize the data
    x: 2D data with 2 features
    '''
    for i in range(x.shape[1]):
        Max = np.max(feature[:, i])
        Min = np.min(feature[:, i])
        x[:, i] = x[:, i]*(Max - Min) + Min
        
    return x 

def predict(x, theta):
    '''
    hypothesis for classification
    x: [batch size, features]
    theta: [features, 1]
    return: [batch size]
    '''
    return sigmoid(np.matmul(x, theta))

def cost(y, h):
    '''
    Show loss function
    y: [batch size, 1]
    h: [batch size]
    '''
    bias = 0.0001
    m = y.shape[0]
    loss = - (1/m) * np.sum(y*np.log(h + bias) + (1 - y)*np.log(1 - h + bias))
    return loss

def gradient(h, y, x):
    '''
    Compute gradient
    return: [features, 1] gradient array
    '''
    m = x.shape[0]
    grad = (np.matmul(x.T, h - y)) / m
    return grad

c1 = []
c2 = []
for i in range(len(label)):
    if label[i] == 0:
        c1.append(feature[i, :])
    elif label[i] == 1:
        c2.append(feature[i, :])
        
c1 = np.array(c1)
c2 = np.array(c2)

x = np.c_[np.ones((feature.shape[0], 1)), feature]
x[:, 1:] = Normalize(x[:, 1:])
y = label[:,np.newaxis]
theta = np.random.random((x.shape[1], 1))
print(x)
print(y)

Iteration = 1000
learn_rate = 0.5
losses = []
print('Train data size', x.shape)

for iterate in range(Iteration):

    Cost = cost(y, predict(x, theta))
    if np.isnan(Cost):
       print(predict(x, theta))

    theta = theta - learn_rate*gradient(predict(x, theta), y, x)
    c = cost(y, predict(x, theta))
    print('Cost', c)
    losses.append(c)

xx, yy = np.meshgrid(np.arange(20, 100, 1), np.arange(30, 100, 1))
grid_point = np.c_[xx.ravel(), yy.ravel()].astype(float)
print(grid_point)
grid_point = Normalize(grid_point)
print(grid_point)
grid = np.c_[np.ones((grid_point.shape[0], 1)), grid_point]
print(grid.shape)
boundary = predict(grid, theta).reshape(xx.shape)
for i in range(boundary.shape[0]):
    for j in range(boundary.shape[1]):
        if boundary[i, j] < 0.5:
            boundary[i, j] = 0
        elif boundary[i, j] >= 0.5:
            boundary[i, j] = 1
print(boundary)

plt.figure(1)
plt.scatter(c1[:, 0], c1[:, 1], label ='Unadmitted')
plt.scatter(c2[:, 0], c2[:, 1], label = 'Admitted')
plt.legend(loc = 'upper right')
plt.contour(xx, yy, boundary, linewidths = 1)

plt.figure(2)
plt.title('Loss')
plt.plot(losses)


plt.show()






