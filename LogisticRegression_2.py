import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

'''
Logistic Regression Implementation
'''

file = pd.read_csv('wine.data')
data = file.values
y_data = data[:, 0]
x_data = data[:, 1:]

shuffle = np.random.permutation(x_data.shape[0])
x_data = x_data[shuffle]
y_data = y_data[shuffle]

total = x_data.shape[0]
train_end = int(total * 0.8)
x_train = x_data[0:train_end, :]
y_train = y_data[0:train_end]
x_test = x_data[train_end:, :]
y_test = y_data[train_end:]    

print('Train data:', x_train.shape, 'Test data:', x_test.shape)

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

def PCA(x, k = 2):
    cov_x = np.cov(x.T)
    u, s, v = np.linalg.svd(cov_x)
    project_m = u[:, 0:k]
    pca_x = np.matmul(x, project_m)
    
    return pca_x


def gradient(h, y, x):
    '''
    Compute gradient
    return: [features, 1] gradient array
    '''
    m = x.shape[0]
    grad = (np.matmul(x.T, h - y)) / m
    return grad

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



# Normalization

x_train = Normalize(x_train)
x_test = Normalize(x_test)


x_train = PCA(x_train)
x_test = PCA(x_test)

#x_train = Normalize(x_train)

'''hx_label: binary labels for binary logistic regression classifier'''
h1_label = np.zeros((len(y_train), 1))
h2_label = np.zeros((len(y_train), 1))
h3_label = np.zeros((len(y_train), 1))
class1 = []
class2 = []
class3 = []
for i in range(len(y_train)):
    if y_train[i] == 1:
        class1.append(x_train[i])
        h1_label[i] = 1
    elif y_train[i] == 2:
        class2.append(x_train[i])
        h2_label[i] = 1
    elif y_train[i] == 3:
        class3.append(x_train[i])
        h3_label[i] = 1

c1 = np.array(class1)
c2 = np.array(class2)
c3 = np.array(class3)

'''Multiclass logistic regression: Traing three binary classifier
h1, h2, h3. When do prediction, apply h1, h2, h3 to input x and determine the class by the max output.'''
# Add an constant x0 for theta0?
x_add = np.ones((x_train.shape[0], x_train.shape[1]+1))
x_add[:, 1:] = x_train[:, :]
x_train = x_add
theta = np.random.normal(loc = 0, scale = 0.1, size = [x_train.shape[1], 1])

print('Train size:', x_train.shape)

# Start training
train_epoch = 1000
learn_rate = 0.5
print(x_train)
print(h1_label)

'''
Train class 1 first
'''
for iterate in range(train_epoch):
    
    Cost = cost(h1_label, predict(x_train, theta))
    theta = theta - learn_rate*gradient(predict(x_train, theta), h1_label, x_train)
    c = cost(h1_label, predict(x_train, theta))
    
    print('Cost', c)

'''Clf1 is the classifier weight for class 1'''
clf1 = theta 

'''Train class 2'''
for iterate in range(train_epoch):
    
    Cost = cost(h2_label, predict(x_train, theta))
    theta = theta - learn_rate*gradient(predict(x_train, theta), h2_label, x_train)

clf2 = theta
    
'''Train class 3'''    
for iterate in range(train_epoch):
    
    Cost = cost(h3_label, predict(x_train, theta))
    theta = theta - learn_rate*gradient(predict(x_train, theta), h3_label, x_train)

clf3 = theta
      
xx, yy = np.meshgrid(np.arange(-1.5, 0.5, 0.005), np.arange(-1.4, 0.2, 0.005))
grid_point = np.c_[xx.ravel(), yy.ravel()]
grid = np.c_[np.ones((grid_point.shape[0], 1)), grid_point]
boundary1 = np.matmul(grid, clf1).reshape(xx.shape)
boundary2 = np.matmul(grid, clf2).reshape(xx.shape)
boundary3 = np.matmul(grid, clf3).reshape(xx.shape)

for i in range(boundary1.shape[0]):
    for j in range(boundary1.shape[1]):
        if boundary1[i, j] < 0.5:
            boundary1[i, j] = 0
        elif boundary1[i, j] >= 0.5:
            boundary1[i, j] = 1

for i in range(boundary2.shape[0]):
    for j in range(boundary2.shape[1]):
        if boundary2[i, j] < 0.5:
            boundary2[i, j] = 0
        elif boundary2[i, j] >= 0.5:
            boundary2[i, j] = 1
            
for i in range(boundary3.shape[0]):
    for j in range(boundary3.shape[1]):
        if boundary3[i, j] < 0.5:
            boundary3[i, j] = 0
        elif boundary3[i, j] >= 0.5:
            boundary3[i, j] = 1

fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1) # show 2 plot in a figure
ax.scatter(c1[:, 0], c1[:, 1], c = 'red')
ax.scatter(c2[:, 0], c2[:, 1], c = 'blue')
ax.scatter(c3[:, 0], c3[:, 1], c = 'green')
ax.contour(xx, yy, boundary1, colors = 'black')
ax.contour(xx, yy, boundary2, colors = 'black')
ax.contour(xx, yy, boundary3, colors = 'black')

plt.show()