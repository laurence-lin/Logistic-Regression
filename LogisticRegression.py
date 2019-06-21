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

# Normalization

x_train = Normalize(x_train)
x_test = Normalize(x_test)

x_train = PCA(x_train)
x_test = PCA(x_test)

'''hx_label: binary labels for binary logistic regression classifier'''
h1_label = np.zeros((1, len(y_train)))
h2_label = np.zeros((1, len(y_train)))
h3_label = np.zeros((1, len(y_train)))
class1 = []
class2 = []
class3 = []
for i in range(len(y_train)):
    if y_train[i] == 1:
        class1.append(x_train[i])
        h1_label[0,i] = 1
    elif y_train[i] == 2:
        class2.append(x_train[i])
        h2_label[0,i] = 1
    elif y_train[i] == 3:
        class3.append(x_train[i])
        h3_label[0,i] = 1

c1 = np.array(class1)
c2 = np.array(class2)
c3 = np.array(class3)

'''Multiclass logistic regression: Traing three binary classifier
h1, h2, h3. When do prediction, apply h1, h2, h3 to input x and determine the class by the max output.'''
# Add an constant x0 for theta0?
x_add = np.ones((x_train.shape[0], x_train.shape[1]+1))
x_add[:, 1:] = x_train[:, :]
x_train = x_add
theta = np.random.normal(loc = 0, scale = 0.1, size = [1, x_train.shape[1]])

print('Train size:', x_train.shape)

# Start training
train_epoch = 100
learn_rate = 0.01
h_1 = sigmoid(np.matmul(x_train, theta.T)) # output of LR result
h_1 = np.squeeze(h_1)
losses = - np.mean( h1_label*np.log(h_1) + (1 - h1_label)*np.log(1 - h_1) )

'''
Train class 1 first
'''
z = (h_1 - h1_label)*x_train[:, 0] 
print(x_train[:, 0].shape)
print((h_1 - h1_label).T.shape)
grad_1 = np.matmul(x_train[:, 0], (h_1 - h1_label).T)/len(y_train) # gradient of loss with respoect to variables
grad_2 = np.matmul(x_train[:, 1], (h_1 - h1_label).T)/len(y_train)
grad_3 = np.matmul(x_train[:, 2], (h_1 - h1_label).T)/len(y_train)
gradient = np.array([grad_1, grad_2, grad_3])

'''print(gradient.shape)
for iterate in range(train_epoch):
    theta = theta - learn_rate*gradient.T
    
    h_1 = sigmoid(np.matmul(x_train, theta.T))
    
    losses = - np.mean( h1_label*np.log(h_1) + (1 - h1_label)*np.log(1 - h_1) )
    print('mean losses:', losses)

plot_line = sigmoid(np.matmul(x_train, theta.T))
for i in range(len(plot_line)):
    if plot_line[i] >= 0.5:
        plot_line[i] = 1
    elif plot_line[i] < 0.5:
        plot_line[i] = 0

accuracy = (plot_line == h1_label).astype(int)
accuracy = np.mean(accuracy)
print(plot_line)
print('Train accuracy:', accuracy)'''
      
'''xx, yy = np.meshgrid(np.arange(-1.5, 0.5, 0.005), np.arange(-1.4, 0.2, 0.005))
grid_point = np.c_[xx.ravel(), yy.ravel()]
boundary = np.matmul(grid_point, theta.T).reshape(xx.shape)

fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1) # show 2 plot in a figure
ax.scatter(c1[:, 0], c1[:, 1], c = 'red')
ax.scatter(c2[:, 0], c2[:, 1], c = 'blue')
ax.scatter(c3[:, 0], c3[:, 1], c = 'green')
ax.contour(xx, yy, boundary)

plt.show()'''






