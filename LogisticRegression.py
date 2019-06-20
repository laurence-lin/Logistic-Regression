import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Logistic Regression Implementation
'''

data = pd.read_csv('balance.data')
data = np.array(data)
label = data[:, 0]
features = data[:, 1:]

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
    '''
    Use PCA to reduce multi-features to 2 features
    x: 2D features data
    k: reduce original feature to k dimension
    '''
    u, s, v = np.linalg.svd(x)
    project_m = v[0:k, :].T
    pca_x = np.matmul(x, project_m)
    
    return pca_x
    

x_data = Normalize(features)
x_data = PCA(x_data.astype(float)) # convert object type to float type, or np.cov will return error

print('Data size:', x_data.shape)

'''Convert string label to numeric values'''
n_label = {'R':0, 'L':1, 'B':2}
label_str = pd.Series(label)
y_data = label_str.map(n_label)
y_data = np.array(y_data)

c1 = []
c2 = []
c3 = []
for i in range(len(y_data)):
    if y_data[i] == 0:
        c1.append(x_data[i, :])
    elif y_data[i] == 1:
        c2.append(x_data[i, :])
    elif y_data[i] == 2:
        c3.append(x_data[i, :])
    
c1 = np.array(c1)
c2 = np.array(c2)
c3 = np.array(c3)
    

fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1) # show 2 plot in a figure
ax.scatter(c1[:, 0], c1[:, 1], c = 'red')
ax.scatter(c2[:, 0], c2[:, 1], c = 'blue')
ax.scatter(c3[:, 0], c3[:, 1], c = 'green')

plt.show()






