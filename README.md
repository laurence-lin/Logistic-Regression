# Logistic-Regression
Model and dataset for logistic regression practice

Try to implement logistic regression on multiclass dataset

Applied PCA on balance dataset first, but the plot show bad result

UCI Balance data set 

![img](https://github.com/laurence-lin/Logistic-Regression/blob/master/balance_PCA.png)

UCI wine data set

![img](https://github.com/laurence-lin/Logistic-Regression/blob/master/wine_PCA.png)

Score admitted dataset

![img]

Logistic Regression Result for wine data set after PCA:

![img](https://github.com/laurence-lin/Logistic-Regression/blob/master/wine_logistic_result.png)



Discussion:

During training, encounter a problem that loss keep diverging, and I still haven't found out why. After implement the function for prediction, loss, gradient, the problem is solved. 

Becareful when doing gradient descent, and make sure the use function to compute these values.
