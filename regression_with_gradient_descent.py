import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

#loading and storing data
data=pd.read_csv("student-mat.csv",sep= ";")
data = data.select_dtypes(exclude=['object'])
predict='G3'
df=data[['G1','G2','G3','health','absences']]
x = np.array(df.drop([predict], 1))
y=np.array(df[predict])

#feature scaling
mu=x.mean(axis=0)
std=x.std(axis=0)
x=(x-mu)/std

#cost function
def cost_function(X, Y, theta):
 m = len(Y)
 J = np.sum((X.dot(theta)-Y) ** 2)/(2 * m)
 return J

# Gradient descent
def batch_gradient_descent(X, Y, theta, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    for i in range(iterations):
        h=X.dot(theta)
        diff= h-Y
        derivative=(X.T.dot(diff))/m
        theta=theta-(alpha*derivative)
        cost=cost_function(X,Y,theta)
        cost_history[i]=cost
    return theta, cost_history


#spliting and giving data
x_train, x_test, y_train, y_test=train_test_split(x,y,train_size = 0.8)
x_train = np.c_[np.ones(len(x_train),dtype=int),x_train]
x_test = np.c_[np.ones(len(x_test),dtype=int),x_test]
theta=np.zeros(x_train.shape[1])
iter_=2000
alpha=0.03
newB, cost_history = batch_gradient_descent(x_train, y_train, theta, alpha, iter_)

