# Assignment 2,
# Implementation for logistic regression.
# implements gradientDescent
#   costFunction
#   costFunctionReg
#   sigmoid function
#
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing as prep
import sys
import numpy as np;
from matplotlib import cm
from numpy import linalg as la
from scipy import optimize as opt
sys.path.append('/home/sanjay/home_work/study/ml/lib')
import mlutils as mu
#============================================


#============================================
#Function to calculate sigmoid of matrix x

# Main body

data = mu.readCSVFile('ex2data1.txt')
dim = data.shape;
print(dim[0]);print(dim[1]);
# put the data in input and output arrays
# X keeps input, y is knonw output
XOrg = data[:,0:2]
y = data[:,2]
y = y.reshape(dim[0],1)
X = np.ones((dim[0],1))
X = np.append(X,XOrg, axis=1)
#print(X)
# Data plotting
Xaxis = data[:,0:1]
Yaxis = data[:,1:2]
posInd = np.where(y==1)
negInd = np.where(y==0)

#mu.printf("Xpos =%d, ypos=%d, xneg = %d, y =%d\n",len(Xaxis_pos),len(Yaxis_pos),len(Xaxis_neg),len(Yaxis_neg))
fig = plt.figure();

plt.scatter(Xaxis[posInd],Yaxis[posInd],marker='^')
plt.scatter(Xaxis[negInd],Yaxis[negInd],marker='o',c='r')

initial_theta = np.zeros((dim[1],1))
cost = mu.costFunctionLogisticRegression(initial_theta,X,y)
grad = mu.GradientDescentLogisticRegression(initial_theta,X,y);
mu.printf('Cost at initial theta (zeros): %.3f\n', cost);
mu.printf('Expected cost (approx): 0.693\n');
mu.printf('Gradient at initial theta (zeros): \n');
mu.printf(' %.4f \n %.4f \n %.4f\n', grad[0],grad[1],grad[2],);
mu.printf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');


initial_theta = np.array([[-24],[0.2],[0.2]])
cost = mu.costFunctionLogisticRegression(initial_theta,X,y)
grad = mu.GradientDescentLogisticRegression(initial_theta,X,y);

mu.printf(' Coast at Theta %.4f \n %.4f \n %.4f\n', grad[0],grad[1],grad[2],);
mu.printf('theta: %.3f\n', cost);
mu.printf('Expected cost (approx): 0.218\n');
mu.printf('Calcuated gradients           %.3f  %.3f %.3f\n', grad[0],grad[1],grad[2],);
mu.printf('Expected gradients (approx): 0.043  2.566  2.647\n');

# using min
#initial_theta = np.array([[-20],[.1],[.4]])
xopt = opt.fmin_cg(mu.costFunctionLogisticRegression,x0=initial_theta,args=(X,y), maxiter=400)
cost = mu.costFunctionLogisticRegression(xopt,X,y)
mu.printf("fmin computed Theta=")
print(xopt)
mu.printf("cost = %f\n",cost);
#plot decision line
#
minx = X[:,2].min(axis=0) - 2
maxx = X[:,2].max(axis=0) + 2
plot_x = np.array([minx,maxx])
plot_y = (-1/xopt[2])*(xopt[1]*plot_x +xopt[0])
plt.plot(plot_x, plot_y,c='k');

# ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.m
#
#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2
x = np.array([1,45,85])
theta = xopt.reshape(3,1)
z = np.dot(x,theta)
prob = mu.sigmoid(z);
mu.printf("For a student with scores 45 and 85, we predict an admission probability of %.3f\n", prob);
mu.printf("Expected value: 0.775 +/- 0.002\n\n");

# Compute accuracy on our training set
p = mu.predict(theta, X);
tmp = np.double(p == y)
mu.printf("Train Accuracy: %.3f\n", tmp.mean() * 100);
mu.printf("Expected accuracy (approx): 89.0\n");
mu.printf("\n");

plt.show();
