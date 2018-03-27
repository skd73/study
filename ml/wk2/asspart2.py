import csv
import copy
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

#main body

data = mu.readCSVFile('ex2data2.txt');
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
#posInd = np.where(y==1)
#negInd = np.where(y==0)

#mu.printf("Xpos =%d, ypos=%d, xneg = %d, y =%d\n",len(Xaxis_pos),len(Yaxis_pos),len(Xaxis_neg),len(Yaxis_neg))
fig = plt.figure(0);
mu.plotData(X[:,1:3],y);
plt.xlabel('Microchip Test 1');
plt.ylabel('Microchip Test 2')
#plt.scatter(Xaxis[posInd],Yaxis[posInd],marker='^')
#plt.scatter(Xaxis[negInd],Yaxis[negInd],marker='o',c='r')
#fig = plt.figure(3);
#xr = np.arange(-10,10,0.1)
#plt.plot(xr,sigmoid(xr))
# =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  Slinearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#
X = mu.mapFeature(Xaxis, Yaxis);
dim = np.shape(X)
mu.printf("X dimensions after mapping = [%d,%d]\n",dim[0],dim[1]);
# Initialize fitting parameters
initial_theta = np.zeros((1,dim[1]))
tdim = np.shape(initial_theta);
print(tdim);
mu.printf("inital theta dim = [%d ]\n", tdim[0]);
# Set regularization parameter l to 1
l = 1;

# Compute and display initial cost and gradient for regularized logistic
# regression
#[cost, grad] = costFunctionReg(initial_theta, X, y, l);
cost = mu.computeCostLogisticRegressionLR(initial_theta, X, y, l);
grad = mu.GradientDescentLogisticRegressionLR(initial_theta, X, y, l);

print(cost);
mu.printf("Cost at initial theta (zeros): %.3f\n", cost);
mu.printf("Expected cost (approx): 0.693\n");
mu.printf("Gradient at initial theta (zeros) - first five values only:\n");
print(grad.shape)
mu.printf(" [%.4f %.4f %.4f %.4f %.4f] \n", grad[0],grad[1],grad[2],grad[3],grad[4]);
mu.printf("Expected gradients (approx) - first five values only:\n");
mu.printf(" [0.0085 0.0188 0.0001 0.0503 0.0115]\n");

mu.printf("\nProgram paused. Press enter to continue.\n");

# Compute and display cost and gradient
# with all-ones theta and l = 10
l=10;
test_theta = np.ones((1,dim[1]),dtype=np.float_);
#[cost, grad] = costFunctionReg(test_theta, X, y, 10);

cost = mu.computeCostLogisticRegressionLR(test_theta, X, y, l);
grad = mu.GradientDescentLogisticRegressionLR(test_theta, X, y, l);

mu.printf('\nCost at test theta (with l = 10): %.3f\n', cost);
mu.printf('Expected cost (approx): 3.16\n');
mu.printf('Gradient at test theta - first five values only:\n');
mu.printf(' [%.4f %.4f %.4f %.4f %.4f  \n', grad[0],grad[1],grad[2],grad[3],grad[4]);
mu.printf('Expected gradients (approx) - first five values only:\n');
mu.printf(' [ 0.3460 0.1614  0.1948 0.2269  0.0922\n');

mu.printf('\nProgram paused. Press enter to continue.\n');

# Set regularization parameter lambda to 1 (you should vary this)
l = 1;
# initial_theta = np.zeros((dim[1],1));

#% Optimize
res = opt.minimize(fun=mu.computeCostLogisticRegressionLR, x0=initial_theta, args = (X, y,l), method='TNC',jac=mu.GradientDescentLogisticRegressionLR)
theta= res.x;

plt.figure(1);
mu.plotDecisionBoundary(theta, X, y);
plt.xlabel('Microchip Test 1');
plt.ylabel('Microchip Test 2')
plt.title('lambda = 1')

p = mu.predict(theta, X);
tmp = np.double(p == y)
mu.printf("l=%d Train Accuracy: %.3f\n",l, tmp.mean() * 100);
mu.printf("\n");

## ============= Part 3: Regularization and Accuracies Optional =============
# Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#%  Try the following values of lambda (0, 1, 10, 100).
#%
#%  How does the decision boundary change when you vary lambda? How does
#%  the training set accuracy vary?
#%

#% Initialize fitting parameters
#initial_theta = np.zeros((dim[0],dim[1]);

#% Set regularization parameter lambda to 1 (you should vary this)
l = 0;

#% Set Options
#options = optimset('GradObj', 'on', 'MaxIter', 400);

#% Optimize
res = opt.minimize(fun=mu.computeCostLogisticRegressionLR, x0=initial_theta, args = (X, y,l), method='TNC', jac=mu.GradientDescentLogisticRegressionLR)
theta= res.x
#[theta, J, exit_flag] = ...
#	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
print(theta.shape)
print(X.shape)
print(y.shape)
#% Plot Boundary
plt.figure(2);
mu.plotDecisionBoundary(theta, X, y);
plt.xlabel('Microchip Test 1');
plt.ylabel('Microchip Test 2')
plt.title('lambda = 0')

#hold on;
#title(smu.printf('lambda = %g', l))

# Labels and Legend
#xlabel('Microchip Test 1')
#ylabel('Microchip Test 2')

#legend('y = 1', 'y = 0', 'Decision boundary')
#hold off;

#% Compute accuracy on our training set
p = mu.predict(theta, X);
tmp = np.double(p == y)
mu.printf("l=%d Train Accuracy: %.3f\n",l, tmp.mean() * 100);
mu.printf("\n");

l = 100;
res = opt.minimize(fun=mu.computeCostLogisticRegressionLR, x0=initial_theta, args = (X, y,l), method='TNC',jac=mu.GradientDescentLogisticRegressionLR)
theta= res.x
#[theta, J, exit_flag] = ...
#	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
#	% Plot Boundary
plt.figure(3);
mu.plotDecisionBoundary(theta, X, y);
plt.xlabel('Microchip Test 1');
plt.ylabel('Microchip Test 2')
plt.title('lambda = 100')
#	hold on;
#	title(smu.printf('lambda = %g', lambda))

#	% Labels and Legend
#	xlabel('Microchip Test 1')
#	ylabel('Microchip Test 2')

#	legend('y = 1', 'y = 0', 'Decision boundary')
#	hold off;

#	% Compute accuracy on our training set
p = mu.predict(theta, X);
tmp = np.double(p == y)
mu.printf("lambda =%d Train Accuracy: %.3f\n",l, tmp.mean() * 100);
mu.printf("\n");
plt.show()
