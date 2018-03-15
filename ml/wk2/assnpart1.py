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

def printf(format,*args):
     sys.stdout.write(format % args)

#============================================
#Function to calculate sigmoid of matrix x
def sigmoid (z):
    return 1/(1 + np.exp(-z))

#Predict
##############################################
def predict(theta, X):
    #PREDICT Predict whether the label is 0 or 1 using learned logistic
    #regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a
    #   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    m = len(X[:,0:1]); # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros((m,1))

    #% ====================== YOUR CODE HERE ======================
    #% Instructions: Complete the following code to make predictions using
    #%               your learned logistic regression parameters.
    #%               You should set p to a vector of 0's and 1's
    #%
    #% probability that y=1 for given x at theta = hthetha(x) = g(z)
    z = np.dot(X,theta);
    h = sigmoid(z);
    posIndex = np.where(h >= 0.5);
    p[posIndex] = 1; # htheta(x)>= 0.5 ==> y = 1;
    return p
# =========================================================================



#============================================
#Function to calculate sigmoid of matrix x
def sigmoid (z):
    return 1/(1 + np.exp(-z))

def costFunction (theta, X, y):
    J = 0;
    grad = np.zeros((len(theta),1))
    # Instructions: Compute the cost of a particular choice of theta.
    # You should set J to the cost.
    # Compute the partial derivatives and set grad to the partial
    # derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    #This function calculate cost for given theta vector.
    # J(Theta) = 1/m ( -ytranspose * log ( h) - (1 - y)transpose * log (1 - h) );
    # h = g(z) = sigmoid(X*theta) where z=X*theta;
    z = np.dot(X,theta);
    h = sigmoid(z)
    m = len(y)
    # Calculate ytranspose * log(h)
    tmp1 = np.dot(y.transpose(),np.log(h))
    tmp2 = np.dot((1-y).transpose(),np.log(1-h));
    J = (-tmp1 - tmp2)/m;
    # Now calculate gradient.
    # Using vectorized formulla to calculate gradient
    # Grad = 1/m * XTranspose * (h-y)
    #
    grad = np.dot(X.transpose(),(h-y))/m;
    return (J,grad)



def computeCost (theta, X,y):
    J = 0;
    grad = np.zeros((len(theta),1))
    # Instructions: Compute the cost of a particular choice of theta.
    # You should set J to the cost.
    # Compute the partial derivatives and set grad to the partial
    # derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    #This function calculate cost for given theta vector.
    # J(Theta) = 1/m ( -ytranspose * log ( h) - (1 - y)transpose * log (1 - h) );
    # h = g(z) = sigmoid(X*theta) where z=X*theta;
    z = np.dot(X,theta);
    h = sigmoid(z)
    m = len(y)
    # Calculate ytranspose * log(h)
    tmp1 = np.dot(y.transpose(),np.log(h))
    tmp2 = np.dot((1-y).transpose(),np.log(1-h));
    J = (-tmp1 - tmp2)/m;
    return J

def computeGrad (theta, X, y):
    J = 0;
    grad = np.zeros((len(theta),1))
    # Instructions: Compute the cost of a particular choice of theta.
    # You should set J to the cost.
    # Compute the partial derivatives and set grad to the partial
    # derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    #This function calculate cost for given theta vector.
    # J(Theta) = 1/m ( -ytranspose * log ( h) - (1 - y)transpose * log (1 - h) );
    # h = g(z) = sigmoid(X*theta) where z=X*theta;
    z = np.dot(X,theta);
    h = sigmoid(z)
    m = len(y)
    # Calculate ytranspose * log(h)
    # Now calculate gradient.
    # Using vectorized formulla to calculate gradient
    # Grad = 1/m * XTranspose * (h-y)
    #
    grad = np.dot(X.transpose(),(h-y))/m;
    return grad


# Main body

with open('ex2data1.txt', newline='\n') as csvfile:
     datareader = csv.reader(csvfile, delimiter=',')
     mat = list(datareader);
data = np.array(mat[0:],dtype=np.float_);
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

#printf("Xpos =%d, ypos=%d, xneg = %d, y =%d\n",len(Xaxis_pos),len(Yaxis_pos),len(Xaxis_neg),len(Yaxis_neg))
fig = plt.figure();

plt.scatter(Xaxis[posInd],Yaxis[posInd],marker='^')
plt.scatter(Xaxis[negInd],Yaxis[negInd],marker='o',c='r')
#fig = plt.figure(3);
#xr = np.arange(-10,10,0.1)
#plt.plot(xr,sigmoid(xr))
#plt.show()

initial_theta = np.zeros((dim[1],1))
[cost,grad] = costFunction(initial_theta,X,y)

printf('Cost at initial theta (zeros): %.3f\n', cost);
printf('Expected cost (approx): 0.693\n');
printf('Gradient at initial theta (zeros): \n');
printf(' %.4f \n %.4f \n %.4f\n', grad[0],grad[1],grad[2],);
printf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');


initial_theta = np.array([[-24],[0.2],[0.2]])
[cost,grad] = costFunction(initial_theta,X,y)

printf(' Coast at Theta %.4f \n %.4f \n %.4f\n', grad[0],grad[1],grad[2],);
printf('theta: %.3f\n', cost);
printf('Expected cost (approx): 0.218\n');
printf('Calcuated gradients           %.3f  %.3f %.3f\n', grad[0],grad[1],grad[2],);
printf('Expected gradients (approx): 0.043  2.566  2.647\n');

# using min
#initial_theta = np.array([[-20],[.1],[.4]])
xopt = opt.fmin_cg(computeCost,x0=initial_theta,args=(X,y), maxiter=400)
cost = computeCost(xopt,X,y)
printf("fmin computed Theta=")
print(xopt)
printf("cost = %f\n",cost);
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
prob = sigmoid(z);
printf("For a student with scores 45 and 85, we predict an admission probability of %.3f\n", prob);
printf("Expected value: 0.775 +/- 0.002\n\n");

# Compute accuracy on our training set
p = predict(theta, X);
tmp = np.double(p == y)
printf("Train Accuracy: %.3f\n", tmp.mean() * 100);
printf("Expected accuracy (approx): 89.0\n");
printf("\n");

plt.show();
