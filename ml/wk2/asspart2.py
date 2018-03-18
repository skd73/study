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

def printf(format,*args):
     sys.stdout.write(format % args)

#============================================
#Function to calculate sigmoid of matrix x
def sigmoid (z):
    return 1/(1 + np.exp(-z))
# Main body

#function out = mapFeature(X1, X2)
#% MAPFEATURE Feature mapping function to polynomial features
#%
#%   MAPFEATURE(X1, X2) maps the two input features
#%   to quadratic features used in the regularization exercise.
#%
#%   Returns a new feature array with more features, comprising of
#%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#%
#%   Inputs X1, X2 must be the same size
#%
def mapFeature(X1,X2):
    degree = 6;
    dim = np.shape(X1);
    out = np.ones((dim[0],1));
    out.reshape(dim[0],1);
    dimout = np.shape(out);
    dim = np.shape(X1);
    print(dimout)
    print(dim)
    printf("Sise of out [%d,%d], size of input X [%d %d]\n",dimout[0],dimout[1],dim[0],dim[1]);
    for i in range (1,degree+1):
        for j in range(0,i+1):
            X1temp = np.power(X1,(i-j));
            X2temp = np.power(X2,j);
            Xtemp = np.multiply(X1temp,X2temp);
            out=np.append(out,Xtemp,axis=1)
            #print("X");
            #out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    return out;

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

def computeGrad(theta,X, y,l):

    # COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, l) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.
    # Initialize some useful values
    m = len(y); # number of training examples

    # You need to return the following variables correctly
    J = 0;
    dim=np.shape(theta);
    theta.reshape(dim[0],1);
    grad = copy.copy(theta);

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    # function used to calculate cost is
    # J(Theta) = 1/m ( -ytranspose * log ( h) - (1 - y)transpose * log (1 - h) ) + ...
    # l /2m *sum(theta.^2)
    # h = g(X*theta) = sigmoid(X*theta)
    # we do not want to penalise theta0 so setting tmpTheta(1) =0;

    tmpTheta = copy.copy(theta);
    tmpTheta[0] = 0;
    #print(theta)
    z = X.dot(theta);
    #h = g(z)
    h = sigmoid(z);
    m = len(y);
    #J = ( -y' * log(h) - (1 - y)' * log ( 1 - h))/m + (l/(2*m)) * thetaSqSum;
    # Now calculate gradient.
    # Using vectorized formulla to calculate gradient
    # Grad = 1/m * XTranspose * (h-y)
    grad = ((X.T).dot(h-y) + np.multiply(l, tmpTheta))/m;

    ## grad = (X'* (h - y) + l * tmpTheta)/m; '

    # =============================================================
    return grad;

def computeCost(theta,X, y,l):

    # COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, l) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.
    # Initialize some useful values
    m = len(y); # number of training examples

    # You need to return the following variables correctly
    J = 0;
    dim=np.shape(theta);
    #print(dim)
    #grad = np.zeros((dim[0],dim[1]));

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    # function used to calculate cost is
    # J(Theta) = 1/m ( -ytranspose * log ( h) - (1 - y)transpose * log (1 - h) ) + ...
    # l /2m *sum(theta.^2)
    # h = g(X*theta) = sigmoid(X*theta)
    # we do not want to penalise theta0 so setting tmpTheta(1) =0;

    tmpTheta = copy.copy(theta);
    tmpTheta[0] = 0;
    #print(theta)
    z = X.dot(theta);
    #h = g(z)
    h = sigmoid(z);
    m = len(y);
    thetaSqSum = np.sum(np.power(tmpTheta,2), axis=0)
    #print(thetaSqSum)
    term1 = (-y.T).dot(np.log(h));
    term2 = ((1-y).T).dot(np.log(1-h))
    J = (term1 - term2)/np.double(m) + np.double(l/(2*m))*thetaSqSum;
    return J

#######################

def costFunctionReg(theta,X, y,l):

    # COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, l) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.
    # Initialize some useful values
    m = len(y); # number of training examples

    # You need to return the following variables correctly
    J = 0;
    dim=np.shape(theta);
    print(dim)
    grad = np.zeros((dim[0],dim[1]));

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    # function used to calculate cost is
    # J(Theta) = 1/m ( -ytranspose * log ( h) - (1 - y)transpose * log (1 - h) ) + ...
    # l /2m *sum(theta.^2)
    # h = g(X*theta) = sigmoid(X*theta)
    # we do not want to penalise theta0 so setting tmpTheta(1) =0;

    tmpTheta = copy.copy(theta);
    tmpTheta[0] = 0;
    #print(theta)
    z = X.dot(theta);
    #h = g(z)
    h = sigmoid(z);
    m = len(y);
    thetaSqSum = np.sum(np.power(tmpTheta,2), axis=0)
    print(thetaSqSum)
    term1 = (-y.T).dot(np.log(h));
    term2 = ((1-y).T).dot(np.log(1-h))
    J = (term1 - term2)/np.double(m) + np.double(l/(2*m))*thetaSqSum;
    #J = ( -y' * log(h) - (1 - y)' * log ( 1 - h))/m + (l/(2*m)) * thetaSqSum;
    # Now calculate gradient.
    # Using vectorized formulla to calculate gradient
    # Grad = 1/m * XTranspose * (h-y)
    grad = ((X.T).dot(h-y) + np.multiply(l, tmpTheta))/m;

    ## grad = (X'* (h - y) + l * tmpTheta)/m; '

    # =============================================================
    return (J, grad)

#########################

with open('ex2data2.txt', newline='\n') as csvfile:
     datareader = csv.reader(csvfile, delimiter=',')
     mat = list(datareader);
data = np.array(mat[0:],dtype=np.double);
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
# =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  Slinearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#
X = mapFeature(Xaxis, Yaxis);
dim = np.shape(X)
printf("X dimensions after mapping = [%d,%d]\n",dim[0],dim[1]);
# Initialize fitting parameters
initial_theta = np.zeros((dim[1],1))
tdim = np.shape(initial_theta);
print(tdim);
printf("inital theta dim = [%d ]\n", tdim[0]);
# Set regularization parameter l to 1
l = 1;

# Compute and display initial cost and gradient for regularized logistic
# regression
#[cost, grad] = costFunctionReg(initial_theta, X, y, l);
cost = computeCost(initial_theta, X, y, l);
grad = computeGrad(initial_theta, X, y, l);
printf('Cost at initial theta (zeros): %.3f\n', cost);
printf('Expected cost (approx): 0.693\n');
printf('Gradient at initial theta (zeros) - first five values only:\n');
printf(' [%.4f %.4f %.4f %.4f %.4f] \n', grad[0],grad[1],grad[2],grad[3],grad[4]);
printf('Expected gradients (approx) - first five values only:\n');
printf(' [0.0085 0.0188 0.0001 0.0503 0.0115]\n');

printf('\nProgram paused. Press enter to continue.\n');

# Compute and display cost and gradient
# with all-ones theta and l = 10
test_theta = np.ones((dim[1],1),dtype=np.float_);
#[cost, grad] = costFunctionReg(test_theta, X, y, 10);
cost = computeCost(test_theta, X, y, l);
grad = computeGrad(test_theta, X, y, l);

printf('\nCost at test theta (with l = 10): %.3f\n', cost);
printf('Expected cost (approx): 3.16\n');
printf('Gradient at test theta - first five values only:\n');
printf(' [%.4f %.4f %.4f %.4f %.4f  \n', grad[0],grad[1],grad[2],grad[3],grad[4]);
printf('Expected gradients (approx) - first five values only:\n');
printf(' [ 0.3460 0.1614  0.1948 0.2269  0.0922\n');

printf('\nProgram paused. Press enter to continue.\n');

# Set regularization parameter lambda to 1 (you should vary this)
l = 1;
# initial_theta = np.zeros((dim[1],1));

#% Optimize
res = opt.minimize(fun=computeCost, x0=initial_theta, args = (X, y,l), method='TNC',jac=computeGrad)
theta= res.x;

p = predict(theta, X);
tmp = np.double(p == y)
printf("l=%d Train Accuracy: %.3f\n",l, tmp.mean() * 100);
printf("\n");

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
res = opt.minimize(fun=computeCost, x0=initial_theta, args = (X, y,l), method='Nelder-Mead')
theta= res.x
#[theta, J, exit_flag] = ...
#	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

#% Plot Boundary
#plotDecisionBoundary(theta, X, y);
#hold on;
#title(sprintf('lambda = %g', l))

# Labels and Legend
#xlabel('Microchip Test 1')
#ylabel('Microchip Test 2')

#legend('y = 1', 'y = 0', 'Decision boundary')
#hold off;

#% Compute accuracy on our training set
p = predict(theta, X);
tmp = np.double(p == y)
printf("l=%d Train Accuracy: %.3f\n",l, tmp.mean() * 100);
printf("\n");

l = 100;
res = opt.minimize(fun=computeCost, x0=initial_theta, args = (X, y,l), method='Nelder-Mead')
theta= res.x
#[theta, J, exit_flag] = ...
#	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
#	% Plot Boundary

#	plotDecisionBoundary(theta, X, y);
#	hold on;
#	title(sprintf('lambda = %g', lambda))

#	% Labels and Legend
#	xlabel('Microchip Test 1')
#	ylabel('Microchip Test 2')

#	legend('y = 1', 'y = 0', 'Decision boundary')
#	hold off;

#	% Compute accuracy on our training set
p = predict(theta, X);
tmp = np.double(p == y)
printf("lambda =%d Train Accuracy: %.3f\n",l, tmp.mean() * 100);
printf("\n");
