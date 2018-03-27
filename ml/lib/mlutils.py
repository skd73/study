import csv
import copy
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing as prep
import sys
import math
import numpy as np;
from numpy import linalg as la

def printf(format,*args):
     sys.stdout.write(format % args)

# Read CSV file and return float data array.abs
def readCSVFile(filename):
    with open(filename, newline='\n') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        mat = list(datareader);
    data = np.array(mat[0:],dtype=np.float_);
    return data;


def computeCostRegression(theta,inputParm,cost):
    #no of training sampleas
    m=len(cost);
    computedCost = 0;
    sizeInput=inputParm.shape;
    #printf("m= %d, sizes [%d,%d]\n", m, sizeInput[0],sizeInput[1]);
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    # Paramter validation: Diimension of X, y and theta should be compatible.
    #
    # J(theta) = 1/2m sum i=1 to m (X*Theta - y)' * (X * Theta -y)
    temp = np.dot(inputParm,theta) - cost;
    sizetmp = temp.shape;
#    printf("sizetmp = [%d,%d]\n",sizetmp[0],sizetmp[1]);
    computedCost = np.dot(temp.transpose(),temp)/(2*m);
#    printf("in func %f \n", computedCost);
    return computedCost;

def costAndGradientDescentRegression(theta, inputParm, cost, alpha, no_iteration):
    #this function calculates gradientDescent.
    #size of training set
    m=len(cost);
    J_history = np.zeros(no_iteration);
    dim = inputParm.shape;
    #print(dim);
    convergenceDirection =0;
    for j in range (0,no_iteration):
        for i in range(0,dim[1]):
            tmpTheta = np.zeros((dim[1],1),dtype=np.float_);
            predictedCost = np.dot(inputParm,theta);
            diffToActualCoast = predictedCost - cost;
            ithClmn = inputParm[:,i];
            #print(ithClmn.shape);
            ithClmn = ithClmn.reshape(dim[0],1);
            tmpTheta[i] = np.dot(diffToActualCoast.transpose(), ithClmn)/m;
            tmpTheta[i] = theta[i] - (alpha*tmpTheta[i]);
            theta[i] = tmpTheta[i];
        J_history[j] = computeCostRegression(inputParm,cost,theta);
        if j>1:
            if J_history[j]>J_history[j-1]:
                if convergenceDirection <= 0:
                    printf("Convergence Direction Change to pos\n")
                    convergenceDirection = 1
            else:
                if convergenceDirection >=0 :
                    printf("Convergence Direction Change to neg\n")
                    convergenceDirection = -11
        #printf("Theta[%f,%f] cost [%f]\n",theta[0],theta[1],J_history[j]);
    #printf("itr=%d, j=%f\n",j,J_history[j]);
    return (theta,J_history);
def gradientDescentRegression(theta, inputParm, cost, alpha, no_iteration):
    #this function calculates gradientDescent.
    #size of training set
    m=len(cost);
    dim = inputParm.shape;
    #print(dim);
    convergenceDirection =0;
    for j in range (0,no_iteration):
        for i in range(0,dim[1]):
            tmpTheta = np.zeros((dim[1],1),dtype=np.float_);
            predictedCost = np.dot(inputParm,theta);
            diffToActualCoast = predictedCost - cost;
            ithClmn = inputParm[:,i];
            #print(ithClmn.shape);
            ithClmn = ithClmn.reshape(dim[0],1);
            tmpTheta[i] = np.dot(diffToActualCoast.transpose(), ithClmn)/m;
            tmpTheta[i] = theta[i] - (alpha*tmpTheta[i]);
            theta[i] = tmpTheta[i];
    return theta;
# Logistic regression specific cost and gradient function.

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

def costFunctionLogisticRegression (theta, X, y):
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

def coastAndGradientDescentLogisticRegression (theta, X, y):
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

def GradientDescentLogisticRegression (theta, X, y):
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

    # Now calculate gradient.
    # Using vectorized formulla to calculate gradient
    # Grad = 1/m * XTranspose * (h-y)
    #
    grad = np.dot(X.transpose(),(h-y))/m;
    return grad;

def GradientDescentLogisticRegressionLR(theta, X, y, l):
    X=np.matrix(X);
    y=np.matrix(y);
    theta = np.matrix(theta);
    theta = theta.getT();
    dim=np.shape(theta);
    #theta.reshape(dim[0],1);
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

    tmpTheta = np.matrix(copy.copy(theta));
    tmpTheta[0] = 0;
    #print(theta)
    z = np.dot(X,theta);
    #h = g(z)
    h = sigmoid(z);
    m = len(y);
    #J = ( -y' * log(h) - (1 - y)' * log ( 1 - h))/m + (l/(2*m)) * thetaSqSum;
    # Now calculate gradient.
    # Using vectorized formulla to calculate gradient
    # Grad = 1/m * XTranspose * (h-y)
    grad = (np.dot(X.getT(),(h-y)) + np.multiply(l, tmpTheta))/m;
    return grad;


def computeCostLogisticRegressionLR(theta,X, y,l):

    # COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, l) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.
    # Initialize some useful values
    m = len(y); # number of training examples

    # You need to return the following variables correctly
    J = 0;
    theta = np.matrix(theta);
    theta = theta.getT();
    dim=np.shape(theta);
    X=np.matrix(X);
    y=np.matrix(y);
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
    z = np.dot(X,theta);
    #h = g(z)
    h = sigmoid(z);
    m = len(y);
    thetaSqSum = np.sum(np.power(tmpTheta,2), axis=0)
    #print(thetaSqSum)
    term1 = (-y.getT()).dot(np.log(h));
    term2 = ((1-y).getT()).dot(np.log(1-h))
    J = (term1 - term2)/np.double(m) + np.double(l/(2*m))*thetaSqSum;
    return J

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
    if not dim :
        out    = np.ones((1,1));
        X1 = np.matrix(X1);
        X2 = np.matrix(X2);
    else:
        out = np.ones((dim[0],1));
        out.reshape(dim[0],1);
    dimout = np.shape(out);
    #print(dimout)
    #print(dim)
    #printf("Sise of out [%d,%d], size of input X [%d %d]\n",dimout[0],dimout[1],dim[0],dim[1]);
    for i in range (1,degree+1):
        for j in range(0,i+1):
            X1temp = np.power(X1,(i-j));
            X2temp = np.power(X2,j);
            Xtemp = np.multiply(X1temp,X2temp);
            #out=np.append(out,Xtemp,axis=1)
            out = np.hstack((out,Xtemp));
            #print("X");
            #out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    return out;

# SImple scatter data
def plotData(X,y):
    Xaxis = X[:,0:1]
    Yaxis = X[:,1:2]
    posIndex = np.where(y==1);
    negIndex = np.where(y==0);
    plt.scatter(Xaxis[posIndex],Yaxis[posIndex], marker='+', label=' y = 1');
    plt.scatter(Xaxis[negIndex],Yaxis[negIndex], marker='o', label=' y = 0');
    plt.legend();
    legend = plt.legend(loc='upper right corner', shadow=True, fontsize='small')
    legend.get_frame().set_facecolor('#FFCC00')

def plotDecisionBoundary(theta,X,y):
    '''
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    %   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    %   positive examples and o for the negative examples. X is assumed to be
    %   a either
    %   1) Mx3 matrix, where the first column is an all-ones column for the
    %      intercept.
    %   2) MxN, N>3 matrix, where the first column is all-ones
    # Plot Data
    '''
    #plotData
    dim = X.shape;
    plotData(X[:,1:3],y);
    if dim[1] <=3:
        #Only need 2 points to define a line, so choose two endpoints
        plot_x =  [np.min(X[:,1])-2, np.max(X[:,1])+2];
        tmpy = np.multiply(theta(1),plot_x) + theta[0];
        plot_y = np.multiply(-(1/theta[2]),tmpy);
        plt.plot(plot_x,plot_y);
    else:
        u=np.linspace(-1,1.5,50);
        v=np.linspace(-1,1.5,50);
        z = np.zeros((len(u),len(v)));
        for i in range(0,len(u)):
            for j in range(0,len(v)):
                z[i,j] = np.dot(mapFeature(u[i],v[j]),theta);
        z = z.transpose();
        plt.contour(u,v,z,0)
