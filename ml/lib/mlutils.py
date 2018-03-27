import csv
import math
from sklearn import preprocessing as prep
import sys
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

def coastAndGradientDescentRegression(theta, inputParm, cost, alpha, no_iteration):
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

    # Now calculate gradient.
    # Using vectorized formulla to calculate gradient
    # Grad = 1/m * XTranspose * (h-y)
    #
    grad = np.dot(X.transpose(),(h-y))/m;
    return grad;
