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

def printf(format,*args):
     sys.stdout.write(format % args)

def computeCost(inputParm,cost, theta):
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

def gradientDescent(inputParm, cost, theta, alpha, no_iteration):
    #this function calculates gradientDescent.
    #size of training set
    m=len(cost);
    J_history = np.zeros(no_iteration);
    dim = data.shape;
    print(dim);
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
        J_history[j] = computeCost(inputParm,cost,theta);
        #printf("Theta[%f,%f] cost [%f]\n",theta[0],theta[1],J_history[j]);
    printf("itr=%d, j=%f\n",j,J_history[j]);
    return theta;

with open('ex1data2.txt', newline='\n') as csvfile:
     datareader = csv.reader(csvfile, delimiter=',')
     mat = list(datareader);
data = np.array(mat[0:],dtype=np.float_);
dim = data.shape;
print(dim[0]);print(dim[1]);
#print(data)
XOrg = data[:,0:2]
y = data[:,2]
y = y.reshape(dim[0],1)
X_norm= prep.normalize(XOrg, norm='l2')
X = np.ones((dim[0],1))
X = np.append(X,X_norm, axis=1)
#print(X)

fig = plt.figure();
ax=fig.gca(projection='3d');
ax.scatter(data[:,0],data[:,1],data[:,2],marker='+');
alpha = 1
num_itr = 400
dmx=X.shape
theta = np.zeros((dmx[1],1))
printf("Theta = ")
print(theta)
theta = gradientDescent(X,y,theta,alpha,num_itr);
printf("Theta computed from gradient descent: \n");
printf(" [%.4f,%.4f, %.4f] \n", theta[0],theta[1],theta[2]);
printf('\n');
#plt.show()

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

price = 1*theta[0] + 1650*theta[1] + 3*theta[2] #You should change this

# ============================================================

printf("Predicted price of a 1650 sq-ft, 3 br house using gradient descent):\n $%f \n", price)

printf("Program paused. Press enter to continue.\n")

# Now calculate using normal matrix multification.abs

m=len(y); #no of training samples
X = np.ones((dim[0],1))
X = np.append(X,XOrg, axis=1)
#print(X);
step1 = np.dot(X.transpose(),X)
step2 = np.dot(X.transpose(),y)
step3 = la.inv(step1)
theta = np.dot(step3,step2)

price = 1*theta[0] + 1650*theta[1] + 3*theta[2];
printf("Theta computed from gradient descent: \n");
printf(" [%.4f,%.4f, %.4f] \n", theta[0],theta[1],theta[2]);
printf('\n');

printf("Predicted price of a 1650 sq-ft, 3 br house using Normal Eqn):\n $%f \n", price)

printf("Program paused. Press enter to continue.\n")
