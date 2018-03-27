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
sys.path.append('/home/sanjay/home_work/study/ml/lib')
import mlutils as mu

data = mu.readCSVFile('ex1data2.txt')
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
alpha = 0.03
num_itr = 400
dmx=X.shape
theta = np.zeros((dmx[1],1))
mu.printf("Theta = ")
print(theta)
[theta,JA] = mu.coastAndGradientDescentRegression(X,y,theta,alpha,num_itr);
mu.printf("Theta computed from gradient descent: \n");
mu.printf(" [%.4f,%.4f, %.4f] \n", theta[0],theta[1],theta[2]);
mu.printf('\n');


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

price = 1*theta[0] + 1650*theta[1] + 3*theta[2] #You should change this

# ============================================================

mu.printf("Predicted price of a 1650 sq-ft, 3 br house using gradient descent):\n $%f \n", price)

mu.printf("Program paused. Press enter to continue.\n")

# Now calculate using normal matrix multification.abs

m=len(y); #no of training samples
X = np.ones((dim[0],1))
X = np.append(X,XOrg, axis=1)
#print(X);
step1 = np.dot(X.transpose(),X)
step2 = np.dot(X.transpose(),y)
step3 = la.inv(step1)
theta = np.dot(step3,step2)
myY = np.dot(X,theta)
ax.scatter(data[:,0],data[:,1],myY,marker='^',c='r');

price = 1*theta[0] + 1650*theta[1] + 3*theta[2];
mu.printf("Theta computed from gradient descent: \n");
mu.printf(" [%.4f,%.4f, %.4f] \n", theta[0],theta[1],theta[2]);
mu.printf('\n');

mu.printf("Predicted price of a 1650 sq-ft, 3 br house using Normal Eqn):\n $%f \n", price)

#plt.show()
fig2 = plt.figure(2)
Xaxis = np.arange(0,len(JA),1)
plt.plot(Xaxis,JA);
plt.show()
