import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import math
import sys
import numpy as np;
from matplotlib import cm
sys.path.append('/home/sanjay/home_work/study/ml/lib')
import mlutils as mu




data = mu.readCSVFile('ex1data1.txt');
dim = data.shape;
print(dim[0]);
print(dim[1]);

plt.scatter(data[:,0],data[:,1], marker='+');
plt.xlabel("Population of city in 10,000.");
plt.ylabel("Profit in $10,000.00")
#plt.show();

#Now create motrix for linear equation
#need to add one to X
y = data[:,1];
y = y.reshape(dim[0],1);
columVec = np.ones((dim[0],1),dtype=np.float_);
#columVec = columVec.reshape(dim[0],1);
X = data[:,0];
X_axis = X=X.reshape(dim[0],1);
X=np.append(columVec,X,axis=1);
#print(X);

theta = np.zeros((2,1), dtype=np.float_);
#theta = theta.reshape(2,1);

print('\n Test the cost function \n...');
J = mu.computeCostRegression(X,y,theta);
mu.printf("Computed cost at theta[0,0] = %.3f\n",J );
theta = np.array([[-1],[2]]);
J = mu.computeCostRegression(X,y,theta);

mu.printf("Computed cost for theta[-1,2] = %.3f\n",J );
iterations = 1500;
alpha = 0.01;
theta = np.zeros((2,1),dtype=np.float_);
#theta = theta.reshape(2,1);
mu.printf("\nRunning Gradient Descent ...\n");
theta = mu.gradientDescentRegression(X,y,theta,alpha, iterations);

# print theta to screen
mu.printf("Theta found by gradient descent:\n");
mu.printf("[%.4f, %.4f]\n", theta[0],theta[1]);
mu.printf("Expected theta values (approx)\n");
mu.printf(" -3.6303\n  1.1664\n\n");
myval = np.dot(X,theta);
print(myval.shape);
plt.scatter(X_axis,myval, marker='.', c='r');
# Predict values for population sizes of 35,000 and 70,000
predict1 = theta[0] + 3.5*theta[1];
mu.printf("For population = 35,000, we predict a profit of %f\n",predict1*10000);
predict2 = theta[0] + 7*theta[1];
mu.printf("For population = 70,000, we predict a profit of %f\n",predict2*10000);
theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);
# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)));

# Fill out J_vals
for i  in range (0,len(theta0_vals)):
	for j in range (0,len(theta1_vals)):
		t = np.array([[theta0_vals[i]], [theta1_vals[j]]]);
		J_vals[i,j] = mu.computeCostRegression(X, y, t);

J_vals = J_vals.transpose();
fig=plt.figure();
ax=fig.gca(projection='3d');
ax.plot_surface(theta0_vals, theta1_vals,J_vals,cmap=cm.Spectral,rstride=4, cstride=4,linewidth=1, antialiased=False);
plt.xlabel('\theta_0'); plt.ylabel('\theta_1'); plt.title('Cost function')

# draw countours
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.figure();
CS = plt.contour(theta0_vals,theta1_vals,J_vals, np.logspace(-2,3,num=20));

#contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
plt.xlabel('\theta_0'); plt.ylabel('\theta_1');
plt.scatter(theta[0],theta[1], marker='+', c='r');

plt.show();
