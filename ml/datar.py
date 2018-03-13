import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import sys
import numpy as np;
from matplotlib import cm

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

with open('ex1data1.txt', newline='\n') as csvfile:
     datareader = csv.reader(csvfile, delimiter=',')
     mat = list(datareader);
data = np.array(mat[0:],dtype=np.float_);
dim = data.shape;
#print(dim[0]);
#print(dim[1]);

plt.scatter(data[:,0],data[:,1]);
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
X=X.reshape(dim[0],1);
X=np.append(columVec,X,axis=1);
#print(X);

theta = np.zeros((2,1), dtype=np.float_);
#theta = theta.reshape(2,1);

print('\n Test the cost function \n...');
J = computeCost(X,y,theta);
printf("Computed cost at theta[0,0] = %.3f\n",J );
theta = np.array([[-1],[2]]);
J = computeCost(X,y,theta);

printf("Computed cost for theta[-1,2] = %.3f\n",J );
iterations = 1500;
alpha = 0.01;
theta = np.zeros((2,1),dtype=np.float_);
#theta = theta.reshape(2,1);
printf("\nRunning Gradient Descent ...\n");
theta = gradientDescent(X,y,theta,alpha, iterations);

# print theta to screen
printf("Theta found by gradient descent:\n");
printf("[%.4f, %.4f]\n", theta[0],theta[1]);
printf("Expected theta values (approx)\n");
printf(" -3.6303\n  1.1664\n\n");

# Predict values for population sizes of 35,000 and 70,000
predict1 = theta[0] + 3.5*theta[1];
printf("For population = 35,000, we predict a profit of %f\n",predict1*10000);
predict2 = theta[0] + 7*theta[1];
printf("For population = 70,000, we predict a profit of %f\n",predict2*10000);
theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);
# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)));

# Fill out J_vals
for i  in range (0,len(theta0_vals)):
	for j in range (0,len(theta1_vals)):
		t = np.array([[theta0_vals[i]], [theta1_vals[j]]]);
		J_vals[i,j] = computeCost(X, y, t);
fig=plt.figure();
ax=fig.gca(projection='3d');
ax.plot_surface(theta0_vals, theta1_vals,J_vals,rstride=8, cstride=8, alpha=0.3);
plt.show();
