import sys
import numpy as np;

X1=np.array([[1],[2],[3]])
X2=np.array([[2],[3],[4]])

degree = 6;
dim = np.shape(X1);
out = np.ones((dim[0],1));
for i in range (1,degree+1):
    for j in range(0,i+1):
        X1temp = np.power(X1,(i-j));
        X2temp = np.power(X2,j);
        Xtemp = np.multiply(X1temp,X2temp);
        out=np.append(out,Xtemp,axis=1)
        print("X");
        #out(:, end+1) = (X1.^(i-j)).*(X2.^j);
return out;
