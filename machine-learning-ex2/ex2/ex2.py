import numpy as np
import math
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

def G(x) :
    ans = 1/(1+np.exp(-x))
    return ans

fd = open('ex2data1.txt')
data = fd.read()

dataLines = data.split('\n')

X = np.array([[1 for x in dataLines],[x.split(",")[0] for x in dataLines],[x.split(",")[1] for x in dataLines]], dtype='float32').T
Y = np.array([x.split(",")[2] for x in dataLines], dtype='float32').T
theta = np.ones((3,1), dtype='float32')

J = []
iter = 1000000
for i in range(iter):
    theta = theta - 0.0005/100*(np.dot(X.T, G(np.dot(X, theta))-Y.reshape(100,1)))

print theta

predict = np.array([1,45,85])
print G(np.dot(theta.T,predict))

pos  = []
neg = []
for x in range(len(Y)) :
    if Y[x] == 1 :
        pos.append(X[x][1 :])
    else :
        neg.append(X[x][1 : ])

#data
neg = np.array(neg)
pos = np.array(pos)

#line


plt.title("Visualization") 
plt.figure(1)
plt.scatter(pos.T[0], pos.T[1] , marker='^')
plt.scatter(neg.T[0], neg.T[1] , marker='X')

t = np.linspace(0, 100, 400)
c = (-theta[0]-theta[1]*t)/theta[2]

plt.plot(t, c, 'g') # plotting t, c separately 
plt.show()

plt.show()