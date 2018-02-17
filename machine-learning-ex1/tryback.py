import numpy as np
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

fd = open('./ex1/ex1data1.txt')
data = fd.read()
tuples = data.split('\n')

X = np.array([[1 for x in range(97)],[float(x.split(',')[0]) for x in tuples]], float)
Y = np.array([[float(y.split(',')[1]) for y in tuples]], float)

#parameters
theta = np.ones((2,1), np.float32)

J = []
iter = 1500
alpha = 0.01
m = 97
for iteration in range(iter) :
    temp = np.arange(len(theta), dtype=float)
    predictions = np.dot(theta.T,X)
    errors = predictions - Y
    J.append(np.sum(errors*errors))
    theta = theta - alpha/m*(np.dot(X, (np.dot(theta.T,X) - Y).T))

predicts = np.dot(theta.T, X)
print "Final theta : " , theta
plt.title("Error curve") 
    
# Plot the points using matplotlib 
plt.figure(1)
plt.plot(X[1], predicts.reshape(97,1), ".r")
plt.plot(X[1], Y[0], '.b')
ax = plt.gca()
ax.set_xticklabels([])

plt.show()