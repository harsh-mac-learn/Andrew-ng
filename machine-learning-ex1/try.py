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
theta = np.ones(2, np.float32)

J = []
iter = 1500
alpha = 0.01
m = 97
for iteration in range(iter) :
    temp = np.arange(len(theta), dtype=float)
    predictions = np.dot(theta.T,X)
    errors = predictions - Y
    
    J.append(np.sum(errors*errors))

    for j in range(len(temp)) :
        temp[j] = theta[j] - np.sum(alpha/m*errors*X[j]) 
    
    theta = temp

print theta
predicts = np.dot(theta.T, X)
print predicts

print "Final theta : " , theta
plt.title("Error curve") 
    
# Plot the points using matplotlib 
plt.figure(1)
plt.subplot(211)
plt.plot(X[1], predicts, "xr")

# now switch back to figure 1 and make some changes
plt.figure(1)
plt.subplot(211)
plt.plot(X[1], Y[0], 'xb')
ax = plt.gca()
ax.set_xticklabels([])

plt.show()