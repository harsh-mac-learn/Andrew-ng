import numpy as np

def sigmoid(x):
    ans = 1/(1+np.exp(-x))
    return ans

def sigmoid_prime(x):
    ans = sigmoid(x) * (1-sigmoid(x))
    return ans

epochs = 100000
input_size, hidden_size, output_size = 2, 3, 1
LR = .1 # learning rate

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([ [0],   [1],   [1],   [0]])

w_hidden = np.random.uniform(size=(input_size, hidden_size))
w_output = np.random.uniform(size=(hidden_size, output_size))

print w_hidden
print w_output

for epoch in range(epochs):
 
    # Forward
    act_hidden = sigmoid(np.dot(X, w_hidden)) #3,4
    
    output = np.dot(act_hidden, w_output) #3,1
    
    # Calculate error
    error = y - output #3,1
    
    if epoch % 5000 == 0:
        # print 'error sum', reduce(lambda x,y:x+y, error)
        pass

    # Backward
    dZ = error * LR
    w_output += act_hidden.T.dot(dZ)
    dH = dZ.dot(w_output.T) * sigmoid_prime(act_hidden)
    w_hidden += X.T.dot(dH)

X_test = np.array([0,1]) # [0, 1]

act_hidden = sigmoid(np.dot(X_test, w_hidden))
print np.dot(act_hidden, w_output)