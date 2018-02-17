import random
import numpy as np

def G(x):
    ans = 1/(1+np.exp(-x))
    return ans

class brain :
    def __init__(self) :
        pass

    def load(self, inputs, outputs) :
        self.X = np.array(inputs, dtype='float32')
        self.Y = np.array(outputs, dtype='float32')

    def train(self) :
        # w12 = np.full((2,3),0.5, dtype='float32')
        # w23 = np.full((3,1),0.6, dtype='float32')

        w12 = np.random.uniform(size=(2, 3))
        w23 = np.random.uniform(size=(3, 1))
        
        for i in range(100000) :
            a2 = G(np.dot(self.X, w12))
            a3 = G(np.dot(a2, w23))

            error = self.Y - a3
            
            slope_hidden = G(a2) * (1- G(a2))
            slope_out = G(a3) * (1 - G(a3))
            
            d_out = error * slope_out
            
            error_hidden = np.dot(d_out, w23.T)
            
            d_hidden = error_hidden * slope_hidden
          
            w23 = w23 + np.dot(a2.T, d_out)*0.005
            w12 = w12 + np.dot(self.X.T, d_hidden)*0.005

        test_input = np.array([[1,1]])
        a2 = G(np.dot(test_input, w12))
        a3 = np.dot(a2, w23)
        print G(a3)

b = brain()
inputs = [[1,1],[1,0],[0,1],[0,0]]
outputs = [[0],[1],[1],[0]] 
b.load(inputs, outputs)
b.train()