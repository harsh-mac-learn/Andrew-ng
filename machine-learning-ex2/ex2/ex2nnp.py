import numpy as np
import random

def sigmoid(x):
    ans = 1/(1+np.exp(-x))
    return ans

def sigmoid_prime(x):
    ans = sigmoid(x) * (1-sigmoid(x))
    return ans

class model :
    def __init__(self):
        pass

    def G(self,x):
        ans = 1/(1+np.exp(-x))
        return ans

    def load(self, fileName, numberOfFeatures):
        fd = open(fileName)
        data = fd.read()
        dataLines = data.split('\n')
        random.shuffle(dataLines)

        trainingSize = int(len(dataLines)*0.7)
        self.trainingData = dataLines[0 : trainingSize]
        self.testingDataRaw = dataLines[trainingSize : ]
        
        self.X = np.array([1 for x in self.trainingData],dtype='float32')
        for feature in range(numberOfFeatures) :
            self.X = np.vstack((self.X , np.array([float(x.split(',')[feature]) for x in self.trainingData], dtype='float32')))
        self.X = self.X[1 : ]

        self.testingData = np.array([1 for x in self.testingDataRaw],dtype='float32')
        for feature in range(numberOfFeatures) :
            self.testingData = np.vstack((self.testingData , np.array([float(x.split(',')[feature]) for x in self.testingDataRaw], dtype='float32')))
        self.testingData = self.testingData[1 : ]

        self.totalX = np.hstack((self.X, self.testingData))

        self.Y = np.array([float(x.split(',')[numberOfFeatures]) for x in self.trainingData],dtype='float32')
        self.testingY = np.array([float(x.split(',')[numberOfFeatures]) for x in self.testingDataRaw],dtype='float32')

        # self.testingData = self.testingData.T
        # self.X = self.X.T
        # self.Y = self.Y.T
        # self.testingY = self.testingY.T
    
        self.w_hidden = np.random.uniform(size=(2, 6)).T
        self.w_output = np.random.uniform(size=(6, 1)).T

        # print self.w_hidden
        # print self.w_output

    def train(self, numberOfIterations, learningRate):
        iter = numberOfIterations
        for iteration in range(numberOfIterations) :
            # forward propogate
            a1 = self.X #2,82
            a2 = sigmoid(self.w_hidden.dot(a1))
            a3 = sigmoid(self.w_output.dot(a2))

            # find error
            errorOut = self.Y.reshape(1,self.Y.shape[0]) - a3 #1,82

            # back propogate
            errorHidden = errorOut.T.dot(self.w_output) #82, 6

            # find slope
            slopeOut = errorOut * sigmoid_prime(a3)
            slopeOut = slopeOut.T
            dWout = a2.dot(slopeOut)

            slopeHidden = errorHidden.T * sigmoid_prime(a2)
            slopeHidden = slopeHidden.T
            dWhidden = self.X.dot(slopeHidden)
            
            self.w_output += dWout.T*0.001
            self.w_hidden += dWhidden.T*0.001

        # print self.w_output
        # print self.w_hidden
            
    def test(self) :
        prediction = np.dot(self.theta.T, self.testingData.T)
        predictionList = []

        for i in prediction:
            for j in i :
                val = self.G(j)
                if val > 0.50 : 
                    predictionList.append(1)
                else :
                    predictionList.append(0)
        
        predictionListNp = np.array(predictionList)
        testingYNp = np.array(self.testingY.T,dtype='int32')
        
        errors = 0
        for i in range(len(testingYNp)):
            if predictionListNp[i] != testingYNp[i] :
                errors += 1
        
        errorPercantage = errors*100/len(testingYNp)
        return 100 - errorPercantage

m = model()
m.load('ex2data2.txt',2)
m.train(400000, 0.005)
# accuracy = m.test()

# print accuracy