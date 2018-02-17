import numpy as np
import math
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

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

        trainingSize = int(len(dataLines)*0.7)
        self.trainingData = dataLines[0 : trainingSize]
        self.testingDataRaw = dataLines[trainingSize : ]
        
        self.X = np.array([1 for x in self.trainingData],dtype='float32')
        for feature in range(numberOfFeatures) :
            self.X = np.vstack((self.X , np.array([x.split(',')[feature] for x in self.trainingData], dtype='float32')))
        
        self.testingData = np.array([1 for x in self.testingDataRaw],dtype='float32')
        for feature in range(numberOfFeatures) :
            self.testingData = np.vstack((self.testingData , np.array([x.split(',')[feature] for x in self.testingDataRaw], dtype='float32')))
        
        
        self.Y = np.array([x.split(',')[numberOfFeatures] for x in self.trainingData],dtype='float32')
        self.testingY = np.array([x.split(',')[numberOfFeatures] for x in self.testingDataRaw],dtype='float32')

        self.testingData = self.testingData.T
        self.X = self.X.T
        self.Y = self.Y.T
        self.testingY = self.testingY.T

        self.theta = np.zeros((numberOfFeatures+1,1),dtype='float32')

    def train(self, numberOfIterations, learningRate):
        iter = numberOfIterations
        for i in range(iter):
            self.theta = self.theta - learningRate/100*(np.dot(self.X.T, self.G(np.dot(self.X, self.theta))-self.Y.reshape(70,1)))
        print "after : " , self.theta.shape
    
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
m.load('ex2data1.txt',2)
m.train(600000, 0.01)
accuracy = m.test()

print accuracy