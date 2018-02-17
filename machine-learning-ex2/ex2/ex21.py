import numpy as np
import math
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import random

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
            self.X = np.vstack((self.X , np.array([x.split(',')[feature] for x in self.trainingData], dtype='float32')))
        
        self.testingData = np.array([1 for x in self.testingDataRaw],dtype='float32')
        for feature in range(numberOfFeatures) :
            self.testingData = np.vstack((self.testingData , np.array([x.split(',')[feature] for x in self.testingDataRaw], dtype='float32')))

        self.totalX = np.hstack((self.X, self.testingData))

        self.testingData = np.vstack((self.testingData, self.testingData[1]*self.testingData[2]))
        self.testingData = np.vstack((self.testingData, self.testingData[1]*self.testingData[1]))
        self.testingData = np.vstack((self.testingData, self.testingData[2]*self.testingData[2]))
        
        self.Y = np.array([x.split(',')[numberOfFeatures] for x in self.trainingData],dtype='float32')
        self.testingY = np.array([x.split(',')[numberOfFeatures] for x in self.testingDataRaw],dtype='float32')


        self.testingData = self.testingData.T
        self.X = self.X.T
        self.Y = self.Y.T
        self.testingY = self.testingY.T
    
        self.theta = np.zeros((numberOfFeatures+1+3,1),dtype='float32')
        print self.theta
        self.Xmodified = np.vstack((self.X.T, self.X.T[1]*self.X.T[2]))
        self.Xmodified = np.vstack((self.Xmodified, self.X.T[1]*self.X.T[1]))
        self.Xmodified = np.vstack((self.Xmodified, self.X.T[2]*self.X.T[2]))
        self.Xmodified = self.Xmodified.T

    def train(self, numberOfIterations, learningRate):
        iter = numberOfIterations
        print "Before " , self.theta.shape
        for i in range(iter):
            self.theta = self.theta - learningRate/100*(np.dot(self.Xmodified.T, self.G(np.dot(self.Xmodified, self.theta))-self.Y.reshape(82,1)))
        print "after", self.theta.shape

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

    def visualizeData(self) :
        pos  = []
        neg = []
        for x in range(len(self.Y)) :
            if self.Y[x] == 1 :
                pos.append(self.X[x][1 :])
            else :
                neg.append(self.X[x][1 : ])

        #data
        neg = np.array(neg)
        pos = np.array(pos)
        plt.title("Visualization") 
        plt.figure(1)
        plt.scatter(pos.T[0], pos.T[1] , marker='^')
        plt.scatter(neg.T[0], neg.T[1] , marker='X')
        
        x = np.linspace(-1.0, 1.0, 100)
        y = np.linspace(-1.0, 1.0, 100)
        X, Y = np.meshgrid(x,y)

        F = self.theta[0] + self.theta[1]*X + self.theta[2]*Y + self.theta[3]*X*Y + self.theta[4]*X*X + self.theta[5]*Y*Y 
        plt.contour(X,Y,F,[0])
        
        plt.show()

m = model()
m.load('ex2data2.txt',2)
m.train(800000, 0.005)
accuracy = m.test()
m.visualizeData()

print accuracy