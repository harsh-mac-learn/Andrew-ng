import random
import numpy as np

def sigmoid(x):
    ans = 1/(1+np.exp(-x))
    return ans

def sigmoid_prime(x):
    ans = sigmoid(x) * (1-sigmoid(x))
    return ans

class NeuralNetwork:
    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        self.w1 = np.random.random((self.inputLayerSize, self.hiddenLayerSize))
        self.w2 = np.random.random((self.hiddenLayerSize, self.outputLayerSize))

    def load(self, x, y):
        self.X = np.array(x)
        self.Y = np.array(y)

    def forward(self,x):
        self.z2 = np.dot(x, self.w1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.w2)

        yhat = sigmoid(self.z3)

        return yhat

    def train(self, numOfIterations, learningRate):
        for i in range(numOfIterations):
            yhat = self.forward(self.X)

            dout = np.multiply(-(self.Y - yhat), sigmoid_prime(self.z3))
            djdw2= np.dot(self.a2.T, dout)

            # print np.dot(dout, self.w2.T).shape, self.w2.shape
            dhidden = np.dot(dout, self.w2.T) * sigmoid_prime(self.z2)
            djdw1 = np.dot(self.X.T, dhidden)

            self.w2 -= djdw2 * learningRate
            self.w1 -= djdw1 * learningRate

            if i%5000 == 0 :
                # print np.sum(abs(self.Y - yhat))
                pass

    def test(self, x, y):
        TX = np.array(x)
        TY = np.array(y)

        predicted = np.array([[0] if x < 0.5 else [1] for x in self.forward(TX)], dtype='float32').tolist()
        original = TY.tolist()

        error = sum([0 if predicted[i] == original[i] else 1 for i in range(len(predicted))])

        print "Accuracy : ", float(len(original) - error) / len(original) * 100


# get data
fd = open('ex2data1.txt')
data = fd.read()
fd.close()

# convert to usable data
dataLines = data.split('\n')

random.shuffle(dataLines)

testData = dataLines[71 :]
dataLines = dataLines[0 : 70]

X = [[(float(x.split(',')[0])-30.05)/(99.8278 - 30.05), (float(x.split(',')[1]) - 30.05)/(99.8278 - 30.05)] for x in dataLines]
Y = [[float(y.split(',')[2])] for y in dataLines]

TX = [[(float(x.split(',')[0])-30.05)/(99.8278 - 30.05), (float(x.split(',')[1]) - 30.05)/(99.8278 - 30.05)] for x in testData]
TY = [[float(y.split(',')[2])] for y in testData]

print X

nn = NeuralNetwork()
nn.load(X,Y)
nn.train(100000,0.05)
nn.test(TX,TY)