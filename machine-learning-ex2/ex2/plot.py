import numpy as np
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

x1 = np.linspace(-1.0, 1.0, 100)
x2 = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x1,x2)
F = X + X*Y + X**2 + Y**2 - 0.6
plt.contour(X,Y,F,[0])
plt.show()