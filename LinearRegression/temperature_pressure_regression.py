from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Temp (F degree)
X = np.array([[194.5, 194.3, 197.9, 198.4, 199.4, 199.9,
               200.9, 201.1, 201.4, 201.3, 203.6, 204.6,
               209.5, 208.6, 210.7, 211.9, 212.2]]).T
# Press (Atm)
y = np.array([[20.79, 20.79, 22.4, 22.67, 23.15, 23.35,
               23.89, 23.99, 24.02, 24.01, 25.14, 26.57,
               28.49, 27.76, 29.04, 29.88, 30.06]]).T
# Visualize data
plt.plot(X, y, 'ro')
plt.axis([193, 213, 19, 31])
plt.xlabel('Temperature (F)')
plt.ylabel('Pressure (Atm)')
plt.show()

# Function to calculate mean of the dataset
def mean(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i]
    return sum / len(x)


def sxx(x):
    sum = 0
    x_mean = mean(x)
    for i in range(len(x)):
        sum += (x[i] - x_mean) * (x[i] - x_mean)
    return sum


def sxy(x, y):
    sum = 0
    x_mean = mean(x)
    y_mean = mean(y)
    for i in range(len(X)):
        sum += (x[i] - x_mean) * (y[i] - y_mean)
    return sum

# Coefficients of the linear regression model
theta_1 = sxy(X, y) / sxx(X)
theta_0 = mean(y) - theta_1 * mean(X)
print(f'Theta_0: {theta_0}')
print(f'Theta_1: {theta_1}')

x0 = np.linspace(193, 213, 2)
y0 = theta_0 + theta_1 * x0

# Drawing the fitting line
plt.plot(X.T, y.T, 'ro')
plt.plot(x0, y0) # data
# the fitting line
plt.axis([193, 213, 19, 31])
plt.xlabel('Temperature (F)')
plt.ylabel('Pressure (Atm)')
plt.show()
