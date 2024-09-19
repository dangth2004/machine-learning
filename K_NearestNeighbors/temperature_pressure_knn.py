from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Temp (F degree)
X = np.array([194.5, 194.3, 197.9, 198.4, 199.4, 199.9,
               200.9, 201.1, 201.4, 201.3, 203.6, 204.6,
               209.5, 208.6, 210.7, 211.9, 212.2]).T
# Press (Atm)
y = np.array([20.79, 20.79, 22.4, 22.67, 23.15, 23.35,
               23.89, 23.99, 24.02, 24.01, 25.14, 26.57,
               28.49, 27.76, 29.04, 29.88, 30.06]).T
# Visualize data
plt.plot(X, y, 'ro')
plt.axis([193, 213, 19, 31])
plt.xlabel('Temperature (F)')
plt.ylabel('Pressure (Atm)')
plt.show()

X_train = np.array(X[:12])
y_train = np.array(y[:12])
X_test = np.array(X[12:len(X)])
y_test = np.array(y[12:len(y)])

k = 2

def distance(array, value):
    x1 = np.array(array)
    return abs(x1 - value)


def find_nearest_index(array, value, k):
    array_D = distance(array, value)
    return np.argsort(array_D)[:k]


y_pred = np.zeros(len(X_test))
for i in range(len(X_test)):
    indexis = find_nearest_index(X_train, X_test[i], k)
    for id in indexis:
        y_pred[i] = y_pred[i] + y_train[id]
    y_pred[i] = y_pred[i] / len(indexis)
    print(y_pred[i], ' | ', y_test[i])

# Computing MSE, MAE, R_Square for model evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_quare = r2_score(y_test, y_pred)

# Print MSE, MAE, R_Square
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-Squared Error: {r_quare}')
