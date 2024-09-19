import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

with open('fuel.txt') as f:
    lines = f.readlines()
x_data = []
y_data = []
lines.pop(0)
for line in lines:
    splitted = line.replace('\n', '').split(',')
    splitted.pop(0)
    splitted = list(map(float, splitted))
    fuel = 1000 * splitted[1] / splitted[5]
    dlic = 1000 * splitted[0] / splitted[5]
    logMiles = math.log2(splitted[3])
    y_data.append([fuel])
    x_data.append([splitted[-1], dlic, splitted[2], logMiles])

data_len = len(x_data)
x_data = np.asarray(x_data)
y_data = np.asarray(y_data).reshape(data_len)

print(x_data.shape)

x_train = x_data[:40]
y_train = y_data[:40]
x_test = x_data[40:len(x_data)]
y_test = y_data[40:len(y_data)]

k = 6


def distance(array, value):
    array = np.array(array)
    return np.linalg.norm(array - value, ord=2, axis=1)


def find_nearest_index(array, value, k):
    array_D = distance(array, value)
    return np.argsort(array_D)[:k]


y_pred = np.zeros(len(x_test))
for i in range(len(x_test)):
    indexis = find_nearest_index(x_train, x_test[i], k)
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
