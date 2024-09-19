import math
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read data
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

# Split training set and validation set
x_train = x_data[:40]
y_train = y_data[:40]
x_test = x_data[40:len(x_data)]
y_test = y_data[40:len(y_data)]

# fit the model by Linear Regression
regression = linear_model.LinearRegression(fit_intercept=False)
regression.fit(x_train, y_train)

# Coefficient of the linear regression model
print(regression.coef_)

# Predicting data
y_predict = regression.predict(x_test)

# Computing MSE, MAE, R-Square
mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
r_quare = r2_score(y_test, y_predict)

# Print MSE, MAE, R_Square
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-Squared Error: {r_quare}')
