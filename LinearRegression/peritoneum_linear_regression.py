import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read data
with open('vidu4_lin_reg.txt') as f:
    lines = f.readlines()

x_data = []
y_data = []
lines.pop(0)

for line in lines:
    splitted = line.replace('\n', '').split(' ')
    splitted.pop(0)
    splitted = list(map(float, splitted))
    y_data.append([splitted[5]])
    x_data.append([splitted[0], splitted[4], splitted[3], splitted[2], splitted[1]])

x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
data_len = len(x_data)

# Calculate the coefficient of the linear regression predict function on the dataset
regression_data_set = LinearRegression()
regression_data_set.fit(x_data, y_data)
print('With the data set:')
print(f'The coefficient is: {regression_data_set.coef_}')
print(f'The intercept is: {regression_data_set.intercept_}\n')

# Split the training set and the validation set
x_train = x_data[:80]
y_train = y_data[:80]
x_test = x_data[80:data_len]
y_test = y_data[80:data_len]

# Train model
regression = LinearRegression()
regression.fit(x_train, y_train)

# Print the coefficient of the linear regression predict function on the training set
print('With the training set:')
print(f'The coefficient is: {regression.coef_}')
print(f'The intercept is: {regression.intercept_}')
y_predict = regression.predict(x_test)

# Calculate error term
errors = y_test - y_predict

# Calculate mean of error term
mean_error = np.mean(errors)

# Calculate variance of error term
variance_error = np.var(errors, ddof=1)

print(f'Kỳ vọng của sai số: {mean_error}')
print(f'Phương sai của sai số: {variance_error}')