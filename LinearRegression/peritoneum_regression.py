import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read data
# Change to data path on your computer
with open('peritoneum.txt') as f:
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
print(f'The intercept is: {regression.intercept_}\n')
y_predict = regression.predict(x_test)

# Calculate error term
errors = y_test - y_predict

# Calculate mean of error term
mean_error = np.mean(errors)

# Calculate variance of error term
variance_error = np.var(errors, ddof=1)

print(f'Mean of error term: {mean_error}')
print(f'Variance of error term: {variance_error}\n')

# Calculate MAE, MSE, R-Squared
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-Squared Error: {r2}')