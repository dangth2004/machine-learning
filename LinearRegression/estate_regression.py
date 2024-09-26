import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read data
# Change to data path on your computer
data = pd.read_csv('real_estate.csv')
# Show the description of data
data.describe()
# Set to training data (x, y)
y = data['Y house price of unit area']
X = data[['X1 transaction date', 'X2 house age',
          'X3 distance to the nearest MRT station',
          'X4 number of convenience stores',
          'X5 latitude', 'X6 longitude']]

# Get the integer part of a value (truncating)
X.iloc[:, 0] = X.iloc[:, 0].apply(lambda x: x // 1)
X.iloc[:, 1] = X.iloc[:, 1].apply(lambda x: x // 1)

y_data = np.asarray(y)
x_data = np.asarray(X)
data_len = len(x_data)

# Split the training set and the validation set
x_train = x_data[:350]
y_train = y_data[:350]
x_test = x_data[350:data_len]
y_test = y_data[350:data_len]
valid_len = len(y_test)

# Linear Regression method
# Train model
regression = LinearRegression()
regression.fit(x_train, y_train)

y_predict_regression = regression.predict(x_test)

# K-NN method
k = 18


def distance(array, value):
    return np.linalg.norm(array - value, ord=2, axis=1)


def find_nearest_index(array, value, k):
    array_D = distance(array, value)
    return np.argsort(array_D)[:k]


y_predict_knn = np.zeros(len(x_test))

for i in range(len(x_test)):
    indexis = find_nearest_index(x_train, x_test[i], k)
    for id in indexis:
        y_predict_knn[i] = y_predict_knn[i] + y_train[id]
    y_predict_knn[i] = y_predict_knn[i] / len(indexis)

print('Linear regression method:')
# Calculate SSE (Sum Squared Error)
mse_regression = mean_squared_error(y_test, y_predict_regression)
sse_regression = mse_regression * valid_len
# Calculate MSE, MAE, R-squared to compare linear regression method vs K-NN method
mae_regression = mean_absolute_error(y_test, y_predict_regression)
r2_score_regression = r2_score(y_test, y_predict_regression)

print(f'The sum of squared error (SSE): {sse_regression}')
print(f'Mean Squared Error (MSE): {mse_regression}')
print(f'Mean Absolute Error (MAE): {mae_regression}')
print(f'R-Squared: {r2_score_regression}\n')

print('K-NN method:')
mse_knn = mean_squared_error(y_test, y_predict_knn)
sse_knn = mse_knn * valid_len
mae_knn = mean_absolute_error(y_test, y_predict_knn)
r2_score_knn = r2_score(y_test, y_predict_knn)
print(f'The sum of squared error (SSE): {sse_knn}')
print(f'Mean Squared Error (MSE): {mse_knn}')
print(f'Mean Absolute Error (MAE): {mae_knn}')
print(f'R-Squared: {r2_score_knn}\n')