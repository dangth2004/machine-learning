import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# Calculate SSE (Sum Squared Error)
mse = mean_squared_error(y_test, y_predict_regression)
sse = mse * valid_len

print(f'The sum of squared error (SSE): {sse}')

# K-NN method
k = 18

def distance(array, value):
    array = np.array(array)
    return abs(array - value)

def find_nearest_index(array, value, k):
    array_D = distance(array, value)
    return np.argsort(array_D)[:k]

y_predict_knn = np.zeros(len(x_test))
for i in range(len(x_test)):
    indexis = find_nearest_index(x_train, x_test[i], k)
    for id in indexis:
        y_predict_knn[i] = y_predict_knn[i] + y_train[id]
    y_predict_knn[i] = y_predict_knn[i] / len(indexis)
    print(y_predict_knn[i], ' | ', y_test[i])