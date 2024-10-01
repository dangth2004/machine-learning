import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Read data
# Change to data path on your computer
data = pd.read_csv('chd_framingham.csv')
# Remove row contain N/A in data set
data.head()
data.isnull().sum()
data = data.dropna(how="any", axis=0)
# Show the description of data
data.describe()

# Set to training data (x, y)
x_data = data[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
               'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
               'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
y_data = data['TenYearCHD']

x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
data_len = len(x_data)
train_size = (data_len * 7) // 10

# Split the training set and the validation set
x_train = x_data[:train_size]
y_train = y_data[:train_size]
x_test = x_data[train_size:data_len]
y_test = y_data[train_size:data_len]

# Train model
# If we use default max_iter of the sklearn library
# there will be an error:
# ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
# therefore we increase max_iter (=10000)
logistic_regression = LogisticRegression(penalty=None, max_iter=10000)
logistic_regression.fit(x_train, y_train)

# Predict result
y_predict = logistic_regression.predict(x_test)

# Calculate the Accuracy, Recall, and Precision
accuracy = accuracy_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)

# Print the result
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')