import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             mean_squared_error, mean_absolute_error, r2_score)

# Read data
# Change to data path on your computer
data = pd.read_csv('Admission_Predict.csv')
# Show the description of data
data.describe()

# Set to training data (x, y)
x_data = np.asarray(data[['GRE Score', 'TOEFL Score', 'University Rating',
                          'SOP', 'LOR ', 'CGPA', 'Research']])
y_data = np.asarray(data['Chance of Admit'])
y_data = (y_data >= 0.75).astype(int)

# Split the training set and the validation test
data_len = len(x_data)
x_train = x_data[:350]
y_train = y_data[:350]
x_test = x_data[350:data_len]
y_test = y_data[350:data_len]

# Logistic Regression approach using Stochastic Gradient Descent
print('Logistic Regression approach using Stochastic Gradient Descent:')

# Add column bias to the matrix
x_train_log = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
x_test_log = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)


# Method to calculate Sigmoid function
def sigmoid(s):
    s = np.clip(s, -706, 708)
    return 1 / (1 + np.exp(-s))


# Method to calculate logistic sigmoid regression
def logistic_sigmoid_regression(x_train, y_train, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]
    N = x_train.shape[1]
    d = x_train.shape[0]
    count = 0
    check_w_after = 20
    # Vòng lặp của gradient descent ngẫu nhiên
    while count < max_count:
        # shuffle the order of data (for stochastic gradient descent).
        # and put into mix_id
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = x_train[:, i].reshape(d, 1)
            yi = y_train[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta * (yi - zi) * xi
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w

start_logistic = time.perf_counter()  # start count execution time

eta = 0.05
d = x_train_log.shape[1]
np.random.seed(2)
w_init = np.random.randn(d, 1)
w = logistic_sigmoid_regression(x_train_log.T, y_train, w_init, eta)
# Calculate and print the parameter coefficient
print(f'Parameter coefficient: w = {w[-1].flatten()}')


# Method to predict the result
def predict(x, w):
    z = sigmoid(np.dot(x, w))
    return (z >= 0.75).astype(int)


# Predict the result
y_pred = predict(x_test_log, w[-1])

# Create a DataFrame to compare actual result vs predicted result
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

# Print the compare result
print(result_df.head(len(x_test)))
accuracy_logistic = accuracy_score(y_test, y_pred)
recall_logistic = recall_score(y_test, y_pred)
precision_logistic = precision_score(y_test, y_pred)

end_logistic = time.perf_counter()  # end count execution time

# Calculate and print the accuracy score, recall score, precision score
print(f"Accuracy: {accuracy_logistic}")
print(f"Recall: {recall_logistic}")
print(f"Precision: {precision_logistic}")
print(f'Execution time: {end_logistic - start_logistic}s\n')

# Linear Regression approach
# Train model
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
# Predict result
y_pred_linear = linear_regression.predict(x_test)
# Calculate MSE, MAE, R-square and print the result
print('Linear Regression approach:')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred_linear)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred_linear)}')
print(f'R-square score: {r2_score(y_test, y_pred_linear)}\n')

# Naive Bayes Classifier approach
print('Naive Bayes Classifier approach:')

# Train model
start_naive = time.perf_counter()  # start count execution time

naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)
# Predict result
y_pred_naive_bayes = naive_bayes.predict(x_test)
# Calculate and print the accuracy score, recall score, precision score
accuracy_naive = accuracy_score(y_test, y_pred_naive_bayes)
recall_naive = recall_score(y_test, y_pred_naive_bayes)
precision_naive = precision_score(y_test, y_pred_naive_bayes)

end_naive = time.perf_counter()  # end count execution time

print(f'Accuracy score: {accuracy_naive}')
print(f'Recall score: {recall_naive}')
print(f'Precision score: {precision_naive}')
print(f'Execution time: {end_naive - start_naive}s\n')

# Logistic Regression approach using library
print('Logistic Regression approach using Scikit-Learn library:')

# Train model
# If we use default max_iter of the sklearn library
# there will be an error:
# ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
# therefore we increase max_iter (=1000)
start_log = time.perf_counter()  # start count execution time

logistic_regression = LogisticRegression(penalty=None, max_iter=1000)
logistic_regression.fit(x_train, y_train)
# Predict result
y_pred_log = logistic_regression.predict(x_test)
# Print the parameter coefficient theta of sigmoid function
print(f'Parameter Coefficient: w = {logistic_regression.coef_}')
# Calculate and print the accuracy score, recall score, precision score
accuracy_log = accuracy_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)

end_log = time.perf_counter()  # end count execution time

print(f'Accuracy score: {accuracy_log}')
print(f'Recall score: {recall_log}')
print(f'Precision score: {precision_log}')
print(f'Execution time: {end_log - start_log}s')