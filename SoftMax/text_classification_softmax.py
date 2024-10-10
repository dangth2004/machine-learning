import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups_vectorized

# Read data from the dataset
n_samples = 20000  # number of samples

x_data, y_data = fetch_20newsgroups_vectorized(subset='all', return_X_y=True)
# The data will include 20000 samples
x_data = x_data[:n_samples]
y_data = y_data[:n_samples]
# Split the training set and the validation set
# The training set include 75% number of samples of the dataset
# The rest is the validation set
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42,
                                                    stratify=y_data, test_size=0.25)
train_samples, n_features = x_train.shape
n_classes = np.unique(y_data).shape[0]

start = time.perf_counter()  # start time counter
# Multinomial Logistic Regression (softmax) approach
softmax = LogisticRegression(multi_class='multinomial')
# Train model
softmax.fit(x_train, y_train)
# Predict results
y_predict = softmax.predict(x_test)
end = time.perf_counter()  # end time counter

# Print accuracy score and confusion matrix
print(f'The accuracy score: {accuracy_score(y_test, y_predict)}')
print(f'The confusion matrix: {confusion_matrix(y_test, y_predict)}')
print(f'The execution time: {end - start}s')