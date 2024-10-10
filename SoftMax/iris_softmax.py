import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy import sparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import the Iris dataset
iris = datasets.load_iris()
x_data = iris.data[:, :4]
y_data = iris.target
C = 3  # Number of classes

# Normalize the entire dataset for PCA visualization
scaler = MinMaxScaler()
x_norm = scaler.fit_transform(x_data)  # Normalized data

# Perform PCA for visualization purposes
pca = PCA(n_components=2)  # 2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(x_norm))

# Visualize the PCA transformation
plt.axis("off")
plt.scatter(transformed[y_data == 0][0], transformed[y_data == 0][1],
            s=9, label='IRIS Setosa', c='red')
plt.scatter(transformed[y_data == 1][0], transformed[y_data == 1][1],
            s=9, label='IRIS Versicolor', c='green', marker="^")
plt.scatter(transformed[y_data == 2][0], transformed[y_data == 2][1],
            s=9, label='IRIS Virginica', c='blue', marker="s")
plt.legend()
plt.show()


# Helper function to convert labels to a binary matrix form
def convert_labels(y, C=C):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))),
                          shape=(C, len(y))).toarray()
    return Y


# Stable softmax implementation
def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A


# Softmax Regression Training Function
def softmax_regression(X, y, W_init, eta, tol=1e-4, max_count=10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # Shuffle the data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax_stable(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta * xi.dot((yi - ai).T)
            count += 1
            # Check for convergence
            if count % check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W


# Predict function to find the class with the maximum probability
def pred(W, X):
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis=0)


# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2, random_state=42)

# Normalize train and test data separately using MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert to the required format for softmax regression (transpose to get features in rows)
x_train = x_train.T
x_test = x_test.T

# Initialize the weights
eta = 0.05
d = x_train.shape[0]
np.random.seed(42)
W_init = np.random.randn(d, C)

# Train the softmax regression model
W = softmax_regression(x_train, y_train, W_init, eta)

# Predict the labels on the test set
y_predict = pred(W[-1], x_test)
print(f'Predicted results: {y_predict}')

# Compute and print the accuracy score
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy score: {accuracy}')
