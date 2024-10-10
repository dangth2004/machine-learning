import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = datasets.load_iris()
x_data = iris.data[:, :4]  # We take the first 4 features
y_data = iris.target

# Normalize the data using StandardScaler
scaler = StandardScaler()
x_norm = scaler.fit_transform(x_data)

# PCA for 2-dimensional visualization
pca = PCA(n_components=2)
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


# Multinomial Logistic Regression approach
def softmax_approach(x_train, x_test, y_train, y_test):
    # Multinomial logistic regression with a fixed random state
    softmax = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=200)
    # Train model
    softmax.fit(x_train, y_train)
    # Predict results
    y_pred = softmax.predict(x_test)
    return accuracy_score(y_test, y_pred)


# Split the dataset into training and testing sets using the normalized data
x_train, x_test, y_train, y_test = train_test_split(x_norm, y_data, test_size=0.2, random_state=42)

# Compute and print the accuracy score using logistic regression on the normalized data
print(f'Accuracy score using normalized features: {softmax_approach(x_train, x_test, y_train, y_test)}')
