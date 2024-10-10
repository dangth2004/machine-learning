import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

# Read data
# Change to data path on your computer
warnings.filterwarnings('ignore')
data = pd.read_csv('pd_speech_features.csv')
data.head()
data['class'].value_counts()

# Set to training data (x, y)
x_data = np.asarray(data.drop(['class', 'id'], axis=1))
x_data = StandardScaler().fit_transform(x_data)
y_data = np.asarray(data['class'])

# PCA method
pca_plot = PCA()
result_plot = pca_plot.fit_transform(x_data)

# Visualization data
pc1 = - result_plot[:, 0]
pc2 = - result_plot[:, 1]
sns.scatterplot(x=pc1, y=pc2, hue=y_data)
plt.show()


# Naive Bayes Classifier approach
def naive_bayes_appoach(x_train, x_test, y_train, y_test):
    naive_bayes = GaussianNB()
    naive_bayes.fit(x_train, y_train)
    y_predict = naive_bayes.predict(x_test)
    return accuracy_score(y_test, y_predict)


# Logistic Regression approach
def log_reg_approach(x_train, x_test, y_train, y_test):
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    y_predict = log_reg.predict(x_test)
    return accuracy_score(y_test, y_predict)


# Reduce dimensions then split training - validation set
def reduce_split(n, X, y, train_size):
    # Reduce dimensions
    pca = PCA(n_components=n)
    x_data = pca.fit_transform(X)
    # Split training set and validation set
    x_train, x_test, y_train, y_test = train_test_split(x_data, y,
                                                        train_size=train_size, random_state=0)
    print(f'Reduce dimension then split training - validation set ({n} dimensions)')
    # Compute and print the accuracy of Logistic Regression approach
    print(f'The accuracy score: '
          f'{log_reg_approach(x_train, x_test, y_train, y_test)}\n')


# Split the training set then reduce dimensions
def split_reduce(n, X, y, train_size):
    # Split training set and validation set
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=train_size, random_state=0)
    # Reduce dimensions
    pca = PCA(n_components=n)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    print(f'Split the training - validation set then reduce dimensions ({n} dimensions)')
    # Compute and print the accuracy of Logistic Regression approach
    print(f'The accuracy score: '
          f'{log_reg_approach(x_train_pca, x_test_pca, y_train, y_test)}\n')


# Original data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=500, random_state=0)
print('Original data')
# Compute and print the accuracy of Naive Bayes Classifier approach
print(f'The accuracy score: '
      f'{log_reg_approach(x_train, x_test, y_train, y_test)}\n')


# Reduce dimensions then split training - validation set
reduce_split(n=200, X=x_data, y=y_data, train_size=500)
# Split the training set then reduce dimensions
split_reduce(n=200, X=x_data, y=y_data, train_size=500)


# Using PCA method for all the dataset
pca_full = PCA()
pca_full.fit(x_data)
# Compute explained variance ratio
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
# Compute the minimum number of dimensions required to retain at least 80% of the information.
min_dim = np.argmax(explained_variance >= 0.80) + 1
print(f'The minimum number of dimensions: {min_dim}\n')
# Reduce dimensions then split training - validation set (n_components = min_dim)
reduce_split(n=min_dim, X=x_data, y=y_data, train_size=500)
# Split the training set then reduce dimensions (n_components = min_dim)
split_reduce(n=min_dim, X=x_data, y=y_data, train_size=500)


# Original data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=500, random_state=1)
print('Original data')
# Compute and print the accuracy of Naive Bayes Classifier approach
print(f'The accuracy score (Naive Bayes approach): '
      f'{naive_bayes_appoach(x_train, x_test, y_train, y_test)}')
# Compute and print the accuracy of Logistic Regression approach
print(f'The accuracy score (Logistic Regression): '
      f'{log_reg_approach(x_train, x_test, y_train, y_test)}\n')


print('Reduce dimensions')
# Reduce dimensions
pca = PCA(n_components=min_dim)
x_data1 = pca.fit_transform(x_data)
# Split training set and validation set
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data1, y_data,
                                                    train_size=0.66, random_state=1)
# Compute and print the accuracy of Naive Bayes Classifier approach
print(f'The accuracy score (Naive Bayes approach): '
      f'{naive_bayes_appoach(x_train1, x_test1, y_train1, y_test1)}')
# Compute and print the accuracy of Logistic Regression approach
print(f'The accuracy score (Logistic Regression): '
      f'{log_reg_approach(x_train1, x_test1, y_train1, y_test1)}\n')