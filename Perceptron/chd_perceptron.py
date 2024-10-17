import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Read data
# Change your path on your computer
data = pd.read_csv('framingham.csv')
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
x_data = StandardScaler().fit_transform(x_data)
y_data = np.asarray(y_data)

# Dimensionality reduction
pca = PCA(n_components=2, random_state=42)
x_data_pca = pca.fit_transform(x_data)

# Data visualization
pc1 = x_data_pca[:, 0]
pc2 = x_data_pca[:, 1]
sns.scatterplot(x=pc1, y=pc2, hue=y_data)
plt.show()

# Split training set and validation set
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.7, random_state=20)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_data_pca, y_data,
                                                                    train_size=0.7, random_state=20)


# Perceptron approach
def perceptron_approach(x_train, x_test, y_train, y_test):
    # Train model
    perceptron = Perceptron(fit_intercept=False)
    perceptron.fit(x_train, y_train)
    # Predict results
    y_predict = perceptron.predict(x_test)
    # Print accuracy score, precision score, recall score, coefficient
    print(f'Accuracy score: {accuracy_score(y_test, y_predict)}')
    print(f'Precision score: {precision_score(y_test, y_predict)}')
    print(f'Recall score: {recall_score(y_test, y_predict)}')
    print(f'Coefficient: {perceptron.coef_}')


# Logistic Regression approach
def logistic_regression_approach(x_train, x_test, y_train, y_test):
    # Train model
    logistic = LogisticRegression(fit_intercept=False)
    logistic.fit(x_train, y_train)
    # Predict results
    y_predict = logistic.predict(x_test)
    # Print accuracy score, precision score, recall score, coefficient
    print(f'Accuracy score: {accuracy_score(y_test, y_predict)}')
    print(f'Precision score: {precision_score(y_test, y_predict)}')
    print(f'Recall score: {recall_score(y_test, y_predict)}')
    print(f'Coefficient: {logistic.coef_}')


# Naive Bayes approach
def naive_bayes_approach(x_train, x_test, y_train, y_test):
    # Train model
    naive_bayes = GaussianNB()
    naive_bayes.fit(x_train, y_train)
    # Predict results
    y_predict = naive_bayes.predict(x_test)
    # Print accuracy score, precision score, recall score
    print(f'Accuracy score: {accuracy_score(y_test, y_predict)}')
    print(f'Precision score: {precision_score(y_test, y_predict)}')
    print(f'Recall score: {recall_score(y_test, y_predict)}')


# Print result
print('Original data:')
print('Perceptron approach: ')
perceptron_approach(x_train, x_test, y_train, y_test)
print('\nLogistic Regression approach')
logistic_regression_approach(x_train, x_test, y_train, y_test)
print('\nNaive Bayes approach')
naive_bayes_approach(x_train, x_test, y_train, y_test)

print('\nDimensionality reduction:')
print('Perceptron approach: ')
perceptron_approach(x_train_pca, x_test_pca, y_train_pca, y_test_pca)
print('\nLogistic Regression approach')
logistic_regression_approach(x_train_pca, x_test_pca, y_train_pca, y_test_pca)
print('\nNaive Bayes approach')
naive_bayes_approach(x_train_pca, x_test_pca, y_train_pca, y_test_pca)
