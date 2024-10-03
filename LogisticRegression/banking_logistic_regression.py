import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Read data
# Change to data path on your computer
data = pd.read_csv('banking.csv')
data.head()

# Remove rows where 'default', 'housing', or 'loan' columns contain 'unknown'
data = data[(data['default'] != 'unknown') & (data['housing'] != 'unknown') & (data['loan'] != 'unknown')]

# Convert the 'month' column to numeric values
month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
                 'nov': 11, 'dec': 12}
data['month'] = data['month'].map(month_mapping)

# Convert the 'day_of_week' column to numeric values
day_mapping = {'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}
data['day_of_week'] = data['day_of_week'].map(day_mapping)

# Convert binary fields
data['default'] = data['default'].map({'yes': 1, 'no': 0})
data['housing'] = data['housing'].map({'yes': 1, 'no': 0})
data['loan'] = data['loan'].map({'yes': 1, 'no': 0})

# Convert categorical fields using one-hot encoding
marital_dummies = pd.get_dummies(data['marital'], prefix='marital')
marital_dummies.drop('marital_unknown', axis=1, inplace=True)
data = pd.concat([data, marital_dummies], axis=1)

job_dummies = pd.get_dummies(data['job'], prefix='job')
job_dummies.drop('job_unknown', axis=1, inplace=True)
data = pd.concat([data, job_dummies], axis=1)

education_dummies = pd.get_dummies(data['education'], prefix='education')
education_dummies.drop('education_unknown', axis=1, inplace=True)
data = pd.concat([data, education_dummies], axis=1)

contact_dummies = pd.get_dummies(data['contact'], prefix='contact')
data = pd.concat([data, contact_dummies], axis=1)

poutcome_dummies = pd.get_dummies(data['poutcome'], prefix='poutcome')
data = pd.concat([data, poutcome_dummies], axis=1)

# Convert 'pdays' values
data['pdays'] = data['pdays'].apply(lambda row: 0 if row == -1 else 1)

# Drop unused columns
data.drop(['job', 'education', 'marital', 'contact', 'poutcome'], axis=1, inplace=True)

# Handle missing values
data.dropna(how='any', axis=0, inplace=True)

# Split the data into training set and validation set
x_data = data.drop('y', axis=1)
y_data = data['y']
train_size = int(len(x_data) * 0.8)
x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

# Train a Logistic Regression model
start_logistic = time.perf_counter()  # start count execution time

logistic_regression = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
logistic_regression.fit(x_train, y_train)
y_pred_logistic = logistic_regression.predict(x_test)

end_logistic = time.perf_counter()  # end count execution time

# Evaluate Logistic Regression approach results
print('Logistic Regression approach:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_logistic)}')
print(f'Precision: {precision_score(y_test, y_pred_logistic)}')
print(f'Recall: {recall_score(y_test, y_pred_logistic)}')
print(f'F1 Score: {f1_score(y_test, y_pred_logistic)}')
print(f'Execution time: {end_logistic - start_logistic}s\n')

# Train a Naive Bayes model
start_naive = time.time()
naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)
y_pred_naive = naive_bayes.predict(x_test)
end_naive = time.time()

# Evaluate Naive Bayes Classifier approach results
print('Naive Bayes Classifier approach:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_naive)}')
print(f'Precision: {precision_score(y_test, y_pred_naive)}')
print(f'Recall: {recall_score(y_test, y_pred_naive)}')
print(f'F1 Score: {f1_score(y_test, y_pred_naive)}')
print(f'Excution time: {end_naive - start_naive}')