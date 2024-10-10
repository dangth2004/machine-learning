from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Read data
# Change to data path on your computer
glass_df = pd.read_csv('glass.csv')
print(glass_df.info())

glass_types = glass_df['Type'].unique()
print(glass_types)
print(glass_df['Type'].value_counts())
x_data = glass_df[glass_df.columns[:-1]]
y_data = glass_df['Type']

# Split the training set and the data set
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.25, random_state=42)

# Multinomial Logistic Regression (softmax) approach
softmax = LogisticRegression(multi_class='multinomial', max_iter=5000)
# Train model
softmax.fit(x_train, y_train)
# Predict results
y_predict = softmax.predict(x_test)

# Print accuracy score and confusion matrix
print(f'\nThe accuracy score: {accuracy_score(y_test, y_predict)}')
print(f'The confusion matrix: {confusion_matrix(y_test, y_predict)}')
