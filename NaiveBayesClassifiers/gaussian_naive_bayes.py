import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('breast-cancer-wisconsin.data')

# Đặt tên cho các cột
data.columns = ['Sample code number', 'Class', 'Clump Thickness',
                'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']

# Tách riêng các mẫu lành tính và ác tính
benign_samples = data[data['Class'] == 2]  # Mẫu lành tính (Diagnosis = 2)
malignant_samples = data[data['Class'] == 4]  # Mẫu ác tính (Diagnosis = 4)

# Chọn ngẫu nhiên 80 mẫu lành tính và 40 mẫu ác tính cho tập test
benign_test = benign_samples.sample(n=80, random_state=42)
malignant_test = malignant_samples.sample(n=40, random_state=42)

# Phần còn lại là dữ liệu training
benign_train = benign_samples.drop(benign_test.index)
malignant_train = malignant_samples.drop(malignant_test.index)

# Gộp lại tập test và training
test_data = pd.concat([benign_test, malignant_test])
train_data = pd.concat([benign_train, malignant_train])

# Tách dữ liệu (X) và nhãn (y) cho tập training và test
X_train = train_data.drop(['Sample code number', 'Class'], axis=1)
y_train = train_data['Class'].apply(lambda x: 1 if x == 4 else 0)  # 1: ác tính, 0: lành tính

X_test = test_data.drop(['Sample code number', 'Class'], axis=1)
y_test = test_data['Class'].apply(lambda x: 1 if x == 4 else 0)  # 1: ác tính, 0: lành tính

# Huấn luyện mô hình Gaussian Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = classifier.predict(X_test)

# Tính toán các chỉ số Accuracy, Precision, và Recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

# In kết quả ra màn hình
print(f'Accuracy: {accuracy:.2f}')  # Accuracy: 0.96
print(f'Precision: {precision:.2f}')  # Precision: 0.91
print(f'Recall: {recall:.2f}')  # Recall: 0.97
print(f'F1_score: {f1_score:.2f}')  # F1_score: 0.94
