import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Split the data into train, validation, and test sets with a ratio of 7:2:1
X_train, X_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, random_state=42)  # 0.33*0.3=0.1

# Scale the features
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_val = Scaler.transform(X_val)
X_test = Scaler.transform(X_test)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Validate the model
y_val_pred = knn.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Test the model
y_test_pred = knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')
