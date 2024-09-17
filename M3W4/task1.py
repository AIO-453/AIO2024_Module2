# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 3. Read dataset
df = pd.read_csv('/content/Housing.csv')

# 4. xử lý categorical data
categorical_cols = df.select_dtypes(include='object').columns.to_list()

data = df.copy()
encoder = OrdinalEncoder()
data[categorical_cols] = encoder.fit_transform(data[categorical_cols])
data.head()

# 5. normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 6. tách bộ dữ liệu X,y
X = scaled_data[:, :-1]
y = scaled_data[:, -1]

# 7. train test split
test_size = 0.3
is_shuffle = True
random_state = 1
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=is_shuffle, random_state=random_state)

# 8. Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 9. Đánh giá model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
print(f'MSE: {mse}')
print(f'MAE: {mae}')