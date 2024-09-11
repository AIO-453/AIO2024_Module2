import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Dữ liệu
data = {
    'Age': [23, 25, 27, 29, 29],
    'Likes English': [0, 1, 1, 0, 0],
    'Likes AI': [0, 1, 0, 1, 0],
    'Salary': [200, 400, 300, 500, 400]
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Định nghĩa biến đầu vào (chỉ sử dụng cột Likes AI) và nhãn (Salary)
X = df[['Likes AI']]  # Sử dụng cột Likes AI làm biến đầu vào
y = df['Salary']  # Cột Salary là giá trị thực tế

# Khởi tạo mô hình Decision Tree Regressor
tree_regressor = DecisionTreeRegressor()

# Huấn luyện mô hình với cột Likes AI
tree_regressor.fit(X, y)

# Dự đoán giá trị Salary
y_pred = tree_regressor.predict(X)

# Hàm tính SSE
def calculate_sse(y_true, y_pred):
    # SSE = sum of squared errors
    sse = np.sum((y_true - y_pred) ** 2)
    return sse

# Tính SSE cho cột Likes AI
sse = calculate_sse(y, y_pred)
print(f'SSE khi sử dụng cột "Likes AI" làm biến đầu vào: {sse}')
