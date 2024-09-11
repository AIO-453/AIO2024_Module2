import numpy as np
import pandas as pd

# Hàm tính Gini
def gini(labels):
    # Đếm số lượng các nhãn khác nhau
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    
    # Tính Gini
    gini_value = 1 - np.sum([p ** 2 for p in probabilities if p > 0])
    return gini_value

# Ví dụ: tạo dữ liệu mẫu
data = {
    'Age': [23, 25, 27, 29, 29],
    'Likes English': [0, 1, 1, 0, 0],
    'Likes AI': [0, 1, 0, 1, 0],
    'Raise Salary': [0, 0, 1, 1, 0]  # Nhãn
}

df = pd.DataFrame(data)

# Cột nhãn 'Raise Salary'
labels = df['Raise Salary'].values

# Tính Gini cho nhãn
gini_value = gini(labels)
print(f'Gini của cột "Raise Salary": {gini_value}')
# 2c