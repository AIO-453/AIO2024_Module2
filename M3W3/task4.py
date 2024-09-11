import numpy as np
import pandas as pd

# Hàm tính Gini cho một tập dữ liệu
def gini(labels):
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    gini_value = 1 - np.sum([p ** 2 for p in probabilities if p > 0])
    return gini_value

# Tạo bộ dữ liệu mẫu
data = {
    'Age': [23, 25, 27, 29, 29],
    'Likes English': [0, 1, 1, 0, 0],
    'Likes AI': [0, 1, 0, 1, 0],
    'Raise Salary': [0, 0, 1, 1, 0]  # Nhãn
}

df = pd.DataFrame(data)

# Chia tập dữ liệu theo điều kiện 'Age <= 26'
group_D1 = df[df['Age'] <= 26]['Raise Salary']
group_D2 = df[df['Age'] > 26]['Raise Salary']

# Tính Gini cho từng nhóm
gini_D1 = gini(group_D1)
gini_D2 = gini(group_D2)

# Tính Gini tổng hợp có trọng số
total_size = len(df)
weighted_gini = (len(group_D1) / total_size) * gini_D1 + (len(group_D2) / total_size) * gini_D2

# In kết quả
print(f'Gini của nhóm "Age <= 26" (D1): {gini_D1}')
print(f'Gini của nhóm "Age > 26" (D2): {gini_D2}')
print(f'Gini tổng hợp cho node gốc "Age" với điều kiện phân chia "Age <= 26": {weighted_gini}')
