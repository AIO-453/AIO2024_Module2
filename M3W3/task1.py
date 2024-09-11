import numpy as np
import pandas as pd

# Hàm tính entropy
def entropy(labels):
    # Đếm số lượng các nhãn khác nhau
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    
    # Tính entropy
    entropy_value = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy_value

labels = [0, 0, 1, 1]

# Tính entropy cho nhãn
entropy_value = entropy(labels)
print(f'Entropy của tập dữ liệu cân bằng: {entropy_value}')
# 1a