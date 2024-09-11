import numpy as np
import pandas as pd

# Hàm tính entropy
def entropy(labels):
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    entropy_value = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy_value

# Hàm tính Information Gain
def information_gain(data, target, feature):
    # Tính entropy của toàn bộ tập dữ liệu (H(D))
    total_entropy = entropy(data[target].values)
    
    # Tính entropy có trọng số của các tập con
    values = data[feature].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[feature] == value]
        subset_entropy = entropy(subset[target].values)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    
    # Tính Gain
    gain = total_entropy - weighted_entropy
    return gain

# Dữ liệu mẫu
data = {
    'Age': [23, 25, 27, 29, 29],
    'Likes English': [0, 1, 1, 0, 0],
    'Likes AI': [0, 1, 0, 1, 0],
    'Raise Salary': [0, 0, 1, 1, 0]  # Nhãn
}

df = pd.DataFrame(data)

# Tính Gain khi 'Likes English' là node gốc
gain_likes_english = information_gain(df, 'Raise Salary', 'Likes English')
print(f'Gain khi "Likes English" là node gốc: {gain_likes_english}')
# 6d