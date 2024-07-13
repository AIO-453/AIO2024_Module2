import numpy as np

# 1. Các phép toán trên vector và ma trận.

# (a) Độ dài của vector:


def compute_vector_length(vector):
    len_of_vector = np.sqrt(np.sum(np.square(vector)))
    return len_of_vector

# (b) Phép tích vô hướng:


def compute_dot_product(vector1, vector2):
    result = np.dot(vector1, vector2)
    return result

# (c) Nhân vector với ma trận:


def matrix_multi_vector(matrix, vector):
    result = np.dot(matrix, vector)
    return result

# (d) Nhân ma trận với ma trận:


def matrix_multi_matrix(matrix_A, matrix_B):
    result = np.dot(matrix_A, matrix_B)
    return result

# (e) Ma trận nghịch đảo:


def inverse_matrix(matrix):
    result = np.linalg.inv(matrix)
    return result

# 2. Eigenvector và eigenvalues:


def compute_eigenvalues_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

# 3. Cosine Similarity:


def compute_cosine(vector1, vector2):
    cos_sim = compute_dot_product(
        vector1, vector2) / (compute_vector_length(vector1) * compute_vector_length(vector2))
    return cos_sim


if __name__ == '__main__':
    # 1a
    vector = np. array([-2, 4, 9, 21])
    result = compute_vector_length([vector])
    print(round(result, 2))

    # 2b
    v1 = np. array([0, 1, -1, 2])
    v2 = np. array([2, 5, 1, 0])
    result = compute_dot_product(v1, v2)
    print(round(result, 2))

    # 3a
    x = np. array([[1, 2],
                   [3, 4]])
    k = np. array([1, 2])
    print('result \n', x.dot(k))

    # 4b
    x = np. array([[-1, 2],
                   [3, -4]])
    k = np. array([1, 2])
    print('result \n', x@k)

    # 5a
    m = np. array([[-1, 1, 1], [0, -4, 9]])
    v = np. array([0, 2, 1])
    result = matrix_multi_vector(m, v)
    print(result)

    # 6c
    m1 = np. array([[0, 1, 2], [2, -3, 1]])
    m2 = np. array([[1, -3], [6, 1], [0, -1]])
    result = matrix_multi_matrix(m1, m2)
    print(result)

    # 7 a
    m1 = np.eye(3)
    m2 = np. array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    result = m1@m2
    print(result)

    # 8d
    m1 = np.eye(2)
    m1 = np. reshape(m1, (-1, 4))[0]
    m2 = np. array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    result = m1@m2
    print(result)

    # 9b
    m1 = np. array([[1, 2], [3, 4]])
    m1 = np. reshape(m1, (-1, 4), "F")[0]
    m2 = np. array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    result = m1@m2
    print(result)

    # 10a
    m1 = np. array([[-2, 6], [8, -4]])
    result = inverse_matrix(m1)
    print(result)

    # 11a
    matrix = np. array([[0.9, 0.2], [0.1, 0.8]])
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(matrix)
    print(eigenvectors)

    # 12c
    x = np. array([1, 2, 3, 4])
    y = np. array([1, 0, 3, 0])
    result = compute_cosine(x, y)
    print(round(result, 3))
