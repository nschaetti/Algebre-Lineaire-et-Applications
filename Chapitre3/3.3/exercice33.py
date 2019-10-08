
# Imports
import numpy as np

# Compute conjugate
def compute_conjugate(input_matrix):
    """
    Compute conjugate
    :param input_matrix:
    :return:
    """
    # Matrix size
    n = input_matrix.shape[0]
    m = input_matrix.shape[1]

    # Conjugate matrix
    c = np.zeros((n, m))

    # For each cofactor
    for i in range(n):
        for j in range(m):
            SubA = input_matrix.copy()
            SubA = np.delete(SubA, (i), axis=0)
            SubA = np.delete(SubA, (j), axis=1)
            c[i][j] = (-1)**(i + j) * np.linalg.det(SubA)
        # end for
    # end for

    return c
# end compute_conjugate

# Random 4x4 matrix
A = np.random.rand(4, 4)
# A = np.matrix([[2, 1, 3], [1, -1, 1], [1, 4, -2]])
print(A)

# B
B = (compute_conjugate(A)).T / np.linalg.det(A)
print(B)

# B - inv(a)
print(B - np.linalg.inv(A))

print(np.linalg.inv(A))