
# Imports
import numpy as np

# Cramer's method
def solve_cramer(input_matrix, input_vector):
    """
    Cramer's method
    :param input_matrix:
    :param input_vector:
    :return:
    """
    # Sizes
    (n, m) = input_matrix.shape

    # Solution
    x = np.zeros((m, 1))

    # Det(A)
    detA = np.linalg.det(input_matrix)

    # For each components
    for i in range(m):
        Ai = input_matrix.copy()
        Ai[:, i:i+1] = input_vector
        x[i] = np.linalg.det(Ai) / detA
    # end for

    return x
# end solve_cramer

# Random 4x4 matrix
A = np.random.rand(4, 4)
print(A)

# Random b vector
b = np.random.rand(4, 1)
print(b)

# Solve with Cramer
print(solve_cramer(A, b))

# Get A-1 and x
invA = np.linalg.inv(A)
print(invA @ b)

