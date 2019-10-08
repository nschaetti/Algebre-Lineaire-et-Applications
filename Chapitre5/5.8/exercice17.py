#
# Méthode de la puissance itérée
#

# Imports
import numpy as np
import numpy.linalg as lin
import sys

# Matrice A
# A = np.matrix([[6, 7], [8, 5]])
# A = np.matrix([[8, 0, 12], [1, -2, 1], [0, 3, 0]])
# A = np.matrix([[5, 2], [2, 2]])
# A = np.matrix([[10, -8, -4], [-8, 13, 4], [-4, 5, 4]])
A = np.matrix([[10, 7, 8, 7], [7, 5, 6, 5], [8, 6, 10, 9], [7, 5, 9, 10]])

# Dimension
dim = 4

# Nombre d'itérations
n_iterations = 20

# Series
Xk = np.zeros((n_iterations + 1, dim))
Vk = np.zeros(n_iterations)
Muk = np.zeros(n_iterations)
Yk = np.zeros((n_iterations + 1, dim))

# 1. Initial guest
a = float(sys.argv[1])

# 2. Starting point x0
# x0 = np.array([1, 0])
# x0 = np.array([1, 0, 0])
x0 = np.array([1, 0, 0, 0])
Xk[0] = x0
Yk[0] = x0

# Loop
for k in range(n_iterations):
    # a. (A - aI)yk = xk
    yk = lin.inv(A - a * np.eye(dim)) @ Xk[k]
    Yk[k + 1] = yk
    # b. mk
    Mk = np.max(np.abs(yk))
    # c. vk = a + (1 / Mk)
    vk = a + (1.0 / Mk)
    Vk[k] = vk
    # d. xk1 = (1.0 / Mk)yk
    Xk[k + 1] = (1.0 / Mk) * yk
# end for

print(Vk)
print(Xk)
print(Yk)

print(Xk[-1] / lin.norm(Xk))

print(lin.eig(A))
