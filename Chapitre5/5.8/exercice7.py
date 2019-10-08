#
# Méthode de la puissance itérée
#

# Imports
import numpy as np

# Rayleigh
def rayleigh(xk, A):
    x = np.matrix(xk).T
    return (x.T @ A @ x) / (x.T @ x)
# end rayleigh

# Matrice A
# A = np.matrix([[6, 7], [8, 5]])
# A = np.matrix([[8, 0, 12], [1, -2, 1], [0, 3, 0]])
A = np.matrix([[5, 2], [2, 2]])

# Starting point x0
x0 = np.array([1, 0])
# x0 = np.array([1, 0, 0])

# Nombre d'itérations
n_iterations = 10

# Series
Xk = np.zeros((n_iterations + 1, 2))
Muk = np.zeros(n_iterations)
Xk[0] = x0

# Loop
for k in range(n_iterations):
    Axk = A @ Xk[k]
    # Muk[k] = np.max(np.abs(Axk))
    Muk[k] = rayleigh(Axk, A)
    Xk[k+1] = (1.0 / Muk[k]) * Axk
    Xk[k + 1] = Xk[k+1] / np.max(np.abs(Xk[k+1]))
# end for

# Results
print(Muk)
print(Xk)
