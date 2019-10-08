#
# Calcule des valeurs propres et de AP - PD
#

# Imports
import numpy as np
import numpy.linalg as lin

# Matrice A
A = np.matrix([[-3, -2, 0], [14, 7, -1], [-6, -3, 1]])

print("Matrice A : \n{}".format(A))

# Calcule valeurs propres + vecteurs propres
valeurs_propres,  vecteurs_propres = lin.eig(A)

print("Valeurs propres : \n{}".format(valeurs_propres))
print("Vecteurs propres : \n{}".format(vecteurs_propres))

# Crée la matrice diagonale
D = np.matrix([
    [valeurs_propres[0], 0, 0],
    [0, valeurs_propres[1], 0],
    [0, 0, valeurs_propres[2]]
])

print("Matrice diagonale D : \n{}".format(D))

# Crée la matrice P
P = np.empty((3, 3))
P[:, 0] = vecteurs_propres[0]
P[:, 1] = vecteurs_propres[1]
P[:, 2] = vecteurs_propres[2]

print("Matrice P : \n{}".format(P))

# Calcule P-1
Pinv = lin.inv(P)

print("Inverse de P : \n{}".format(Pinv))

# Calcule AP et PD
AP = A @ P
PD = P @ D

print("AP - PD = \n{}".format(AP - PD))

# Calcule PDP-1
print("PDP-1 = \n{}".format(P @ (D @ Pinv)))

# Calcule la valeur de conditionnement
print("Valeur de conditionnement : {}\n".format(lin.cond(P)))
