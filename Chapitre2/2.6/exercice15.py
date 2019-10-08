
# Imports
import numpy as np

# Iteration
iteration = 20

# Production matrix
C = np.matrix([[0.1588, 0.0064, 0.0025, 0.0304, 0.0014, 0.0083, 0.1594], [0.0057, 0.2645, 0.0436, 0.0099, 0.0083, 0.0201, 0.3413], [0.0264, 0.1506, 0.3557, 0.0139, 0.0142, 0.0070, 0.0236], [0.3299, 0.0565, 0.0495, 0.3636, 0.0204, 0.0483, 0.0649], [0.0089, 0.0081, 0.0333, 0.0295, 0.3412, 0.0237, 0.0020], [0.1190, 0.0901, 0.0996, 0.1260, 0.1722, 0.2368, 0.3369], [0.0063, 0.0126, 0.0196, 0.0098, 0.0064, 0.0132, 0.0012]])

# Demande finale
d = np.array([74000, 56000, 10500, 25000, 17500, 196000, 5000])

# Compute (I - C)^-1
iteratedInvC = np.eye(7)
for i in range(iteration):
    iteratedInvC += C**(i+1)
# end for

# Calcule la production
invx = np.linalg.inv(np.eye(7) - C) @ d
iteratedx = iteratedInvC @ d

# Print
print(C)
print(d)
print(np.linalg.inv(np.eye(7) - C))
print(iteratedInvC)
print(invx)
print(iteratedx)
