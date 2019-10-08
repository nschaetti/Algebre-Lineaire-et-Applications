
# Imports
import numpy as np

# Iteration
iteration = 40

# Matrix
A = np.matrix([[0.0, 0.2, 0.3], [0.0, 0.6, 0.3], [0.9, 0.2, 0.4]])


# Iteration
for i in range(iteration):
     print(A**(i+1))
# end for
