import ipdb
import numpy as np

Nc = 32
Delta = 5

A = np.ones((Nc, Nc))
B = np.ones(Nc)
for n in range(Nc):
    B[n] = (1/Delta) * ((2*n+1)**0.5)
    for k in range(Nc):
        if k <= n:
            A[n, k] = - (1/Delta)*((2*n+1)**(0.5)
                                   )*((2*k+1)**(0.5)) * 1
        else:
            A[n, k] = - (1/Delta)*((2*n+1)**(0.5)) * \
                ((2*k+1)**(0.5)) * (-1)**(n-k)

lamb, v = np.linalg.eig(A)
ipdb.set_trace()
