import numpy as np


def ll_fucntion(xx, yy):
    return xx + yy

def border(x):
    return x**2


h = 0.01
n = 100

net = np.((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            net[i][j] = 1


