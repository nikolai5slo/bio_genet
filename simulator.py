#!/bin/python

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import time

PROTEIN_NUM = 3
T_MAX = 100

alpha = 100
delta = 1
Kd = 1
Km = 2
dt = 0.01

# Map of activators and repressors
#gmap = np.array([[0,-Kd,0],
#                 [0,0,-Kd],
#	  	          [-Kd,0,0]])

#initiatie semi-random map of activators
gmap = np.zeros((PROTEIN_NUM, PROTEIN_NUM))
idx = np.random.randint(1, PROTEIN_NUM, size=1)[0]
for i in range(gmap.shape[0]):
    gmap[idx,i] = -Kd
    idx += 1
    if idx >= PROTEIN_NUM:
        idx = 0

print(gmap)

''' Universal model generator '''
def repressilator_model(p, t, M, degradation='linear'):
    dp = alpha * np.prod(np.where(M != 0, (0 <= np.sign(M)*(M + p)).astype(int), 1), axis=1)

    if degradation == 'linear':
        dp = dp - delta * p
    elif degradation == 'enzyme':
        dp = dp - delta * (p / (p + Km))

    return dp

#r = integrate.odeint(repressilator_model, [100, 0, 0], t)

# Generate timestamps
t = np.arange(0, T_MAX, dt)

# Dp ODE integration
tim = time.clock()
r, info = integrate.odeint(repressilator_model, np.random.randint(0, 10, size=PROTEIN_NUM), t, args=(gmap,'linear'), full_output=True, printmessg=True)
print(time.clock()-tim)
#print(info)

print(r)
plt.plot(t, r)
plt.show()
