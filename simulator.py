#!/bin/python

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

alpha = 100
Kd = 1
delta = 1

dt = 0.01

def repressilator_model(P, t):
	P1=[0,0,0]
	P1[0] = alpha*int(0 <= (Kd - P[2])) - delta * P[0]
	P1[1] = alpha*int(0 <= (Kd - P[0])) - delta * P[1]
	P1[2] = alpha*int(0 <= (Kd - P[1])) - delta * P[2]
	return P1

t = np.arange(0,100,dt)
r = integrate.odeint(repressilator_model, [100, 0, 0], t)
print(r[:,0])
plt.plot(t,r[:,0])
plt.show()