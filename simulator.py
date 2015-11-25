#!/bin/python

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import time

alpha = 100
Kd = 1
delta = 1

dt = 0.01

# Map of activators and repressors
gmap = np.matrix([[0,-Kd,0],
				  [0,0,-Kd],
		    	  [-Kd,0,0]])

# Alphas and deltas
alphas = np.array([alpha,alpha,alpha])
deltas = np.array([delta,delta,delta])

''' Universal model generator '''
def generate_model(gmap, alphas, deltas):
	# Prepare matrix for computation
	mmat = np.transpose(np.append(np.sign(gmap), np.sum(gmap, axis=0)*-1, axis=0))
	def repr_model(p, t):
		m = mmat.dot(np.append(p,[1]))
		m = np.subtract(np.where(m >= 0, alphas, 0), np.multiply(deltas, p)).flatten()
		return m

	return repr_model

''' Reprissilator model '''
def repressilator_model(P, t):
	P1=[0,0,0]
	P1[0] = alpha*int(0 <= (Kd - P[2])) - delta * P[0]
	P1[1] = alpha*int(0 <= (Kd - P[0])) - delta * P[1]
	P1[2] = alpha*int(0 <= (Kd - P[1])) - delta * P[2]
	return P1


#r = integrate.odeint(repressilator_model, [100, 0, 0], t)

# Generate timestamps
t = np.arange(0, 100, dt)

# Dp ODE integration
tim = time.clock()
r = integrate.odeint(generate_model(gmap,alphas,deltas), np.array([100, 0, 0]), t)
print(time.clock()-tim)

print(r)
plt.plot(t,r[:,0])
plt.show()