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
		    	  [-Kd,0,2]])

# Alphas and deltas
alphas = np.array([alpha,alpha,alpha])
deltas = np.array([delta,delta,delta])

''' Universal model generator '''
def generate_model(gmap, alphas, deltas):
	# Prepare matrixes for computation
	amat = np.where(gmap > 0, 1, 0) 
	amat = np.transpose(np.append(amat, [np.sum(np.where(gmap > 0,-gmap , 0), axis=0)], axis=0))

	rmat = np.where(gmap < 0, -1, 0) 
	rmat = np.transpose(np.append(rmat, [np.sum(np.where(gmap < 0,-gmap , 0), axis=0)], axis=0))

	def repr_model(p, t):
		tp = np.append(p,[1])
		m = np.multiply(np.where(amat.dot(tp) >= 0, alphas, 0), np.where(rmat.dot(tp) >= 0, 1, 0))
		m = np.subtract(m, np.multiply(deltas, p)).flatten()
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