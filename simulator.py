#!/bin/python
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import time

from contextlib import redirect_stdout

from perturbate import *
from population import *

from config import *


''' Universal model generator '''
def generate_model(model):
    #sestava matrike in maske za linearno modifikacijo
    m_mat = np.zeros((model['proteins'], model['proteins']))
    for i, (beta, pos, type) in enumerate(zip(model['betas'], model['mod'], model['type'])):
        if type != 0:
            m_mat[i, i] = -beta
            m_mat[pos, i] = beta

    ad_mat = -np.repeat([model['deltas']], model['proteins'], axis=0).T
    np.fill_diagonal(ad_mat, 0)

    def repressilator_model(p, t):
        # Genska represija
        dg = model['alphas'] * np.prod(np.where(model['M'] != 0, (0 <= np.where(model['M'] > 0, p - model['M'], -model['M'] - p )).astype(int), 1), axis=1)

        # Modifikacija
        lm = np.dot(m_mat, p) # Linearna deg
        #dp = np.where(m_map, dm, dg) # linearna

        #Encimska modifikacija
        dem_brez = p * (1 / (model['Km'] + p)) # Encimska modifikacija pred mnozejem z beto
        em = np.dot(m_mat, dem_brez)
        #dp = np.where(em_map, dem, dp)

        dp = np.where(model['type'] == 0, dg, np.where(model['type'] == 1, lm, em))

        # Degradacija
        dl = -model['deltas'] * dp # Linearna degradacija
        da = np.dot(ad_mat, dp) * dp # Aktivna degradacija
        de = -model['deltas'] * (dp / (model['Km'] + dp)) # Encimska degradacija

        dp = np.where(model['deg_type']==0, dl, np.where(model['deg_type']==1, da, de)) # Filtered degradation

        return dp
    return repressilator_model



def simulate(sub):
    # Generate timestamps
    t = np.arange(0, 100, dt)


    # Dp ODE integration
    #tim = time.clock()
    #sub = initiate_subject()
    # print(sub)

    #r, info = integrate.odeint(generate_model(sub), np.random.randint(0, 10, size=sub['proteins']), t, args=(), full_output=False, printmessg=False)
    #r = integrate.odeint(generate_model(sub), np.random.randint(0, 10, size=sub['proteins']), t, args=(), full_output=False, printmessg=False)
    r = integrate.odeint(generate_model(sub), np.random.rand(sub['proteins']), t)
    #print(time.clock() - tim)


    #print(r)

#    plt.plot(t, r)
    #plt.xlabel('Time')
    #plt.ylabel('Protein concetration')
    #plt.show()

    return r


#sub = initiate_subject()
pop = generate_population(100)
for i in range(100):

    print("Generation:" + str(i))

    res = [simulate(sub) for sub in pop]

    perturbate(pop)

#print(generate_population())

exit(0)
