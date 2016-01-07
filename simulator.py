#!/bin/python -W ignore
from scipy import integrate

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import redirect
import os
from time import gmtime, strftime
import sys

from contextlib import redirect_stdout

from perturbate import *
from population import *
from analysis import *
from config import *

directory = "results/" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
if not os.path.exists(directory):
    os.makedirs(directory)

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

    def osc_model(p, t):
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
    return osc_model

def simulate(sub):
    # Generate timestamps
    t = np.arange(0, T_MAX, dt)

    # Dp ODE integration
    #tim = time.clock()
    #sub = initiate_subject()
    # print(sub)

    #r, info = integrate.odeint(generate_model(sub), np.random.randint(0, 10, size=sub['proteins']), t, args=(), full_output=False, printmessg=False)
    #r = integrate.odeint(generate_model(sub), np.random.randint(0, 10, size=sub['proteins']), t, args=(), full_output=False, printmessg=False)
    r = integrate.odeint(generate_model(sub), np.ones(sub['proteins']), t)
    #print(time.clock() - tim)

    #print(r)

#    plt.plot(t, r)
    #plt.xlabel('Time')
    #plt.ylabel('Protein concetration')
    #plt.show()

    return r

#sub = initiate_subject()
input_protein = np.array([IN_AMPL * math.sin(2 * math.pi * IN_FREQ * t) + IN_AMPL for t in np.arange(0, T_MAX, dt)])
plt.plot(input_protein)
plt.xlabel('Time')
plt.ylabel('Protein concetration')
plt.savefig(directory + "/ref.png")
plt.close()

best_score = sys.maxsize
pop = generate_population(POPULATION_SIZE)
for i in range(1000):
    #with redirect.stdout_redirected():
    res = [simulate(sub) for sub in pop]

    # by default first protein of a subject is considered as output
    evals = [(j, fitness(input_protein, res[j][:,0])) for j in range(len(res))]
    evals.sort(key=lambda t: t[1])
    print("Generation #%d - best score: %4d %.4f" % (i+1, evals[0][0], evals[0][1]))

    if evals[0][1] < best_score:
        best_score = evals[0][1]
        plt.plot(res[evals[0][0]][:,0])
        plt.xlabel('Time')
        plt.ylabel('Protein concetration')
        plt.savefig(directory + "/gen" + str(i+1) + "_sco" + str(best_score) + ".png")
        plt.close()

    pop = perturbate(pop, evals)

exit(0)
