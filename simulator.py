#!/bin/python

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import pi
import scipy.fftpack
import analysis

PROTEIN_NUM_MAX = 5
KD_MAX = 100
PROTEINS_MAX = 3
ALPHA_MAX = 1

DEGRADATION_WEIGHTS = np.array([0.6,0.3,0.1])

PERTURBAIION_PROBABILITY = 0.1
PERTURBATION_WEIGHT_DEGRADATION = np.array([0.6, 0.2, 0.2])
PERTURBATION_WEIGHT_TYPE = np.array([0.6, 0.2, 0.2])

POPULATION_SIZE = 100
T_MAX = 20
dt = 0.1

def initiate_subject(num_proteins=5,alphas_type='scalar',deltas_type='scalar',degradation='linear'):
    if (num_proteins == None):
        num_proteins = np.random.randint(1, PROTEIN_NUM_MAX+1)

    gmap = np.zeros((num_proteins, num_proteins))
    idx = np.random.randint(1, num_proteins)

    Kd = np.random.random_sample() * KD_MAX
    for i in range(gmap.shape[0]):
        gmap[idx, i] = -Kd
        idx += 1
        if idx >= num_proteins:
            idx = 0

    ad = -np.random.rand(num_proteins, num_proteins)
    np.fill_diagonal(ad, 0)

    return {
        'Kd': Kd,
        'proteins': num_proteins,
        'alphas': np.random.random_sample(size=num_proteins) * ALPHA_MAX,
        #'Km': np.random.randint(1, Kd),
        'M': gmap,
        #'M': np.zeros((num_proteins,num_proteins)),
        'type' : np.random.randint(0, 3, size=num_proteins),# 0 - gensko izrazanje, 1 - linearna modifikacija, 2 - encimska modifikacija

        # Degradation
        'deg_type' : np.random.randint(0, 3, size=num_proteins), # 0 - linearna deg., 1 - aktivna deg., 2 - encimska deg.
        'deltas' : np.random.rand(num_proteins),
        'Km' : np.random.rand(num_proteins), # Vektorja, za encimsko degradacijo. Prvi stolpec je delta, drugi Km

        #'LD': np.random.rand(num_proteins),
        #'AD' : ad, # Matrika delt, za aktinvo degradacijo. Po diagonali so 0 ker ne more vplivati sam nase
        #'Km' : np.random.rand(num_proteins), # Vektorja, za encimsko degradacijo. Prvi stolpec je delta, drugi Km

        # Modification
        'betas' : np.random.rand(num_proteins),
        'mod' : np.random.randint(0, num_proteins, size=(num_proteins))
        #'LM' : np.random.randint(-10, num_proteins, size=(num_proteins))
        #'EM' : np.vstack((np.random.randint(-10, num_proteins, size=(num_proteins)), np.random.rand(num_proteins),np.random.rand(num_proteins))) #ENCIMSKA MODIFIKACIJA 3 vrstice, 1. na katerega vpliva, 2. beta, 3. Km
    }

def generate_population(size, num_proteins):
    subjects = []

    #zaenkrat se po utezeh dodaja le degradacija
    #verjetno bo treba se kaj spreminjat

    for i in range(size):
        degr_pos = np.random.rand()

        if degr_pos < DEGRADATION_WEIGHTS[0]:
            degradation_type = 'linear'
        elif degr_pos < np.sum(DEGRADATION_WEIGHTS[0:2]):
            degradation_type = 'enzyme'
        else:
            degradation_type = 'active'

        subjects.append(initiate_subject(num_proteins, degradation=degradation_type))

    return subjects

def perturbate(population):
    for subject in population:
        # doloci ali se ta osebek spreminja
        if np.random.rand() < PERTURBAIION_PROBABILITY:
            # TODO: dodati se sistem za nakljucno izbiro metode?


            # spreminjanje kineticnih parametrov
            a_map=np.random.choice(2, len(subject['proteins']), p=[0.9, 0.1])
            subject['alphas'] = np.where(a_map > 0, subject['alphas'] * np.random.rand() * 2, subject['alphas'])

            b_map=np.random.choice(2, len(subject['proteins']), p=[0.9, 0.1])
            subject['betas'] = np.where(b_map > 0, subject['betas'] * np.random.rand() * 2, subject['betas'])

            d_map=np.random.choice(2, len(subject['proteins']), p=[0.9, 0.1])
            subject['deltas'] = np.where(d_map > 0, subject['deltas'] * np.random.rand() * 2, subject['deltas'])

            ###################################
            # par_num = np.random.randint(len(subject['alphas']) + len(subject['LD']))
            # if par_num < len(subject['alphas']):
            #     subject['alphas'][par_num] *= np.random.rand() * 2
            # else:
            #     subject['alphas'][par_num - len(subject['alphas'])] *= np.random.rand() * 2

            # dodajanje novega proteina
            # TODO

            # spreminjanje tipa degradacije
            subject['deg_type'][np.random.randint(0, subject['proteins'])] = np.random.choice(3, p=PERTURBATION_WEIGHT_DEGRADATION)

            # degr_pos = np.random.rand()
            #
            # if degr_pos < PERTURBATION_WEIGHT_DEGRADATION[0]:
            #     subject['degradation'] = 'linear'
            # elif degr_pos < np.sum(PERTURBATION_WEIGHT_DEGRADATION[0:2]):
            #     subject['degradation'] = 'enzyme'
            #     subject['Km'] = np.random.randint(1, subject['Kd'])
            # else:
            #     subject['degradation'] = 'active'
            #     # TODO: Tudi verjetno se potrebno spremeniti parametre

            # spreminjanje tipa generiranja protein
            subject['type'][np.random.randint(0, subject['proteins'])] = np.random.choice(3, p=PERTURBATION_WEIGHT_TYPE)

            # spreminjanje tipa genske regulacije
            p = np.where(subject['type'] == 0, 1.0/np.count_nonzero(subject['type']), 0)
            idxr = np.random.choice(subject['proteins'], p=p)
            idxc = np.random.randint(0, subject['proteins'])

            oldKd = subject['M'][idxr, idxc]
            subject['M'][idxr, idxc] = np.random.choice([0, -oldKd, oldKd * np.random.rand() * 2])

            # odstranjevanje proteina
            idx = np.random.randint(0, subject['proteins'])
            subject['proteins'] -= 1
            subject['alphas'] = np.delete(subject['alphas'], idx)
            subject['betas'] = np.delete(subject['betas'], idx)
            subject['deltas'] = np.delete(subject['deltas'], idx)
            subject['type'] = np.delete(subject['type'], idx)
            subject['deg_type'] = np.delete(subject['deg_type'], idx)
            subject['Km'] = np.delete(subject['Km'], idx)
            subject['mod'] = np.delete(subject['mod'], idx)
            subject['mod'] = np.where(subject['mod'] >= idx, subject['mod']-1, subject['mod'])
            subject['M'] = np.delete(np.delete(subject['M'], idx, axis=0), idx, axis=1)

    return population

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


# Generate timestamps
t = np.arange(0, 100, dt)

# Dp ODE integration
tim = time.clock()
sub = initiate_subject()
print(sub)
r, info = integrate.odeint(generate_model(sub), np.random.randint(0, 10, size=sub['proteins']), t, args=(), full_output=True, printmessg=True)
print(time.clock() - tim)


print(r)
plt.plot(t, r)
plt.xlabel('Time')
plt.ylabel('Protein concetration')
plt.show()




print(generate_population(10, 3))

exit(0)


