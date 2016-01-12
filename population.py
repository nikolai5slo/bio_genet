import numpy as np

from config import *

def initiate_subject(num_proteins=None):
    if num_proteins == None:
        num_proteins = np.random.randint(2, PROTEIN_NUM_MAX+1)

    sub = {
        'init': np.random.rand(num_proteins) * KD_MAX,

        'proteins': num_proteins,
        'alphas': np.random.random_sample(size=num_proteins) * ALPHA_MAX,
        #'M': np.random.randint(-1, 2, size=(num_proteins, num_proteins)),
        'M': generate_repres_topology(num_proteins),
        'type': np.random.randint(0, 3, size=num_proteins),# 0 - gensko izrazanje, 1 - linearna modifikacija, 2 - encimska modifikacija
        #'Kd': np.random.random_sample(size=(num_proteins, num_proteins)) * KD_MAX,
        'Kd': np.random.random_sample(num_proteins) * KD_MAX,

        # Degradation
        'deg_type': np.random.randint(0, 3, size=num_proteins), # 0 - linearna deg., 1 - aktivna deg., 2 - encimska deg.
        'deltas': np.random.rand(num_proteins),
        'Km': np.random.rand(num_proteins) * KM_MAX, # Vektorja, za encimsko degradacijo. Prvi stolpec je delta, drugi Km

        # Modification
        'betas': np.random.rand(num_proteins),
        'mod': np.random.randint(0, num_proteins, size=num_proteins)
    }

    if M_SETUP_OSCILATE:
        gmap = np.zeros((sub['proteins'], sub['proteins']))
        idx = np.random.randint(1, sub['proteins'], size=1)[0]
        for i in range(gmap.shape[0]):
            gmap[idx,i] = -1
            idx += 1
            if idx >= sub['proteins']:
                idx = 0
        sub['M'] = gmap
        #sub['Kd'] = np.array([[KD_MAX * 0.5 for _ in range(num_proteins)] for _ in range(num_proteins)])
        sub['alphas'] = np.array([ALPHA_MAX for _ in range(num_proteins)])
        sub['deltas'] = np.array([DELTA_MAX for _ in range(num_proteins)])
        sub['Kd'] = np.array([KD_MAX for _ in range(num_proteins)])
        sub['type'] = np.zeros(num_proteins)
        sub['deg_type'] = np.zeros(num_proteins)
        sub['betas'] = np.zeros(num_proteins)

    return sub

def copy_subject(sub):
    newSub = {
        'init': np.copy(sub['init']),

        'proteins': sub['proteins'],
        'alphas': np.copy(sub['alphas']),
        'M': np.copy(sub['M']),
        'type': np.copy(sub['type']),
        'Kd': np.copy(sub['Kd']),

        # Degradation
        'deg_type': np.copy(sub['deg_type']),
        'deltas': np.copy(sub['deltas']),
        'Km': np.copy(sub['Km']),

        # Modification
        'betas': np.copy(sub['betas']),
        'mod': np.copy(sub['mod'])
    }

    return newSub

def generate_population(size, num_proteins=None):
    return [initiate_subject(num_proteins) for _ in range(size)]

def generate_repres_topology(num_prot):
    topology = np.zeros((num_prot,num_prot))
    idx = np.random.randint(0, num_prot);
    for i in range(topology.shape[0]):
        topology[idx,i] = -1
        idx+=1
        if idx >= num_prot:
            idx=0
    return topology
