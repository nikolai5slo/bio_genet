import numpy as np

from config import *

def initiate_subject(num_proteins=None):
    if num_proteins == None:
        num_proteins = np.random.randint(1, PROTEIN_NUM_MAX+1)

    return {
        'proteins': num_proteins,
        'alphas': np.random.random_sample(size=num_proteins) * ALPHA_MAX,
        'M': np.random.randint(-1, 2, size=(num_proteins, num_proteins)),
        'type': np.random.randint(0, 3, size=num_proteins),# 0 - gensko izrazanje, 1 - linearna modifikacija, 2 - encimska modifikacija

        # Degradation
        'deg_type': np.random.randint(0, 3, size=num_proteins), # 0 - linearna deg., 1 - aktivna deg., 2 - encimska deg.
        'deltas': np.random.rand(num_proteins),
        'Km': np.random.rand(num_proteins), # Vektorja, za encimsko degradacijo. Prvi stolpec je delta, drugi Km

        # Modification
        'betas': np.random.rand(num_proteins),
        'mod': np.random.randint(0, num_proteins, size=num_proteins)
    }

def copy_subject(sub):
    newSub = {
        'proteins': sub['proteins'],
        'alphas': np.copy(sub['alphas']),
        'M': np.copy(sub['M']),
        'type': np.copy(sub['type']),

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

