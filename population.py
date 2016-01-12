import numpy as np

from config import *

def initiate_subject(num_proteins=None):
    """
    Zacetna inicializacija osebka
    :param num_proteins: Stevilo proteinov
    :return: Nov osebek
    """
    if num_proteins == None:
        num_proteins = np.random.randint(2, PROTEIN_NUM_MAX+1)

    sub = {
        'init': np.random.rand(num_proteins) * KD_MAX,                              # Zacetna koncentracija proteinov pri simulaciji

        'proteins': num_proteins,                                                   # Stevilo proteinov v omrezju
        'alphas': np.random.random_sample(size=num_proteins) * ALPHA_MAX,           # Kineticni parametri alfa, ki se uporabljajo pri genski ekspresiji
        'M': np.random.randint(-1, 2, size=(num_proteins, num_proteins)),           # Matrika omrezja
        'type': np.random.randint(0, 3, size=num_proteins),                         # Tip akcije:
                                                                                        # 0 - gensko izrazanje
                                                                                        # 1 - linearna modifikacija
                                                                                        # 2 - encimska modifikacija

        'Kd': np.random.random_sample(num_proteins) * KD_MAX,                       # Kineticni parameteri

        # Parametri degradacij
        'deg_type': np.random.randint(0, 3, size=num_proteins),                     # Tip degradacije:
                                                                                        # 0 - linearna degradacija
                                                                                        # 1 - aktivna degradacija
                                                                                        # 2 - encimska degradacija
        'deltas': np.random.rand(num_proteins),                                     # Kineticni parametri delta, ki se uporabljajo pri degradacijah
        'Km': np.random.rand(num_proteins) * KM_MAX,                                # Vektorja, za encimsko degradacijo. Prvi stolpec je delta, drugi Km

        # Parametri genske modifikacije
        'betas': np.random.rand(num_proteins),                                      # Kineticni parametri beta, ki se uporabljajo pri genski modifikaciji
        'mod': np.random.randint(0, num_proteins, size=num_proteins)                # Parameter kazalcev na katerega vplivajo proteini pri genski modifikaciji
    }

    if M_SETUP_OSCILATE:
        """ V naprej pripravimo omrezje ki oscilira """
        gmap = np.zeros((sub['proteins'], sub['proteins']))
        idx = np.random.randint(1, sub['proteins'], size=1)[0]
        for i in range(gmap.shape[0]):
            gmap[idx,i] = -1
            idx += 1
            if idx >= sub['proteins']:
                idx = 0
        sub['M'] = gmap
        sub['alphas'] = np.array([ALPHA_MAX for _ in range(num_proteins)])
        sub['deltas'] = np.array([DELTA_MAX for _ in range(num_proteins)])
        sub['Kd'] = np.array([KD_MAX for _ in range(num_proteins)])
        sub['type'] = np.zeros(num_proteins)
        sub['deg_type'] = np.zeros(num_proteins)
        sub['betas'] = np.zeros(num_proteins)

    return sub

def copy_subject(sub):
    """
    Kopiranje osebka
    :param sub: Osebek ki ga kopiramo
    :return: Nov kopiran osebek
    """
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
    """
    Generiranje populacije osebkov
    :param size: Stevilo osebkov
    :param num_proteins: Stevilo proteinov znotraj vsakega osebka
    :return: Polje osebkov, ki predstavlja populacijo
    """
    return [initiate_subject(num_proteins) for _ in range(size)]

