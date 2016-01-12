import numpy as np
import copy

from config import *
from population import *

"""
    Izvajanje perturbacij nad populacijo (selekcija in mutacija)
        :param population
            Populacija
        :param evals
            Rezultat evalvacije osebkov

        :return
            Nova populacija za naslednjo generacijo

"""
def perturbate(population, evals):
    pop_len = len(population)
    best_sub_num = int(pop_len * PERTURBATION_POPULATION_PERCENTAGE)

    # Izberi najboljse osebke
    newPop = [population[evals[i][0]] for i in range(0, best_sub_num)]

    # Dodajanje osebkov za mutiranje
    pop_idx = 0
    for i in range(best_sub_num, pop_len):
        newPop.append(copy_subject(population[evals[pop_idx][0]]))
        pop_idx += 1
        if pop_idx >= best_sub_num:
            pop_idx = 0

    # Izvedi mutacije nad osebki
    for i in range(best_sub_num, pop_len):
        subject = newPop[i]

        # Izbira tipa permutacije
        permutation_type = np.random.choice(6, p=PERTURBATION_PERMUTATION_WEIGHTS)

        # Spreminjanje kineticnih parametrov
        if permutation_type == 0:
            a_map = np.random.choice(2, subject['proteins'], p=[0.9, 0.1])
            subject['alphas'] = np.where(a_map > 0, subject['alphas'] * np.random.rand() * 2, subject['alphas'])
            subject['alphas'] = np.where(subject['alphas'] > ALPHA_MAX, ALPHA_MAX * np.random.rand(), subject['alphas'])

            b_map = np.random.choice(2, subject['proteins'], p=[0.9, 0.1])
            subject['betas'] = np.where(b_map > 0, subject['betas'] * np.random.rand() * 2, subject['betas'])
            subject['betas'] = np.where(subject['betas'] > BETA_MAX, BETA_MAX * np.random.rand(), subject['betas'])

            d_map = np.random.choice(2, subject['proteins'], p=[0.9, 0.1])
            subject['deltas'] = np.where(d_map > 0, subject['deltas'] * np.random.rand() * 2, subject['deltas'])
            subject['deltas'] = np.where(subject['deltas'] > DELTA_MAX, DELTA_MAX * np.random.rand(), subject['deltas'])

            km_map = np.random.choice(2, subject['proteins'], p=[0.9, 0.1])
            subject['Km'] = np.where(km_map > 0, subject['Km'] * np.random.rand() * 2, subject['Km'])
            subject['Km'] = np.where(subject['Km'] > KM_MAX, KM_MAX * np.random.rand(), subject['Km'])

            kd_map = np.random.choice(2, subject['proteins'], p=[0.6, 0.4])
            subject['Kd'] = np.where(kd_map > 0, subject['Kd'] * np.random.rand() * 2, subject['Kd'])
            subject['Kd'] = np.where(subject['Kd'] > KD_MAX, np.random.rand() * KD_MAX, subject['Kd'])

        # Dodajanje proteina
        elif permutation_type == 1 and subject['proteins'] < PROTEIN_NUM_MAX:
            subject['proteins'] += 1

            subject['M'] = np.vstack((subject['M'], np.random.randint(-1, 2, size=subject['proteins'] - 1)))
            subject['M'] = np.hstack((subject['M'], np.random.randint(-1, 2, size=(subject['proteins'], 1))))

            subject['alphas'] = np.append(subject['alphas'], np.random.random() * ALPHA_MAX)
            subject['betas'] = np.append(subject['betas'], np.random.random() * BETA_MAX)
            subject['deltas'] = np.append(subject['deltas'], np.random.random() * DELTA_MAX)
            subject['type'] = np.append(subject['type'], np.random.randint(0, 3))
            subject['deg_type'] = np.append(subject['deg_type'], np.random.randint(0, 3))
            subject['Kd'] = np.append(subject['Kd'], np.random.random() * KD_MAX)
            subject['Km'] = np.append(subject['Km'], np.random.random() * KM_MAX)
            subject['mod'] = np.append(subject['mod'], np.random.randint(0, subject['proteins'] - 1))
            subject['init'] = np.append(subject['init'], np.random.rand() * KD_MAX)

        # Spreminjanje tipa degradacije
        elif permutation_type == 2:
            subject['deg_type'][np.random.randint(0, subject['proteins'])] = np.random.choice(3, p=PERTURBATION_WEIGHT_DEGRADATION)

        # Spreminjanje tipa generiranja proteina
        elif permutation_type == 3:
            subject['type'][np.random.randint(0, subject['proteins'])] = np.random.choice(3, p=PERTURBATION_WEIGHT_TYPE)

        # Spreminjanje tipa genskega izrazanja
        elif permutation_type == 4:
            num_zero = (len(subject['type']) - np.count_nonzero(subject['type']))
            if num_zero > 0:
                p = np.where(subject['type'] == 0, 1.0/num_zero, 0)
                idxr = np.random.choice(subject['proteins'], p=p)
                idxc = np.random.randint(0, subject['proteins'])

                oldKd = subject['M'][idxr, idxc]
                subject['M'][idxr, idxc] = np.random.choice([0, -oldKd, oldKd * np.random.rand() * 2])

        # Odstranjevanje proteina
        elif permutation_type == 5 and subject['proteins'] > 2:
            idx = np.random.randint(0, subject['proteins'])
            subject['proteins'] -= 1
            subject['alphas'] = np.delete(subject['alphas'], idx)
            subject['betas'] = np.delete(subject['betas'], idx)
            subject['deltas'] = np.delete(subject['deltas'], idx)
            subject['type'] = np.delete(subject['type'], idx)
            subject['deg_type'] = np.delete(subject['deg_type'], idx)
            subject['Km'] = np.delete(subject['Km'], idx)
            subject['Kd'] = np.delete(subject['Kd'], idx)
            subject['init'] = np.delete(subject['init'], idx)

            subject['mod'] = np.delete(subject['mod'], idx)
            subject['mod'] = np.where(subject['mod'] >= idx, subject['mod']-1, subject['mod'])

            subject['M'] = np.delete(np.delete(subject['M'], idx, axis=0), idx, axis=1)

    return newPop