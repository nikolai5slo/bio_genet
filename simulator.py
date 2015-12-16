#!/bin/python

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import pi
import scipy.fftpack

PROTEIN_NUM_MAX = 5
KD_MAX = 100
PROTEINS_MAX = 3
ALPHA_MAX = 1

DEGRADATION_WEIGHTS = np.array([0.6,0.3,0.1])

PERTURBAIION_PROBABILITY = 0.1
PERTURBATION_WEIGHT_DEGRADATION = np.array([0.6, 0.2, 0.2])

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
        gmap[idx,i] = -Kd
        idx += 1
        if idx >= num_proteins:
            idx = 0

    ad = -np.random.rand(num_proteins, num_proteins)
    np.fill_diagonal(ad, 0)

    return {
        'Kd': Kd,
        'proteins': num_proteins,
        'alphas': np.random.random_sample(size=num_proteins) * ALPHA_MAX,
        'Km': np.random.randint(1, Kd),
        'M': gmap,
        #'M': np.zeros((num_proteins,num_proteins)),

        # Degradation
        'LD': np.random.rand(num_proteins),
        'AD' : ad, # Matrika delt, za aktinvo degradacijo. Po diagonali so 0 ker ne more vplivati sam nase
        'ED' : np.random.rand(2, num_proteins), # Vektorja, za encimsko degradacijo. Prvi stolpec je delta, drugi Km
        'LM' : np.vstack((np.random.randint(-10, num_proteins, size=(num_proteins)), np.random.rand(num_proteins))),
        'EM' : np.vstack((np.random.randint(-10, num_proteins, size=(num_proteins)), np.random.rand(num_proteins),np.random.rand(num_proteins))) #ENCIMSKA MODIFIKACIJA 3 vrstice, 1. na katerega vpliva, 2. beta, 3. Km
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
            par_num = np.random.randint(len(subject['alphas']) + len(subject['LD']))
            if par_num < len(subject['alphas']):
                subject['alphas'][par_num] *= np.random.rand() * 2
            else:
                subject['alphas'][par_num - len(subject['alphas'])] *= np.random.rand() * 2

            # dodajanje novega proteina
            # TODO

            # spreminjanje tipa degradacije
            degr_pos = np.random.rand()

            if degr_pos < PERTURBATION_WEIGHT_DEGRADATION[0]:
                subject['degradation'] = 'linear'
            elif degr_pos < np.sum(PERTURBATION_WEIGHT_DEGRADATION[0:2]):
                subject['degradation'] = 'enzyme'
                subject['Km'] = np.random.randint(1, subject['Kd'])
            else:
                subject['degradation'] = 'active'
                # TODO: Tudi verjetno se potrebno spremeniti parametre

            # spreminjanje tipa generiranja protein
            # TODO

            # spreminjanje tipa genske regulacije
            # TODO

            # odstranjevanje proteina
            # TODO

    return population

''' Universal model generator '''
def generate_model(model):
    #sestava matrike in maske za linearno modifikacijo
    lm_vec = model['LM'].T
    lm_map = model['LM'][0,:] >= 0
    lm_mat = np.zeros((model['proteins'], model['proteins']))
    for i, v in enumerate(lm_vec):
        if(v[0] >= 0):
            lm_mat[i, i] = -v[1]
            lm_mat[v[0], i] = v[1]
            lm_map[v[0]] = True
            lm_map[i] = True

    #sestava matrike in maske za encimsko modifikacijo
    em_vec = model['EM'].T
    em_map = model['EM'][0,:] >= 0
    em_mat = np.zeros((model['proteins'], model['proteins']))
    for i, v in enumerate(em_vec):
        if(v[0] >= 0):
            em_mat[i, i] = -v[1]
            em_mat[v[0], i] = v[1]
            em_map[v[0]] = True
            em_map[i] = True

    def repressilator_model(p, t):
        # Genska represija
        dg = model['alphas'] * np.prod(np.where(model['M'] != 0, (0 <= np.where(model['M'] > 0, p - model['M'], -model['M'] - p )).astype(int), 1), axis=1)

        # Modifikacija
        dm = np.dot(lm_mat, p)
        dp = np.where(lm_map, dm, dg) # linearna

        #Encimska modifikacija
        dem_brez = p * (p / (model['EM'][2,:] + p)) # Encimska modifikacija pred mnozejem z beto
        dem = np.dot(em_mat, dem_brez)
        dp = np.where(em_map, dem, dg)

        # Degradacija
        dp = np.dot(model['AD'], dp) * dp # Aktivna degradacija
        dp = -model['LD'] * dp # Linearna degradacija
        dp = -model['ED'][0,:] * (dp / (model['ED'][1,:] + dp)) # Encimska degradacija

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

"""
FFT = abs(scipy.fft(r[:,0]))
freqs = scipy.fftpack.fftfreq(r[:,0].size, t[1]-t[0])

plt.subplot(211)
plt.plot(t, r[:,0])
plt.subplot(212)
plt.plot(freqs,20*scipy.log10(FFT),'x')
plt.show()
"""
"""xF = np.fft.fft(r[:,0])
N = len(xF)
xF = xF[0:N/2]
fr = np.linspace(0,100/2,N/2)
"""

print(generate_population(10, 3))

exit(0)

def sigFFT(signal):
    FFT = abs(scipy.fft(signal))
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])

    plt.subplot(211)
    plt.plot(t, signal)
    plt.subplot(212)
    plt.plot(freqs[0:500],FFT[0:500],'x')
    #plt.plot(freqs,20*scipy.log10(FFT),'x')
    plt.show()

    threshold = 0.5 * max(FFT[0:500])
    mask = FFT > threshold
    peaks = freqs[mask]
    print(peaks[0])

t = scipy.linspace(0,120,1000)
acc = lambda t: 10*scipy.sin(2*pi*2.0*t)
signal = acc(t)
sigFFT(signal)

