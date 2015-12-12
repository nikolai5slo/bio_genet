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
ALPHA_MAX = 100

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
        'degradation': degradation,
        'deltas': np.random.rand(num_proteins),
        'AD' : ad, # Matrika delt, za aktinvo degradacijo. Po diagonali so 0 ker ne more vplivati sam nase
        'ED' : np.random.rand(2, num_proteins), # Vektorja, za encimsko degradacijo. Prvi stolpec je delta, drugi Km
        'LM' : np.vstack((np.random.randint(-10, num_proteins, size=(num_proteins)), np.random.rand(num_proteins)))
    }

def generate_population(size, num_proteins, degradation_weights=np.array([0.6,0.3,0.1])):
    subjects = []

    #zaenkrat se po utezeh dodaja le degradacija
    #verjetno bo treba se kaj spreminjat

    for i in range(size):
        degr_pos = np.random.rand()

        if degr_pos < degradation_weights[0]:
            degradation_type = 'linear'
        elif degr_pos < np.sum(degradation_weights[0:2]):
            degradation_type = 'enzyme'
        else:
            degradation_type = 'active'

        subjects.append(initiate_subject(num_proteins, degradation=degradation_type))

    return subjects

''' Universal model generator '''
def generate_model(model):

    lm_vec = model['LM'].T
    lm_map = model['LM'][0,:] >= 0
    lm_mat = np.zeros((model['proteins'], model['proteins']))
    for i, v in enumerate(lm_vec):
        if(v[0] >= 0):
            lm_mat[i, i] = -v[1]
            lm_mat[v[0], i] = v[1]
            lm_map[v[0]] = 1


        def repressilator_model(p, t):
            # Genska represija
            dg = model['alphas'] * np.prod(np.where(model['M'] != 0, (0 <= np.where(model['M'] > 0, p - model['M'], -model['M'] - p )).astype(int), 1), axis=1)

            # Modifikacija
            dm = np.dot(lm_mat, p)
            dp = np.where(lm_map >= 0, dm, dg)

            # Degradacija
            if model['degradation'] == 'active':
                dp = np.dot(model['AD'], dp) * dp # Aktivna degradacija
            elif model['degradation'] == 'linear':
                dp = -model['deltas'] * dp # Linearna degradacija
            elif model['degradation'] == 'enzyme':
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

