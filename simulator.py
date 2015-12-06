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

POPULATION_SIZE = 20
T_MAX = 20
dt = 0.1

def initiate_subject(num_proteins=5,alphas_type='scalar',deltas_type='scalar'):
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

    return {
        'Kd': Kd,
        'proteins': num_proteins,
        'alphas': np.random.random_sample() * ALPHA_MAX if (alphas_type == 'scalar') else np.random.random_sample(size=num_proteins) * ALPHA_MAX,
        'deltas': np.random.random_sample() if (deltas_type == 'scalar') else np.random.random_sample(size=num_proteins),
        'degradation': 'linear',
        'Km': None,
        'M': gmap
    }

''' Universal model generator '''
def repressilator_model(p, t, model):
    dp = model['alphas'] * np.prod(np.where(model['M'] != 0, (0 <= np.sign(model['M'])*(model['M'] + p)).astype(int), 1), axis=1)

    if model['degradation'] == 'linear':
        dp = dp - model['deltas'] * p
    elif model['degradation'] == 'enzyme':
        dp = dp - model['deltas'] * (p / (p + model['Km']))

    return dp


# Generate timestamps
t = np.arange(0, 100, dt)

# Dp ODE integration
tim = time.clock()
sub = initiate_subject()
print(sub)
r, info = integrate.odeint(repressilator_model, np.random.randint(0, 10, size=sub['proteins']), t, args=(sub,), full_output=True, printmessg=True)
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

