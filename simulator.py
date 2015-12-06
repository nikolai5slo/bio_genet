#!/bin/python

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import time


from scipy import pi
import scipy.fftpack


PROTEIN_NUM = 3
T_MAX = 20

alpha = 100
delta = 1
Kd = 1
Km = 2
dt = 0.1

# Map of activators and repressors
#gmap = np.array([[0,-Kd,0],
#                 [0,0,-Kd],
#	  	          [-Kd,0,0]])

#initiatie semi-random map of activators
gmap = np.zeros((PROTEIN_NUM, PROTEIN_NUM))
idx = np.random.randint(1, PROTEIN_NUM, size=1)[0]
for i in range(gmap.shape[0]):
    gmap[idx,i] = -Kd
    idx += 1
    if idx >= PROTEIN_NUM:
        idx = 0

''' Universal model generator '''
def repressilator_model(p, t, M, degradation='linear'):
    dp = alpha * np.prod(np.where(M != 0, (0 <= np.sign(M)*(M + p)).astype(int), 1), axis=1)

    if degradation == 'linear':
        dp = dp - delta * p
    elif degradation == 'enzyme':
        dp = dp - delta * (p / (p + Km))

    return dp


# r = integrate.odeint(repressilator_model, [100, 0, 0], t)

# Generate timestamps
t = np.arange(0, 100, dt)

# Dp ODE integration
tim = time.clock()
r, info = integrate.odeint(repressilator_model, np.random.randint(0, 10, size=PROTEIN_NUM), t, args=(gmap,'linear'), full_output=True, printmessg=True)
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

