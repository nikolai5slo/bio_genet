#!/bin/python

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import time


from scipy import pi
import scipy.fftpack


alpha = 100
Kd = 12
delta = 1

dt = 0.01

# Map of activators and repressors
gmap = np.array([[0, -Kd, 0],
                 [0, 0, -Kd],
                 [-Kd, 0, 0]])

# Alphas and deltas
alphas = np.array([alpha, alpha, alpha])
deltas = np.array([delta, delta, delta])

''' Universal model generator '''


def generate_model(mat, alphas, deltas):
    mat = np.transpose(mat)

    # Prepare matrixes for computation
    def repr_model(p, t):
        return alphas * np.prod(np.where(mat != 0, (0 <= np.sign(mat) * (mat + p)).astype(int), 1), axis=1) - (
        deltas * p)

    return repr_model


''' Reprissilator model '''


def repressilator_model(P, t):
    P1 = [0, 0, 0]
    P1[0] = alpha * int(0 <= (Kd - P[2])) - delta * P[0]
    P1[1] = alpha * int(0 <= (Kd - P[0])) - delta * P[1]
    P1[2] = alpha * int(0 <= (Kd - P[1])) - delta * P[2]
    return P1


# r = integrate.odeint(repressilator_model, [100, 0, 0], t)

# Generate timestamps
t = np.arange(0, 100, dt)

# Dp ODE integration
tim = time.clock()
r = integrate.odeint(generate_model(gmap, alphas, deltas), np.array([100, 0, 0]), t)
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

