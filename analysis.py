import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt

def rms(singals):
    rmss = []
    for sig in singals:
        ampl = 0
        for j in sig:
            ampl+=np.power(j,2)
        print(ampl)
        rmss.append(np.sqrt(ampl/len(sig)))
    return rmss

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
"""
t = scipy.linspace(0,120,1000)
acc = lambda t: 10*scipy.sin(2*pi*2.0*t)
signal = acc(t)
sigFFT(signal)

"""
