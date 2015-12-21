import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import peakutils
import findpeaks as fp


def rms(singals):
    rmss = []
    for sig in singals:
        ampl = 0
        for j in sig:
            ampl += np.power(j, 2)
        print(ampl)
        rmss.append(np.sqrt(ampl / len(sig)))
    return rmss


def measureOsc(sig, timeStamps, threshold):
    damped = 0
    indexes = fp.detect_peaks(sig, mph=None, mpd=1, threshold=threshold, edge='rising', kpsh=False, valley=False,
                              show=False, ax=None)
    peaks = sig[indexes]
    if len(peaks) >= 2:
        threshold2 = 0.1 * peaks[np.ceil(peaks[-1] / 2)]
        indexes2 = fp.detect_peaks(sig, mph=None, mpd=1, threshold=threshold2, edge='rising', kpsh=False, valley=False,
                                   show=False, ax=None)
        peaks2 = sig[indexes2]
        if len(peaks2) < 2:
            damped = 1

    amplitude = 0
    period = 0
    oscillatory = 0
    frequency = 0

    if len(peaks) >= 2:
        amplitude = sig[indexes[-2]] - min(sig[indexes[-2]:indexes[-1]])
        period = timeStamps[indexes[-1]] - timeStamps[indexes[-2]]
        if timeStamps[indexes[-1]] < timeStamps[-1] - 1.5 * period:
            amplitude = 0
            period = 0
            damped = 1
        else:
            frequency = 1.0 / period
            oscillatory = 1;

    fig, ax = plt.subplots()
    ax.plot(sig)
    ax.plot(indexes, sig[indexes], 'ro')
    ymin, ymax = x[np.isfinite(sig)].min(), x[np.isfinite(sig)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)

    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.show()

    return {
        'osc': oscillatory,
        'freq': frequency,
        'per': period,
        'ampl': amplitude
    }


def sigFFT(signal, t):
    FFT = abs(scipy.fft(signal))
    freqs = scipy.fftpack.fftfreq(signal.size, t[1] - t[0])

    plt.subplot(211)
    plt.plot(t, signal)
    plt.subplot(212)
    plt.plot(freqs[0:500], FFT[0:500], 'x')
    # plt.plot(freqs,20*scipy.log10(FFT),'x')
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

# t = scipy.linspace(0,120,1000)
# acc = lambda t: 10*scipy.sin(2*pi*2.0*t)
# signal = acc(t)
# sigFFT(signal)

dt = 0.1
t = np.arange(0, 100, dt)
x = np.sin(2 * 3.14 * 300 * t)
a=measureOsc(x, t, 0)
print(a)
centers = (30.5, 72.3)
x = np.linspace(0, 120, 121)
y = (peakutils.gaussian(x, 5, centers[0], 3) +
    peakutils.gaussian(x, 7, centers[1], 10) +
    np.random.rand(x.size))


