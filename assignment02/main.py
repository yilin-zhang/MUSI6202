from typing import Tuple
import time
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread


def myTimeConv(x: np.array, h: np.array) -> np.array:
    # allocate memory for the convolution result
    x_size = x.size
    h_size = h.size
    y_size = x_size + h_size - 1
    y = np.zeros(y_size)

    # Sum_m x[m] * h[n-m]
    for n in range(y_size):
        m_min = max(0, n - (h_size - 1))
        m_max = min(x_size-1, n)
        sum_for_n = 0
        for m in range(m_min, m_max+1):
            sum_for_n += x[m] * h[n-m]
        y[n] = sum_for_n

    return y


# If the length of 'x' is 200 and the length of 'h' is 100, what is the length of 'y' ?
# 200 + 100 - 1 = 299
def CompareConv(x: np.array, h: np.array) -> Tuple[float, float, float, np.array]:
    start_time = time.time()
    custom_conv = myTimeConv(x, h)
    custom_conv_time = time.time()
    scipy_conv = convolve(x, h)
    scipy_conv_time = time.time()

    m = np.mean(custom_conv - scipy_conv)
    mabs = np.mean(np.absolute(custom_conv - scipy_conv))
    stdev = np.std(custom_conv - scipy_conv)
    t = np.zeros(2)
    t[0] = custom_conv_time - start_time
    t[1] = scipy_conv_time - custom_conv_time

    return m, mabs, stdev, t


def loadSoundFile(filename):
    [samplerate, x] = wavread(filename)

    # take only the left channel if the file is not mono
    if len(x.shape) > 1 and x.shape[1] > 1:
        x = x[:, 0]

    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2**(nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    return audio


if __name__ == '__main__':
    # Q1
    x = np.ones(200)
    h = np.zeros(51)
    for i in range(51):
        h[i] = 1. - abs(25. - i) / 25.

    y_time = myTimeConv(x, h)
    plt.xlabel('n')
    plt.ylabel('y')
    plt.title('Question 1: Time Domain Convolution')
    plt.plot(y_time)
    plt.savefig('results/q1.png')

    # Q2
    x = loadSoundFile('wav/piano.wav')
    h = loadSoundFile('wav/impulse-response.wav')
    m, mabs, stdev, t = CompareConv(x, h)

    with open('results/q2.txt', 'w') as f:
        f.write(f'm: {m}\n'
                f'mabs: {mabs}\n'
                f'stdev: {stdev}\n'
                f'time of custom conv: {t[0]} s\n'
                f'time of scipy conv: {t[1]} s\n')
