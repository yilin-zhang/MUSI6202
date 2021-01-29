import os
import sys
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def loadSoundFile(filename):
    [samplerate, x] = wavread(filename)

    # take only the left channel if the file is not mono
    if x.shape[1] > 1:
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


def crossCorr(x, y):
    return np.correlate(x, y, 'full')


def findSnarePosition(snareFilename, drumloopFilename):
    loop_audio = loadSoundFile(drumloopFilename)
    snare_audio = loadSoundFile(snareFilename)
    corr = crossCorr(loop_audio, snare_audio)

    # plot the correlation
    # (putting this part inside this function avoids loading the files twice)
    dirname = os.path.dirname(sys.argv[0])
    results_path = os.path.join(dirname, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    plt.figure()
    plt.plot(range(1-len(snare_audio), len(loop_audio)), corr)  # shift the horizontal axis
    plt.savefig(os.path.join(results_path, '01-correlation.png'))

    # find the peaks (remove the points where lag < 0)
    peaks, _ = find_peaks(corr[len(snare_audio)-1:], height=100)
    peaks.tofile(os.path.join(results_path, '02-snareLocation.txt'), sep=',')


if __name__ == '__main__':
    dirname = os.path.dirname(sys.argv[0])
    loop_path = os.path.join(dirname, 'wav', 'drum_loop.wav')
    snare_path = os.path.join(dirname, 'wav', 'snare.wav')
    findSnarePosition(snare_path, loop_path)
