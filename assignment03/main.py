from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


# Question 1
def generateSinusoidal(amplitude: float, sampling_rate_Hz: int, frequency_Hz: float,
                       length_secs: float, phase_radians: float) -> Tuple[np.array, np.array]:
    num_samples = round(length_secs * sampling_rate_Hz)
    steps = np.arange(0, num_samples, dtype=np.int64)
    audio_array = amplitude * np.sin(2 * np.pi * frequency_Hz * steps / sampling_rate_Hz + phase_radians)
    time_array = steps / sampling_rate_Hz

    return audio_array, time_array


# Question 2
def generateSquare(amplitude: float, sampling_rate_Hz: int, frequency_Hz: float,
                   length_secs: float, phase_radians: float) -> Tuple[np.array, np.array]:
    # x(t) = 4/pi * ( sin(wt) + (1/3) * sin(3wt) + (1/5) * sin(5wt) + ...)
    harmonics = []
    time = np.array([])

    for i in range(10):
        nth = i * 2 + 1  # the nth harmonic
        harmonic, time = generateSinusoidal(amplitude * 4/np.pi * 1/nth,
                                            sampling_rate_Hz, nth * frequency_Hz,
                                            length_secs, nth * phase_radians)
        harmonics.append(harmonic)

    return sum(harmonics), time


def computeSpectrum(x: np.array, sample_rate_Hz: int) -> \
        Tuple[np.array, np.array, np.array, np.array, np.array]:
    signal_len = len(x)
    x_fft = np.fft.fft(x) / signal_len

    # calculate the length of the non-repeated fft
    if signal_len % 2 == 0:
        # the length is even
        fft_len = signal_len // 2 + 1
    else:
        # the length is odd
        fft_len = (signal_len + 1) // 2

    half_fft = x_fft[:fft_len]
    f = np.linspace(0, sample_rate_Hz/2, num=fft_len, endpoint=False)

    XAbs = abs(half_fft)
    XPhase = np.angle(half_fft)
    XRe = half_fft.real
    XIm = half_fft.imag

    return f, XAbs, XPhase, XRe, XIm


def generateBlocks(x: np.array, sample_rate_Hz: int,
                   block_size: int, hop_size: int) -> Tuple[np.array, np.array]:
    index = 0
    signal_len = len(x)
    blocks = []
    time_stamps = []
    while True:
        time_stamps.append(1 / sample_rate_Hz * index)
        if index + block_size >= signal_len:
            # treat it as the last block
            if index + block_size != signal_len:
                # requires zero padding
                blocks.append(np.concatenate((x[index: signal_len],
                                              np.zeros(block_size - (signal_len - index)))))
            else:
                # no need to pad zero
                blocks.append(x[index: index+block_size])
            break
        blocks.append(x[index: index+block_size])
        index += hop_size

    t = np.array(time_stamps)
    X = np.transpose(np.array(blocks))

    return t, X


def mySpecgram(x: np.array, block_size: int, hop_size: int, sampling_rate_Hz: int, window_type: str) \
        -> Tuple[np.array, np.array, np.array]:
    time_vector, blocks = generateBlocks(x, sampling_rate_Hz, block_size, hop_size)

    if window_type == 'hann':
        win = np.hanning(block_size).reshape(block_size, 1)
        blocks = blocks * win

    magnitude_spectrogram = abs(np.fft.rfft(blocks, axis=0) / block_size)
    freq_vector = np.linspace(0, sampling_rate_Hz/2, num=magnitude_spectrogram.shape[0], endpoint=False)

    plt.figure()
    plt.title(f'Q4: spectrogram ({window_type} window)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.pcolormesh(time_vector, freq_vector, magnitude_spectrogram, shading='nearest')
    plt.savefig(f'results/q4-{window_type}.png')
    return freq_vector, time_vector, magnitude_spectrogram


if __name__ == '__main__':
    # Question 1
    sin_audio, sin_time = generateSinusoidal(1.0, 44100, 400, 0.5, np.pi/2)
    num_samples_for_plot = round(44100 * 0.005)  # 5ms
    sin_audio_for_plot = sin_audio[:num_samples_for_plot]
    sin_time_for_plot = sin_time[:num_samples_for_plot]
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Question 1: Generating Sinusoids')
    plt.plot(sin_time_for_plot, sin_audio_for_plot)
    plt.savefig('results/q1.png')

    # Question 2
    sqr_audio, sqr_time = generateSquare(1.0, 44100, 400, 0.5, 0)
    num_samples_for_plot = round(44100 * 0.005)  # 5ms
    sqr_audio_for_plot = sqr_audio[:num_samples_for_plot]
    sqr_time_for_plot = sqr_time[:num_samples_for_plot]
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Question 2: Combining Sinusoids')
    plt.plot(sqr_time_for_plot, sqr_audio_for_plot)
    plt.savefig('results/q2.png')

    # Question 3
    sin_f, sin_spec_mag, sin_spec_phase, _, _ = computeSpectrum(sin_audio, 44100)
    plt.figure()
    plt.subplot(211)
    plt.title('Question 3: Sin Wave Magnitude and Phase')
    plt.ylabel('Magnitude')
    plt.plot(sin_f, sin_spec_mag)

    plt.subplot(212)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    plt.plot(sin_f, sin_spec_phase)

    plt.savefig('results/q3-sin.png')

    sqr_f, sqr_spec_mag, sqr_spec_phase, _, _ = computeSpectrum(sqr_audio, 44100)
    plt.figure()
    plt.subplot(211)
    plt.title('Question 3: Square Wave Magnitude and Phase')
    plt.ylabel('Magnitude')
    plt.plot(sqr_f, sqr_spec_mag)

    plt.subplot(212)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    plt.plot(sqr_f, sqr_spec_phase)

    plt.savefig('results/q3-sqr.png')

    # Question 4
    mySpecgram(sqr_audio, 2048, 1024, 44100, 'rect')
    mySpecgram(sqr_audio, 2048, 1024, 44100, 'hann')