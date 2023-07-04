import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch, math
from scipy.linalg import hadamard

rng = np.random.default_rng()

import generate_dataset.QAM_codebook as cb

def raised_root_cosine(upsample, num_positive_lobes, alpha):
    N = upsample * (num_positive_lobes * 2 + 1)
    t = (np.arange(N) - N / 2) / upsample
    # result vector
    h_rrc = np.zeros(t.size, dtype=np.float)
    # index for special cases
    sample_i = np.zeros(t.size, dtype=np.bool)
    # deal with special cases
    subi = t == 0
    sample_i = np.bitwise_or(sample_i, subi)
    h_rrc[subi] = 1.0 - alpha + (4 * alpha / np.pi)
    subi = np.abs(t) == 1 / (4 * alpha)
    sample_i = np.bitwise_or(sample_i, subi)
    h_rrc[subi] = (alpha / np.sqrt(2)) \
                * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
                + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
    # base case
    sample_i = np.bitwise_not(sample_i)
    ti = t[sample_i]
    h_rrc[sample_i] = np.sin(np.pi * ti * (1 - alpha)) \
                    + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
    h_rrc[sample_i] /= (np.pi * ti * (1 - (4 * alpha * ti) ** 2))

    return h_rrc

def QAM(qam, fs=1, T_bits=5, cf=0.0, ml=16384, span=1, alpha=1.0):
    
    Ns = fs * T_bits
    N = ml * Ns
    code = cb.modulation[qam]
    bit_seq = rng.integers(0, qam, ml)

    M = []
    for i in bit_seq:
        M.append(code[i])
    M = np.asarray(M).ravel()

    zero_mat = np.zeros((int(Ns)-1, ml), dtype=M.dtype)
    sym_seq = np.concatenate((M[np.newaxis, :], zero_mat), axis=0)
    sym_seq = np.reshape(sym_seq, (int(N),), order='F')

    # Create rectangular pulse function
    h = raised_root_cosine(upsample=Ns, num_positive_lobes=span, alpha=alpha)
    bpsk_f = signal.convolve(sym_seq, h)

    t = np.r_[0.0:len(bpsk_f)] / fs
    s = bpsk_f * np.exp(1j*2*np.pi*float(cf)*t)
    return s


def add_noise(s, snr):
    # Calculate signal power and convert to dB
    sig_avg_watts = np.sqrt(np.mean(s.real**2 + s.imag**2) / 2.0)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - snr
    noise_avg_watts = 10.0 ** (noise_avg_db / 10.0)
    
    # Generate an sample of white noise
    mean_noise = 0
    noise_i = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(s))
    noise_q = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(s))
    noise = noise_i + 1j*noise_q
    return s + noise

def npsk(snr=None, n=2, fs=1, T_bits=5, cf=0.0, ml=16384, span=1, alpha=1.):

    s = QAM(qam=n, fs=fs, T_bits=T_bits, cf=cf, ml=ml, span=span, alpha=alpha)
    # normalization
    a = s.real
    b = s.imag
    s = s / np.sqrt(np.mean(a**2 + b**2))
    # add awgn
    if snr is not None:
        s = add_noise(s, snr=snr)
    print (f'generated signal length: {s.shape}')
    return s
