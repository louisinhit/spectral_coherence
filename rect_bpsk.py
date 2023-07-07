import generate_dataset.QAM_codebook as cb

def QAM(qam, fs=1, T_bits=5, cf=0.0, ml=16384):
    
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
    p_t = np.ones((int(Ns)), dtype=M.dtype)
    t = np.r_[0.0:N] / fs
    s = signal.lfilter(p_t, [1], sym_seq)
    s = s * np.exp(1j*2*np.pi*float(cf)*t)
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

def npsk(snr=None, n=2, fs=1, T_bits=5, cf=0.0, ml=16384):
    s = QAM(qam=n, fs=fs, T_bits=T_bits, cf=cf, ml=ml)
    # normalization
    a = s.real
    b = s.imag
    s = s / np.sqrt(np.mean(a**2 + b**2))
    # add awgn
    if snr is not None:
        s = add_noise(s, snr=snr)
    print (f'generated signal length: {s.shape}')
    return s
