import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import signal


def calc_xspec_block(x, y, t, alpha=0):
    # applies frequency shift of +/- alpha/2
    u_block = x*np.exp(-1j*np.pi*alpha*t)
    v_block = y*np.exp(+1j*np.pi*alpha*t)

    # take FFT of data blocks
    u_f = np.fft.fft(u_block)
    v_f = np.fft.fft(v_block)

    # calculates auto- and cross-power spectra
    Suu = (u_f * u_f.conj()).real
    Svv = (v_f * v_f.conj()).real
    return Suu, Svv


def cyclic_periodogram(x, Np, alpha_vec):
    print ('***** This is L (step) is 1 ******')
    step = Np // 4
    L = x.shape[-1]
    no = int(np.floor((L - Np) / step)) + 1
    print ('block numbers :', no)
    
    if no & (no - 1) != 0:
        print ('lenght not enough! padding zero!!')
        Pe = int(np.floor(int(np.log(no)/np.log(2))))
        P = 2**(Pe+1)
        print (P)
        x = np.concatenate((x, np.zeros((int((P-1)*step+Np)-L), dtype=x.dtype)), axis=-1)
        N = P
    else:
        N = no

    print ('x length', len(x))
    
    N_alpha = alpha_vec.shape[0]
    
    # Input Channelization
    X = np.zeros((Np, N), dtype=x.dtype)

    for k in range(N):
        X[:, k] = x[k * step : k * step + Np]
    
    Y = X
    Y = Y.conj()

    Sxx, Syy = np.zeros((N, N_alpha, Np)), np.zeros((N, N_alpha, Np))
    
    for n in range(N):
        n_start = n*step
        t_block = np.linspace(n_start, n_start+Np, Np)
        # calculate spectra for alpha values in 'alpha_vec'
        for a, alpha in enumerate(alpha_vec):
            Sxx[n, a, :], Syy[n, a, :] = calc_xspec_block(X[:, n], Y[:, n], t_block, alpha)

    # apply FFT shift
    Sxx = np.fft.fftshift(Sxx, axes=(-1))
    Syy = np.fft.fftshift(Syy, axes=(-1))
    
    Sxx = Sxx.sum(axis=0)/N
    Syy = Syy.sum(axis=0)/N
    
    M = np.sqrt(Sxx * Syy)
    print (M.shape)

    S = np.ones((Np, 2*N), dtype=M.dtype)
    
    for q in range(-N//2, N//2):
        for k in range(-Np//2, Np//2):         
            alpha = q/N + k/Np
            f = (k/Np - q/N) / 2
            
            m = int(Np * (f + 0.5))
            l = int(N * (alpha + 1))
            S[m, l] = M[q+N//2, k+Np//2]

    return S
