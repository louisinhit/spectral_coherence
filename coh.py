import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import signal

def calc_xspec_block(x, y, t, alpha=0, wht=False):
    # applies frequency shift of +/- alpha/2
    u_block = x*np.exp(-1j*np.pi*alpha*t)
    v_block = y*np.exp(+1j*np.pi*alpha*t)

    if wht == False:
        # take FFT of data blocks
        u_f = np.fft.fft(u_block, axis=0)
        v_f = np.fft.fft(v_block, axis=0)
    else:
        u_f = bu_fft(u_block).numpy()
        v_f = bu_fft(v_block).numpy()

    # calculates auto- and cross-power spectra
    Suu = (u_f * u_f.conj()).real
    Svv = (v_f * v_f.conj()).real
    return Suu, Svv


def cyclic_periodogram(x, Np, alpha_vec):
    print ('***** This is L (step) is 1 ******')
    step = Np
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
    #Y = Y.conj()

    Sxx, Syy = np.zeros((N_alpha, Np)), np.zeros((N_alpha, Np))
    
    t_block = []
    for n in range(N):
        n_start = n*step
        t_block.append(np.linspace(n_start, n_start+Np, Np))
        # calculate spectra for alpha values in 'alpha_vec'

    for a, alpha in enumerate(alpha_vec):
        su, sv = calc_xspec_block(X, Y, np.asarray(t_block).T, alpha)
        Sxx[a, :] = su.sum(axis=1)
        Syy[a, :] = sv.sum(axis=1)

    # apply FFT shift
    Sxx = np.fft.fftshift(Sxx, axes=(-1))
    Syy = np.fft.fftshift(Syy, axes=(-1))
    
    Sxx = Sxx/(N*step)
    Syy = Syy/(N*step)
    
    M = np.sqrt(Sxx * Syy)
    print (M.shape)
    return M.T
