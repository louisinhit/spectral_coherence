import numpy as np
from scipy import signal

def autossca_L_1(x, fs, Np):
    print ('***** This is L (step) is 1 ******')
    step = 1
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
    # Input Channelization
    X = np.zeros((Np, N), dtype=x.dtype)
    for k in range(N):
        X[:, k] = x[k * step : k * step + Np]

    # Windowing
    a = signal.windows.chebwin(Np, at=100)
    XW = X * a[:,None]

    # First FFT
    XF1 = np.fft.fft(XW, axis=0, norm='ortho')
    XF1 = np.fft.fftshift(XF1, axes=0)

    # Downconversion
    E = np.zeros((Np, N), dtype=complex)
    for k in range(-Np // 2, Np // 2):
        for m in range(N):
            E[k + Np // 2, m] = np.exp(-2j * np.pi * k * m / Np)

    XD = XF1 * E

    # Multiplication
    xc = np.conj( x[Np//2 : Np//2+N] )
    XM = XD * xc
    XM = XM.T

    # Second FFT
    XF2 = np.fft.fft(XM, axis=0, norm='ortho')
    XF2 = np.fft.fftshift(XF2, axes=0)
    M = np.abs(XF2)
    print ('mmmmmmmmmmm', M.shape)

    alpha_o = np.linspace(-1, 1, 2*N, endpoint=True) * fs
    f_o = np.linspace(-0.5+(0.5/N), 0.5-(0.5/Np), Np, endpoint=True)* fs

    print (alpha_o.shape)
    print (f_o.shape)

    Sx = np.zeros((Np, 2*N), dtype=M.dtype)
    
    for q in range(-N//2, N//2):
        for k in range(-Np//2, Np//2):

            alpha = q/N + k/Np
            f = (k/Np - q/N) / 2
            
            m = int(Np * (f + 0.5))
            l = int(N * (alpha + 1))
            Sx[m, l] = M[q+N//2, k+Np//2]

    return Sx, alpha_o, f_o



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


def cyclic_periodogram(x, Np):
    print ('***** This is L (step) is 1 ******')
    step = 1
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
        
    alpha_vec = np.linspace(-0.5, 0.5, N, endpoint=True)
    N_alpha = alpha_vec.shape[0]
    
    # Input Channelization
    X = np.zeros((Np, N), dtype=x.dtype)

    for k in range(N):
        X[:, k] = x[k * step : k * step + Np]
    
    Y = X
    Y = Y.conj()

    Sxx, Syy = np.zeros((Np, N)), np.zeros((Np, N))
    
    print ('******************** \n')
    print (X.shape, Y.shape, Sxx.shape, Syy.shape)
    print (N_alpha)

    for a, alpha in enumerate(alpha_vec):
        t_block = np.linspace(0+a, Np+a, 1, endpoint=False)
        Sxx[:, a], Syy[:, a] = calc_xspec_block(X[:, a], Y[:, a], t_block, alpha)

    # apply FFT shift
    Sxx = np.fft.fftshift(Sxx, axes=(-1))
    Syy = np.fft.fftshift(Syy, axes=(-1))
    
    M = np.sqrt(Sxx * Syy).T
    print (M.shape)

    S = np.ones((Np, 2*N), dtype=M.dtype)
    
    for q in range(-N//2, N//2):
        for k in range(-Np//2, Np//2):         
            alpha = q/N + k/Np
            f = (k/Np - q/N) / 2
            
            m = int(Np * (f + 0.5))
            l = int(N * (alpha + 1))
            Sx[m, l] = M[q+N//2, k+Np//2]

    return S
