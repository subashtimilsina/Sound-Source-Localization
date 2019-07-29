import numpy as np


def gccphat(sig,refsig,fs,max_tau,interp=1):
    
    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    
    max_shift = np.minimum(int(interp * fs * max_tau), int(interp * n/2))
    
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    
    shift = np.argmax(np.abs(cc)) - max_shift
    
    tau = shift / float(interp * fs)
    
    return tau,cc