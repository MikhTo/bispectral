import numpy as np
from bispectral import bispectral, plot_bicoherence, bicoh_slice, bicoh_summed

def KimPowersSimpleTest(Dt, fs, freqs, nperseg, noverlap=0, show=False, interaction=True):
    t_len = int(Dt*fs)
    t = np.arange(0, t_len).reshape(-1, nperseg)*1./fs

    random_phases_1 = np.random.uniform(-np.pi, np.pi, t_len//nperseg).reshape(t_len//nperseg, -1)
    random_phases_2 = np.random.uniform(-np.pi, np.pi, t_len//nperseg).reshape(t_len//nperseg, -1)
    if interaction:
        random_phases_3 = random_phases_1 + random_phases_2
    else:
        random_phases_3 = np.random.uniform(-np.pi, np.pi, t_len//nperseg).reshape(t_len//nperseg, -1)

    phases = (random_phases_1, random_phases_2, random_phases_3)


    sig = np.ravel(sum(np.cos(2*np.pi*f_i*t + ph_i) for f_i, ph_i in zip(freqs, phases)))
    sig += np.random.uniform(-0.1, 0.1, size=sig.shape)

    freq_x, freq_y, bicoh, bispec = bispectral(sig, fs, nperseg)
    if show:
        plot_bicoherence(freq_x, freq_y, bicoh)
    return freq_x, freq_y, bicoh, bispec




def KimPowersProductTest(Dt, fs, freqs, nperseg, noverlap=0, show=False):
    t_len = int(Dt*fs)
    t = np.arange(0, t_len).reshape(-1, nperseg)*1./fs

    random_phases_1 = np.random.uniform(-np.pi, np.pi, t_len//nperseg).reshape(t_len//nperseg, -1)
    random_phases_2 = np.random.uniform(-np.pi, np.pi, t_len//nperseg).reshape(t_len//nperseg, -1)
    random_phases_3 = np.random.uniform(-np.pi, np.pi, t_len//nperseg).reshape(t_len//nperseg, -1)
        

    phases = (random_phases_1, random_phases_2, random_phases_3)

    ampls = (1,1,0.5)
    sig = np.ravel(sum(a_i*np.cos(2*np.pi*f_i*t + ph_i) for a_i, f_i, ph_i in zip(ampls, freqs, phases))
                   + np.cos(2*np.pi*freqs[0]*t + phases[0])*np.cos(2*np.pi*freqs[1]*t + phases[1]))

    sig += np.random.uniform(-0.1, 0.1, size=sig.shape)

    freq_x, freq_y, bicoh, bispec = bispectral(sig, fs, nperseg)
    if show:
        plot_bicoherence(freq_x, freq_y, bicoh)
    return freq_x, freq_y, bicoh, bispec

def SliceTest():
    pass

def SummedTest():
    pass


def MonteCarloTest(t, s, fs, nperseg, noverlap=0, show=False):
    psi = lambda t, x, k_x, f, delta, tau=1: np.exp(-tau**-2 * ((k_x*x+delta)/(2*np.pi*f) - t)**2) * np.cos(k_x*x - 2*np.pi*f*t + delta) 

if __name__ == "__main__":
    KimPowersProductTest(10, 2e4, (1e3, 1.5e3, 2.5e3), int(1e3), show=True)