from unittest import case

import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

from enum import StrEnum, auto

class Region(StrEnum):
    A = auto()
    B = auto()
    AB = auto() 

class bi_mode(StrEnum):
    auto_mode = auto()
    double_cross_mode = auto()
    triple_cross_mode = auto()

def _get_sampled(signal, nperseg, noverlap):
    step = nperseg - noverlap
    num_windows = (len(signal) - nperseg) // step + 1
    return np.lib.stride_tricks.as_strided(
        signal, shape=(num_windows, nperseg),
        strides=(signal.strides[0] * step, signal.strides[0]))


def _get_bicspec_grids(n, region:Region = Region.AB):
    """Index (j, k) grids for selected frequency region. 
    n = N/2 + 1 (rfft length)."""

    half = n - 1                      
    x_freq = np.arange(0, half + 1)
    match region:
        case Region.A:
            y_freq = np.arange(0, half // 2 + 1)
        case Region.B:
            y_freq = np.arange(-half, 0)
        case Region.AB:
            y_freq = np.arange(-half, half // 2 + 1)
    
    X, Y = np.meshgrid(x_freq, y_freq, indexing='ij')

    match region:
        case Region.A:
            mask = (Y >= 0) & (Y <= X) & (X + Y <= half)
        case Region.B:
            mask = (Y < 0) & (-Y <= X)
        case Region.AB:
            mask = (Y >= 0) & (Y <= X) & (X + Y <= half) | (Y < 0) & (-Y <= X)
             
    return X, Y, mask

def _calculate_valid_bispectrum(signal, fs, nperseg, noverlap=0, region:Region = Region.AB):
    samples = _get_sampled(signal, nperseg, noverlap)   # get the sampled signal
    samples -= np.mean(samples, axis=1, keepdims=True)   # extraction of mean value
    samples *= np.hanning(nperseg) # apply window function for energy leaking reduction
    
    FT = scipy.fft.rfft(samples, workers=-1, axis=1)     # perform FT. FT.shape == (nwin, n)
    n = FT.shape[1]
    X, Y, valid = _get_bicspec_grids(n, region)          # forming the mask for the region of interest
    Xv, Yv = X[valid], Y[valid]                          # Extracting the valid coordinates (be carrefull, Xv and Yv is flattened here)
    
    # Fourier amplitudes for pairs of interset. Result shape (nwin, P)
    Xx = FT[:, Xv] 
    Xy = FT[:, np.abs(Yv)]
    
    Xy[:, Yv < 0] = np.conj(Xy[:, Yv < 0])               # X(-m) = conj(X(m)) for diff couple
    
    Xxy = FT[:, Xv + Yv]                                 # j+k >= 0 
    
    #squared bicoherence b^2 = |<B>|^2 / (<|Xj Xk|^2> * <|Xjk|^2>)
    B_mean = np.mean(Xx * Xy * np.conj(Xxy), axis=0)
    denom = np.mean(np.abs(Xx * Xy)**2, axis=0) * np.mean(np.abs(Xxy)**2, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        b2 = np.abs(B_mean)**2 / denom
        B_mean /= np.sqrt(denom)
    return b2, B_mean, Xv, Yv


def bispectral(signal, fs, nperseg, noverlap=0, region:Region=Region.AB):

    b2, B_mean, Xv, Yv = _calculate_valid_bispectrum(signal, fs, nperseg, noverlap, region)

    half = nperseg // 2
    # упаковка в 2D-массив (j, k) для графика; вне A∪B — NaN
    match region:
        case Region.A:  yaxis = np.arange(0, half // 2 + 1)
        case Region.B:  yaxis = np.arange(-half, 0)
        case Region.AB: yaxis = np.arange(-half, half // 2 + 1)

    mat = np.full((half + 1, yaxis.size), np.nan)
    mat[Xv, Yv - yaxis[0]] = b2

    bs = np.full((half + 1, yaxis.size), np.nan, dtype=np.complex64)
    bs[Xv, Yv - yaxis[0]] = B_mean

    df = fs / nperseg
    freq_x = np.arange(0, half + 1) * df
    freq_y = yaxis * df
    return freq_x, freq_y, mat, bs

def surrogate_summed(signal, fs, nperseg, noverlap, n_surr=160, region=Region.AB):
    fur = np.abs(np.fft.fft(signal)); N = signal.size; h = N // 2
    sums = []
    for _ in range(n_surr):
        ph = np.zeros(N); rnd = np.random.uniform(0, 2*np.pi, h)
        ph[1:h] = rnd[1:h]; ph[h+1:] = -rnd[1:h][::-1]
        s = np.fft.ifft(fur * np.exp(1j*ph)).real
        b2s, _, Xs, Ys = _calculate_valid_bispectrum(s, fs, nperseg, noverlap, region)
        M = Xs + Ys
        sums.append(np.bincount(M, b2s) / np.bincount(M))
    return np.percentile(np.array(sums), 97.5, axis=0)

def bicoh_summed(signal, fs, nperseg, noverlap=0, region = Region.AB , threshold = None) -> tuple[np.ndarray, np.ndarray]:
    b2, _, Xv, Yv = _calculate_valid_bispectrum(signal, fs, nperseg, noverlap, region)

    M = Xv + Yv
    b2_sum = np.bincount(M, b2)
    count = np.bincount(M)
    b2_sum /= count

    if threshold is not None:
        b2_sum_thresholded = surrogate_summed(signal, fs, nperseg, noverlap, n_surr=160, region=region)
        

    df = fs / nperseg
    freq_m   = np.arange(len(b2_sum)) * df
    return freq_m, b2_sum, b2_sum_thresholded if threshold is not None else None

def plot_bicoherence(freq_j, freq_k, mat, 
                     mask=None, title="", c_map = "inferno_r"):
    if mask is not None:
        mat = mat.copy()
        mat[~mask & ~np.isnan(mat)] = 1e-12
    fig, ax = plt.subplots(figsize=(10, 10))
    pm = ax.pcolormesh(freq_j, freq_k, mat.T, shading='gouraud',
                        cmap=plt.get_cmap(c_map),
                        vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    def format_coord(x, y):
        j = np.argmin(np.abs(freq_j - x))
        k = np.argmin(np.abs(freq_k - y))
        z = mat[j, k]
        if np.isnan(z):
            return f'{x=:.2f}, {y=:.2f}'
        return f'{x=:.2f}, {y=:.2f}, {z=:.2f}'
    ax.format_coord = format_coord
    
    cb = plt.colorbar(pm)
    ax.set_xlabel(r"$f_1,\ kHz$")
    ax.set_ylabel(r"$f_2,\ kHz$")
    ax.set_aspect('equal')
    plt.show()

def bicoh_slice(freq_x, b2, foi):
    
    df = freq_x[1] - freq_x[0]
    half = len(freq_x) - 1
    m0 = int(round(foi / df))

    # consider region with f_2 < 0 and |f_2| < |f_1| (k < 0 and |k| > |j|)
    # it corresponds to the 
    freq_slice = np.arange(-half, half - m0 + 1)*df
    neg_sum = b2[m0:, half-m0]
    low_sum = b2[m0, half - m0 + 1 : half + m0]
    high_sum = b2[m0 : half-m0+1, half + m0]

    return freq_slice, np.hstack((neg_sum[::-1], low_sum, high_sum))


def surrogate_test(signal: np.ndarray, fs: float, nperseg: int, noverlap: int = 0, n_surrogates: int = 100, region: Region = Region.AB) -> np.ndarray:
    """
    Generate surrogate data and compute significance threshold (95th percentile).
    """

    surrogate_bicoh = []
    fur_ampl = np.abs(np.fft.fft(signal))
    for _ in range(n_surrogates):
        # Phase-randomized surrogate
        phases = np.zeros_like(signal)
        half = phases.size//2
        rnd = np.random.uniform(0, 2 * np.pi, half)
        phases[1:half] = rnd[1:half]
        phases[half + 1:] = -rnd[1:half][::-1] 
        surrogate = np.fft.ifft(fur_ampl * np.exp(1j * phases)).real
        _, _, bicoh, _= bispectral(surrogate, fs, nperseg, noverlap, region=region)
        surrogate_bicoh.append(bicoh)
    
    return np.percentile(np.array(surrogate_bicoh), 97.5, axis=0)

if __name__ == "__main__":
    data = np.load("bicoh_data.npz")
    t = data["t"]
    sig = data["sig"]
    fs = 1 / (t[1] - t[0])

    if True:
        #bicoh_slice(sig, fs, 20, int(0.5*fs))
        freqx, freqy, bicoh, bispec = bispectral(sig, fs, int(1*fs), region=Region.AB, noverlap=int(0.5*fs))# было 5 мс
        threshold = surrogate_test(sig, fs, int(1*fs), n_surrogates=10, noverlap=int(0.5*fs), region=Region.AB)
        #np.savez("result_data.npz", freqx=freqx, freqy=freqy, bicoh=bicoh, bispec=bispec, threshold=threshold)
    else:
        data = np.load("result_data.npz")
        freqx = data["freqx"]
        freqy = data["freqy"]
        bicoh = data["bicoh"]
        bispec = data["bispec"]
        threshold = data["threshold"]

    if True:
        mask = (bicoh > threshold)
        plot_bicoherence(freqx, freqy, bicoh, mask)
        plot_bicoherence(freqx, freqy, np.angle(bispec, deg=True), None,
                        c_map="twilight_shifted")
        #plot_bicoherence(freqx, freqy, None)
        #plot_bicoherence(freqx, freqy, np.imag(bispec))
        #plot_bicoherence(threshold, freq)

    if False:
        f_20, b2s_20 = bicoh_slice(freqx, bicoh, 19.8)#bicoh_summed(sig, fs, int(0.5*fs), region = Region.A)
        f_40, b2s_40 = bicoh_slice(freqx, bicoh, 39.8)
        f_58, b2s_58 = bicoh_slice(freqx, bicoh, 58.8)
        f_th, b2s_th = bicoh_slice(freqx, threshold, 19.8)

        fontsize = 24
        plt.rcParams.update({'font.size': fontsize})
        plt.figure()
        plt.plot(f_20, b2s_20, label=r"$f_\ast = 20\ kHz$")
        plt.plot(f_40, b2s_40, label=r"$f_\ast = 40\ kHz$")
        plt.plot(f_th, b2s_th, label=r"$Threshold$")
        plt.ylabel(r"$b^2(f, f_\ast)$")
        plt.xlabel(r"$f, kHz$")
        plt.legend(fontsize=18, borderpad=0.8, borderaxespad=1.0, labelspacing=0.6,handleheight=1.6)    
        plt.grid(alpha=0.5)
        plt.show()
    else:
        f_m, b2s_m, b2s_m_thresholded = bicoh_summed(sig, fs, int(1*fs), noverlap=int(0.5*fs),  region = Region.AB, threshold = threshold)
        plt.figure()
        plt.plot(f_m, b2s_m)
        if b2s_m_thresholded is not None:
            plt.plot(f_m, b2s_m_thresholded, label="Threshold", color="red")
        plt.ylabel(r"$b_\Sigma^2(f)$")
        plt.xlabel(r"$f, kHz$")
        plt.grid(alpha=0.5)
        plt.show()