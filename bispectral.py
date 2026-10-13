import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

from enum import StrEnum, auto

class Region(StrEnum):
    A = auto()
    AB = auto() 

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
    y_freq = np.arange(-half if region is Region.AB else 0,
                                            half // 2 + 1)
    
    X, Y = np.meshgrid(x_freq, y_freq, indexing='ij')

    mask = (Y >= 0) & (Y <= X) & (Y + X <= half) # sum freqs area (region A)
    if region is Region.AB:
        mask |= (Y <= -1) & (Y >= -X) # diff freqs area (region B)    
             
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
    if region is Region.AB:
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
    yaxis = np.arange(-half if region is Region.AB else 0, half // 2 + 1)

    mat = np.full((half + 1, yaxis.size), np.nan)
    mat[Xv, Yv - yaxis[0]] = b2

    bs = np.full((half + 1, yaxis.size), np.nan, dtype=np.complex64)
    bs[Xv, Yv - yaxis[0]] = B_mean

    df = fs / nperseg
    freq_x = np.arange(0, half + 1) * df
    freq_y = yaxis * df
    return freq_x, freq_y, mat, bs

def bicoh_summed(signal, fs, nperseg, noverlap=0, region = Region.AB):
    b2, _, Xv, Yv = _calculate_valid_bispectrum(signal, fs, nperseg, noverlap, region)

    M = Xv + Yv
    b2_sum = np.bincount(M, b2)
    count = np.bincount(M)
    b2_sum /= count

    df = fs / nperseg
    freq_m   = np.arange(1, len(b2_sum)) * df
    return freq_m, b2_sum

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

def bicoh_slice(signal, fs, foi, nperseg, noverlap=0):
    freq_x, freq_y, b2, _ = bispectral(signal, fs, nperseg, noverlap)
    
    df = freq_x[1] - freq_x[0]
    half = len(freq_x) - 1
    m0 = int(round(foi / df))

    # consider region with f_2 < 0 and |f_2| < |f_1| (k < 0 and |k| > |j|)
    # it corresponds to the 
    freq_slice = np.arange(-half, half - m0 + 1)*df
    neg_sum = b2[m0:, half-m0]
    low_sum = b2[m0, half - m0 + 1 : half + m0]
    high_sum = b2[m0 : half-m0+1, half + m0]
    plt.figure()
    plt.plot(freq_slice, np.hstack((neg_sum[::-1], low_sum, high_sum)))
    plt.show()


def surrogate_test(signal: np.ndarray, fs: float, nperseg: int, noverlap: int = 0, n_surrogates: int = 100) -> np.ndarray:
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
        _, _, bicoh, _= bispectral(surrogate, fs, nperseg, noverlap)
        surrogate_bicoh.append(bicoh)
    
    return np.percentile(np.array(surrogate_bicoh), 97.5, axis=0)

if __name__ == "__main__":
    data = np.load("bicoh_data.npz")
    t = data["t"]
    sig = data["sig"]
    fs = 1 / (t[1] - t[0])


    #bicoh_slice(sig, fs, 20, int(0.5*fs))
    freqx, freqy, bicoh, bispec = bispectral(sig, fs, int(1*fs), region=Region.AB, noverlap=int(0*fs))# было 5 мс
    threshold = surrogate_test(sig, fs, int(1*fs), n_surrogates=150, noverlap=int(0*fs))

    mask = (bicoh > threshold)
    plot_bicoherence(freqx, freqy, bicoh, mask)
    plot_bicoherence(freqx, freqy, np.angle(bispec, deg=True), None,
                     c_map="twilight_shifted")
    plot_bicoherence(freqx, freqy, threshold)
    #plot_bicoherence(freqx, freqy, np.imag(bispec))
    #plot_bicoherence(threshold, freq)

    
    #f, b2s = bicoh_slice(sig, fs, 19.8, int(0.5*fs))#bicoh_summed(sig, fs, int(0.5*fs), region = Region.A)
    #plt.figure()
    #plt.plot(f, b2s)
    #plt.ylabel(r"$b^2_{\Sigma}$")
    #plt.xlabel(r"$f, kHz$")
    #plt.show()