from xhistogram.xarray import histogram
import xarray as xr
import numpy as np


def get_histograms(sets, amps, n_bins):
    hist_range = np.arange(0, 1 + 1 / n_bins, 1 / n_bins)
    sets = xr.DataArray(sets, dims=['his', 'LKdata1', 'LKdata2'], name='sets')
    amps = xr.DataArray(amps, dims=['his', 'LKdata1', 'LKdata2'], name='amps')

    hist = histogram(sets, bins=hist_range, dim=['his'], weights=amps, density=True)

    return np.asarray(hist)


def fast_hwps(amplitude_matrix, freq_matrix, slices, n_bins=20):
    L, K = freq_matrix.shape
    # A fast implementation of Harmonically Wrapped Peak Similarity

    # compute hF matrix (L*K, L*K), "conservative version"
    # print('compute hF matrix')
    freq_1d_tile = np.tile(freq_matrix.reshape([1, L * K]), (L * K, 1))
    hF_matrix = np.maximum(np.minimum(freq_1d_tile, freq_1d_tile.T), [43])

    '''
    # "ground truth version"
    hF_matrix = np.zeros([K,K])
    for i in range(K):
      for j in range(i, K):
        hF_matrix[i, j] = min(freq_matrix[0,i], freq_matrix[0,j])
    diag = np.diag(hF_matrix)
    hF_matrix = hF_matrix.T + hF_matrix
    np.fill_diagonal(hF_matrix, diag)
    hF_matrix = np.tile(hF_matrix, (L, L))
    '''

    # compute sets
    # print('compute sets')
    F = np.tile(freq_matrix, (L, 1, 1))
    sets = np.transpose(F, (1, 0, 2)) - F  # size = (l, l, k), sets[:,l,k] is the set for peak[l, k]

    # compute histograms
    print('computing histograms:')
    sets = sets.reshape(L, 1, L * K)
    sets = np.round(sets * n_bins) + 1  # quantization step, according to official matlab implementation

    # serialize to tradeoff memory and speed
    # setting slice=1 means no serialization, results stay the same
    block_size = L * K // slices
    for i in range(slices):
        if i % 1 == 0: print('\r', round((i + 1) * 100 / slices, 1), '%', end='')

        amp_block = np.tile(amplitude_matrix, (L, 1, 1)).transpose(1, 0, 2).reshape(L, 1, L * K)
        if i < slices - 1:
            sets_block = np.tile(sets[:, :, i * block_size:(i + 1) * block_size], (1, L * K, 1)) / hF_matrix[:,i * block_size:(i + 1) * block_size]
            sets_block -= sets_block.astype(np.int)
            amps_block = np.tile(amp_block[:, :, i * block_size:(i + 1) * block_size], (1, L * K, 1))
            hist = get_histograms(sets_block, amps_block, n_bins)
        else:
            sets_block = np.tile(sets[:, :, i * block_size:], (1, L * K, 1)) / hF_matrix[:, i * block_size:]
            sets_block -= sets_block.astype(np.int)
            amps_block = np.tile(amp_block[:, :, i * block_size:], (1, L * K, 1))
            hist = get_histograms(sets_block, amps_block, n_bins)
        if i == 0:
            histograms = hist
        else:
            histograms = np.concatenate((histograms, hist), axis=1)

    histograms = histograms.transpose(2, 0, 1)

    # print('\ncompute harmoinicity matrix')
    # compute harmoinicity matrix
    num = np.sum(histograms * (histograms.transpose(0, 2, 1)), axis=0)
    square = np.sum(histograms * histograms, axis=0)
    denom = np.sqrt(square * square.T)
    Harm_matrix = np.exp((num / denom) ** 2)

    return Harm_matrix


class hwps_lagrange():
    # python reimplementation of the original matlab code provided by the authors

    def __init__(self, amps_matrix, freqs_matrix, l=20):
        super().__init__()
        self.A = amps_matrix
        self.F = freqs_matrix
        self.L = amps_matrix.shape[0]
        self.K = amps_matrix.shape[1]
        self.l = l  # number of bins in histograms
        self.histogram_mode = 'official'

    def get_histograms(self, p1, p2, f1, f2, a1, a2):

        l = self.l

        fsp1 = f1 - p1
        fsp2 = f2 - p2
        hF = max(min(p1, p2), 43)

        if self.histogram_mode == 'ours':
            hist_range = np.arange(0, 1.1, 0.1)

            sets1 = xr.DataArray(fsp1, dims=['his'], name='sets1')
            sets2 = xr.DataArray(fsp2, dims=['his'], name='sets2')

            a1 = xr.DataArray(a1, dims=['his'], name='a1')
            a2 = xr.DataArray(a1, dims=['his'], name='a2')

            h1 = histogram(sets1, bins=hist_range, dim=['his'], weights=a1, density=True)
            h2 = histogram(sets2, bins=hist_range, dim=['his'], weights=a2, density=True)

        elif self.histogram_mode == 'official':
            fhw1, fhw2 = (fsp1 / hF) % 1, (fsp2 / hF) % 1
            h1, h2 = np.zeros(l), np.zeros(l)
            i1 = (np.round(fhw1 * self.l) % l).astype(np.int)
            i2 = (np.round(fhw2 * self.l) % l).astype(np.int)
            for k in range(self.L):
                h1[i1[k]] += a1[k]
                h2[i2[k]] += a2[k]

        sim = (h1 @ h2.T) / (np.sqrt(h1 @ h1.T) * np.sqrt(h2 @ h2.T))
        return sim

    def get_hwps_matrix(self):
        num_peaks = self.L * self.K
        hwps_matrix = np.zeros([num_peaks, num_peaks])
        for comp1 in range(self.L):
            for frame1 in range(self.K):
                idx1 = comp1 * self.K + frame1
                peak1 = self.F[comp1, frame1]
                f_set1 = self.F[:, frame1]
                a_set1 = self.A[:, frame1]
                print('\r', 'computing hwps l=%d k=%d' % (comp1, frame1), end='')
                for comp2 in range(self.L):
                    for frame2 in range(self.K):
                        idx2 = comp2 * self.K + frame2
                        peak2 = self.F[comp2, frame2]
                        f_set2 = self.F[:, frame2]
                        a_set2 = self.A[:, frame2]
                        hwps_matrix[idx1, idx2] = self.get_histograms(peak1, peak2, f_set1, f_set2, a_set1, a_set2)
        return np.exp(hwps_matrix ** 2)


def sort_by_freq(freqs, settings):
    # returns frequency and amplitude matrix sorted by frequency

    sorted_freqs_args = np.argsort(freqs, axis=0)
    K = settings['K']

    sorted_freqs = np.zeros(freqs.shape)
    for i in range(K):
        sorted_freqs[:, i] = freqs[:, i][sorted_freqs_args[:, i]]

    sorted_A_wrt_freq = np.zeros(A.shape)
    for i in range(K):
        sorted_A_wrt_freq[:, i] = A[:, i][sorted_freqs_args[:, i]]

    return sorted_freqs, sorted_A_wrt_freq
