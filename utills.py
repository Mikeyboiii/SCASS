# Implementation of the "Harmonically wrapped peak similarity" as described in
# M. Lagrange et al. Normalized Cuts for Predominant Melodic Source Separation. M. Lagrange et al.
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4432646
import numpy as np

def get_shifted_spectral_pattern(freq_matrix):
    # takes [L, K] sized matrix as input
    # KL sized matrix of frequencies sorted by amplitude
    L, K = freq_matrix.shape
    spectral_pattern_matrix = [[0] * K] * L
    for k in range(K):
        F_kl =  freq_matrix[:, k]
        for l in range(L):
            F_kl_tilde = F_kl - freq_matrix[l][k]
            spectral_pattern_matrix[l][k] = F_kl_tilde
    return np.asarray(spectral_pattern_matrix)


def HWPS(p1, p2, set1, set2):
    hF = min(set1[0], set2[0])
    set1 -= p1
    set2 -= p2
    set1 /= hF
    set2 /= hF

    set1 -= set1.astype(int)
    set2 -= set2.astype(int)


def HWPS_matrix(freq_matrix, spectral_pattern_matrix):
    L, K = freq_matrix.shape
    Harm_matrix = np.zeros([K*L, K*L])
    freq_1d = freq_matrix.reshape(L * K, 1)
    for p1 in np.range(freq_1d):
        for p2 in np.range(freq_1d):
            peak1 = freq_1d[p1]
            peak2 = freq_1d[p2]

            set1 = freq_matrix[:, p1%K]
            set2 = freq_matrix[:, p2%K]

            Harm_matrix[p1, p2] = HWPS(peak1, peak2, set1, set2)

    return Harm_matrix # [K*L, K*L]



