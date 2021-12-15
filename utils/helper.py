import librosa
import numpy as np


def get_peaks(D, L, stft_settings, to_db=False):
    # D: STFT data matrix
    # L: keep top L frequency components

    n_fft = stft_settings['n_fft']
    sr = stft_settings['sr']
    K = stft_settings['K']

    args = np.argsort(-np.abs(D), axis=0)[:L, :]

    # amplitude matrix
    A = np.zeros([L, K])
    for i in range(K):
        A[:, i] = np.abs(D)[:, i][args[:, i]]
    if to_db: A = librosa.amplitude_to_db(A,ref=np.max)

    # frequency matrix
    original_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = original_freqs[args]

    # phase shifr matrix
    original_phase_shift = np.unwrap(np.angle(D))
    phase_shift = np.zeros([L, K])
    for i in range(K):
        phase_shift[:, i] = original_phase_shift[:, i][args[:, i]]

    #phase_next = np.hstack([original_phase_shift[:,1:],original_phase_shift[:,-2:-1]])
    #original_freqs = np.diff(phase_next - original_phase_shift) * sr / (2 * np.pi)
    #original_freqs = np.diff(np.unwrap(np.angle(D)))
    #freqs = np.zeros([L, K])
    #for i in range(K):
    #    freqs[:, i] = original_freqs[:, i][args[:, i]]

    return A, freqs, phase_shift, args


def to_bark_scale(f_matrix):
    new_freqs = np.zeros_like(f_matrix)
    freqs = f_matrix.copy()
    new_freqs += ((freqs>=20) & (freqs<100)) * 1
    new_freqs += ((freqs>=100) & (freqs<200)) * 2
    new_freqs += ((freqs>=200) & (freqs<300)) * 3
    new_freqs += ((freqs>=300) & (freqs<400)) * 4
    new_freqs += ((freqs>=400) & (freqs<510)) * 5
    new_freqs += ((freqs>=510) & (freqs<630)) * 6
    new_freqs += ((freqs>=630) & (freqs<770)) * 7
    new_freqs += ((freqs>=770) & (freqs<920)) * 8
    new_freqs += ((freqs>=920) & (freqs<1080)) * 9
    new_freqs += ((freqs>=1080) & (freqs<1270)) * 10
    new_freqs += ((freqs>=1270) & (freqs<1480)) * 11
    new_freqs += ((freqs>=1480) & (freqs<1720)) * 12
    new_freqs += ((freqs>=1720) & (freqs<2000)) * 13
    new_freqs += ((freqs>=2000) & (freqs<2320)) * 14
    new_freqs += ((freqs>=2320) & (freqs<2700)) * 15
    new_freqs += ((freqs>=2700) & (freqs<3150)) * 16
    new_freqs += ((freqs>=3150) & (freqs<3700)) * 17
    new_freqs += ((freqs>=3700) & (freqs<4400)) * 18
    new_freqs += ((freqs>=4400) & (freqs<5300)) * 19
    new_freqs += ((freqs>=5300) & (freqs<6400)) * 20
    new_freqs += ((freqs>=6400) & (freqs<7700)) * 21
    new_freqs += ((freqs>=7700) & (freqs<9500)) * 22
    new_freqs += ((freqs>=9500) & (freqs<12000)) * 23
    new_freqs += ((freqs>=12000) & (freqs<15500)) * 24

    return new_freqs

def recover(lab, k, l, num_components, args, cluster_num):
    #kl = lab.reshape(k, l).transpose(1,0)
    kl = lab.reshape(l,k)
    mask = np.zeros([num_components, k])
    print()
    for i in range(l):
        for j in range(k):
            if kl[i,j] in cluster_num:
                mask[args[i,j],j] = 1
    return mask

def match(sources, c1, c2, c3):
  truth = [c1, c2, c3]
  loss_matrix = np.zeros([3,3])
  for i in range(1,4):
    for j in range(3):
      loss_matrix[i-1, j] = ((sources[i] - truth[j][:sources[i].shape[0]]) ** 2).sum()

  loss_min = 100000000
  for i in [0, 1, 2]:
    for j in [0, 1, 2]:
      for z in [0, 1, 2]:
        if i != j and i != z and j!= z:
          sum = loss_matrix[0,i] + loss_matrix[1,j] + loss_matrix[2, z]
          if sum <= loss_min:
            loss_min = sum
            out = (i,j,z)

  return loss_matrix, loss_min, out

