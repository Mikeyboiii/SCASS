import musdb
import numpy as np
import pandas as pd
from scipy import linalg
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import librosa
import IPython.display as ipd
import plotly.express as px
import matplotlib.pyplot as plt
from utils.visualize import *
from utils.helper import *
from utils.spectral_clustering import *
from utils.HWPS import *

data_path = '/data/'
audio_file = 'A Classic Education - NightOwl.stem.mp4'
song_name="education"

start, end = 0,4  #in seconds
audio_data , sr = librosa.load(data_path + audio_file)
audio_data = audio_data[sr*start:sr*end]
# audio_data /= np.max(np.abs(audio_data))

WIN_LENGTH = 1024
HOP_LENGTH = 512

D = librosa.stft(audio_data, n_fft=1024, win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
save_name = "_"+song_name+"_"+str(start)+'-'+str(end)+"_"+str(WIN_LENGTH)+"_"+str(HOP_LENGTH)

settings = {}
settings['WIN_LENGTH '] = 1024
settings['HOP_LENGTH'] = 512
settings['n_fft'] = 1024
settings['sr'] = sr
settings['K'] = D.shape[1]
settings['L'] = 10
K, L = settings['K'], settings['L']

A, freqs, phase_shift, args = get_peaks(D, settings['L'], settings,to_db=False)

# original reimplementatino
# hwps_model = hwps_lagrange(A, freqs)
# HWPS_feature = hwps_model.get_hwps_matrix()

# fast implementation
HWPS_feature = fast_hwps(A,freqs, 1)
similarity_matrix, distance_matrix = construct_similarity(librosa.amplitude_to_db(A, ref=np.max),
                                                          to_bark_scale(freqs),
                                                          HWPS_feature,
                                                          settings,
                                                          alpha=4)

# clustering methods
cls = sklearn.cluster.AgglomerativeClustering(n_clusters=4,linkage="average",affinity='precomputed')
agg_label = cls.fit_predict(distance_matrix)

#cls = sklearn.cluster.SpectralClustering(n_clusters=4,n_components=5,affinity="precomputed")
#spec_label = cls.fit_predict(similarity_matrix)

cls = sklearn.cluster.DBSCAN(eps=0.01, metric="precomputed")
DB_label = cls.fit_predict(distance_matrix)

X = spectral_clustering(similarity_matrix)
label = X.fit(k=4) #input k if you need
visualize_with_time(A, freqs,settings, label)

recon_audio = recover_music(D, args, label, settings, cluster_num=[0])


