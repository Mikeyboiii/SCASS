import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def visualize_with_time(amplitudes, frequencies,settings, label=0, marker_size=2):
    #plot against amplitudes, frequencies and time
    K = settings['K']
    L = settings['L']
    sr = settings['sr']
    v = np.zeros((K * L, 4))
    v[:, 0] = amplitudes.T.reshape(K * L) # amplitude axis
    v[:, 1] = frequencies.T.reshape(K * L) # frequency axis
    v[:, 2] = np.arange(K * L) / sr *1000 #time axis -> frames in miliseconds
    v[:, 3] = label.reshape(L,K).T.reshape(L*K)

    df = pd.DataFrame(v, columns=[ 'Amplitude', 'Frequency','Time/Frames', 'Cluster'])
    fig = px.scatter_3d(df, x='Amplitude', y='Frequency', z='Time/Frames',
                        color='Cluster',color_continuous_scale="plasma")

    fig.update_traces(marker=dict(size=marker_size, line=dict(width=2)), selector=dict(mode='markers'))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(yaxis_range=[0,2600])
    fig.update_layout(scene_aspectmode='cube')
    fig.show()

def visualize_feature(amplitudes, frequencies, phase_shift, settings, label=0, marker_size=2):
    # visualize data points in amplitude, frequency, phase space
    K = settings['K']
    L = settings['L']
    v = np.zeros((K * L, 4))
    v[:, 0] = amplitudes.T.reshape(K * L) # amplitude axis
    v[:, 1] = frequencies.T.reshape(K * L) # frequency axis
    v[:, 2] = phase_shift.T.reshape(K * L) # phase axis
    v[:, 3] = label

    df = pd.DataFrame(v, columns=['Amplitude', 'Frequency', 'Phase Shift', 'Cluster'])
    fig = px.scatter_3d(df, x='Amplitude', y='Frequency', z='Phase Shift',
                      color='Cluster',color_continuous_scale="plasma")
    fig.update_traces(marker=dict(size=marker_size, line=dict(width=2)), selector=dict(mode='markers'))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(scene_aspectmode='cube')
    fig.show()

def visualize_label_heatmap(label, settings):
    K = settings['K']
    L = settings['L']
    label_kl = label.reshape(K, L).T
    fig = plt.figure(figsize=(50,50))
    plt.imshow(label_kl)
