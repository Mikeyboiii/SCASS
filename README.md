# Unsupervised Harmonic Sound Source Separation with Spectral Clustering 

Lingyu Zhang, Yiming Lin, Lucy Wang, Zhaoyuan Deng

## Abstract

  With the growing amount of audio data generated and processed in modern day media industry, music retrieval largely dependson automated algorithms. Separation of harmonic music components plays a crucial role in labeling and retrieving music data. Due to the amount of labeling required for instument identification, supervised methods are often difficult to perform. In this work, we first implemented the the unsupervised source separation method introduced in [1]. We modeled mixed sources of audio signals by sinusoidal modeling with Short-Time Fourier Transforms. Based on selected spectral peaks of sinusoidal parameters, we constructed a similarity function between frames and frequency components, and applied spectral clusteringto globally partition data. We then define a matching loss metric to evaluate clustering results and analyzed the effectiveness and weighted combinations of similarity features as well as clustering methods. We finally propose a fast implementation of Harmonically Wrapped Peak Similarity to run full graph spectral clustering, taking only 3.2% of the time of the naive implementation.
  
  (This is a course project for CS4774 Unsupervised Machine Learning at Columbia University Fall 2021, instructed by Prof. Nakul Verma)
  
  
[1] L. G. Martins, J. J. Burred, G. Tzanetakis, and M. Lagrange.Polyphonic instrument recognition using spectral clustering.InISMIR, 2007
