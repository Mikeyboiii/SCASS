import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def construct_similarity(A_in_db, bark_freqs, HWPS_feature, settings, alpha=4):
    # A_in_db: amplitude matrix converted to db (L*K)
    # bark_freqs: frequency matrix converted to bark scale (L*K)
    # HWPS_feature: Hwps similarity matrix (L*K, L*K)
    # alpha: controls the weight of HWPS feature in the overall similarity

    K, L = settings['K'], settings['L']
    matrix = np.stack([A_in_db, bark_freqs])

    matrix = np.tile(matrix.reshape(2, K * L, -1), (1, K * L))
    delta = matrix - matrix.transpose(0, 2, 1)

    sigma = np.ones([2, 1, 1])
    sigma[0] = np.std(matrix[0, :, :])
    sigma[1] = np.std(matrix[0, :, :])
    # sigma=1

    similarity_matrix = np.exp(-np.sum((delta / sigma) ** 2, axis=0))
    similarity_matrix = similarity_matrix * (HWPS_feature ** alpha)

    distance_matrix = -np.log(similarity_matrix)  # compute distance matrix from similarity matrix
    distance_matrix -= distance_matrix[0, 0]  # the diagonal needs to be zero

    return similarity_matrix, distance_matrix

class spectral_clustering(object):
    def __init__(self, similarity_matrix):
        super(spectral_clustering, self).__init__()
        self.A = similarity_matrix # adjacency matrix
        self.D = np.identity(self.A.shape[0]) * np.sum(self.A, axis=1) # degree matrix
        self.L = self.D - self.A # laplacian matrix

        self.eigen_vals, self.eigen_vecs = linalg.eigh(self.L, self.D) # SVD

    def get_k(self, eigen_vals):
        # find the number of clusters using the eigengap heuristic
        n = len(eigen_vals)
        max_gap = 0.0
        gaps = []
        k = 1
        for i in range(n - 1):
            gap = np.abs(eigen_vals[i + 1] - eigen_vals[i])
            gaps.append(gap)
            if gap > max_gap:
                max_gap = gap
                k = i + 1
        print("eigengap heuristic:",k)
        self.k=k
        return k

    def fit(self,k=None):
        if not k:
          k=self.get_k(self.eigen_vals)
        new_x = self.eigen_vecs[:, 1:k+1]
        pred_label = KMeans(n_clusters=k).fit_predict(normalize(new_x))

        return pred_label