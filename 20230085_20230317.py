import numpy as np

class PCA:

    def __init__(self, k=None):
        self.k = k
        self.mean = None
        self.Q = None
        self.eigenvalues = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)

        shifted = X - self.mean

        cov = np.dot(shifted.T, shifted) / (len(X) - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.eigenvalues = eigenvalues

        if self.k is not None:
            eigenvectors = eigenvectors[:, :self.k]

        self.Q = eigenvectors.T

    def transform(self, X):
        shifted = X - self.mean
        return np.dot(self.Q, shifted.T).T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        return np.dot(self.Q.T, X_reduced.T).T + self.mean

    def get_explained_variance_ratio(self):
        total = np.sum(np.abs(self.eigenvalues))
        return np.abs(self.eigenvalues) / total
