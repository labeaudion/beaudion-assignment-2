import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tolerance=1e-4, init_method='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_method = init_method
        self.centroids = None
        self.labels = None
        self.history = []

    def fit(self, X):
        self.history = []
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._calculate_centroids(X, labels)
            self.history.append((labels, self.centroids.copy()))

            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break
            
            self.centroids = new_centroids

        self.labels = labels

    def initialize_centroids(self, X):
        if self.init_method == 'random':
            np.random.seed(42)
            random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[random_indices]
        elif self.init_method == 'farthest_first':
            centroids = [X[np.random.randint(X.shape[0])]]
            for _ in range(1, self.n_clusters):
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                min_distances = np.min(distances, axis=1)
                next_centroid = X[np.argmax(min_distances)]
                centroids.append(next_centroid)
            return np.array(centroids)
        elif self.init_method == 'kmeans++':
            centroids = [X[np.random.randint(X.shape[0])]]
            for _ in range(1, self.n_clusters):
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                min_distances = np.min(distances, axis=1)
                probabilities = min_distances ** 2
                probabilities /= np.sum(probabilities)
                next_centroid = X[np.random.choice(X.shape[0], p=probabilities)]
                centroids.append(next_centroid)
            return np.array(centroids)
        elif self.init_method == 'manual':
            return np.array([[2, 2], [5, 5], [8, 8]])  # Example manual centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X, labels):
        return np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
