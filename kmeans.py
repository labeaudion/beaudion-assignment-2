import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.steps = []  # To store the steps for step-through

    def fit(self, data, init_method='random', initial_centroids=None):
        # Initialize centroids
        if init_method == 'random':
            self.centroids = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]
        elif init_method == 'manual':
            if initial_centroids is not None:
                self.centroids = np.array(initial_centroids)
            else:
                raise ValueError("Initial centroids must be provided for manual initialization.")
        elif init_method == 'farthest':
            self.centroids = self.farthest_first(data)
        elif init_method == 'kmeans++':
            self.centroids = self.kmeans_plus_plus(data)
        else:
            raise ValueError("Unknown initialization method.")

        for _ in range(self.max_iter):
            # Assign labels based on closest centroid
            self.labels = self._assign_labels(data)

            # Store the current step
            self.steps.append({
                'centroids': self.centroids.tolist(),
                'labels': self.labels.tolist()
            })

            # Calculate new centroids
            new_centroids = np.array([data[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        # Store the final state
        self.steps.append({
            'centroids': self.centroids.tolist(),
            'labels': self.labels.tolist()
        })

    def _assign_labels(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def farthest_first(self, data):
        # Implement Farthest First initialization
        centroids = [data[np.random.choice(data.shape[0])]]  # Choose first centroid randomly
        for _ in range(1, self.n_clusters):
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            farthest_point_index = np.argmax(np.min(distances, axis=1))
            centroids.append(data[farthest_point_index])
        return np.array(centroids)

    def kmeans_plus_plus(self, data):
        # Implement KMeans++ initialization
        centroids = [data[np.random.choice(data.shape[0])]]  # Choose first centroid randomly
        for _ in range(1, self.n_clusters):
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            min_distances = np.min(distances, axis=1)
            probabilities = min_distances / np.sum(min_distances)
            next_centroid_index = np.random.choice(data.shape[0], p=probabilities)
            centroids.append(data[next_centroid_index])
        return np.array(centroids)

    def get_centroids(self):
        return self.centroids

    def get_labels(self):
        return self.labels

    def get_steps(self):
        return self.steps
