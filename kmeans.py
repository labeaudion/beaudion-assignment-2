import numpy as np
import matplotlib.pyplot as plt

# algorithm for Kmeans
class Kmeans:
    def __init__(self, n_clusters=3, max_iter=100, tolerance=1e-4, init_method='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_method = init_method
        self.centroids = None

    # goes through the algorithm
    def fit(self, X):
        # Initialize centroids based on the specified method
        self.centroids = self.initialize_centroids(X)

        for i in range(self.max_iter):
            # Step 1: Assign clusters
            labels = self.assign_clusters(X)

            # Step 2: Calculate new centroids
            new_centroids = self.calculate_centroids(X, labels)

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break
            
            self.centroids = new_centroids


    # initializes centroids
    def initialize_centroids(self, X):
        if self.init_method == 'random':
            np.random.seed(42)
            random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[random_indices]
        elif self.init_method == 'farthest_first':
            centroids = [X[np.random.randint(X.shape[0])]]  # Choose one random point
            for _ in range(1, self.n_clusters):
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                min_distances = np.min(distances, axis=1)
                next_centroid = X[np.argmax(min_distances)]
                centroids.append(next_centroid)
            return np.array(centroids)
        elif self.init_method == 'kmeans++':
            centroids = [X[np.random.randint(X.shape[0])]]  # Choose one random point
            for _ in range(1, self.n_clusters):
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                min_distances = np.min(distances, axis=1)
                probabilities = min_distances ** 2
                probabilities /= np.sum(probabilities)  # Normalize
                next_centroid = X[np.random.choice(X.shape[0], p=probabilities)]
                centroids.append(next_centroid)
            return np.array(centroids)
        elif self.init_method == 'manual':
            return self.manual_initialization(X)
        else:
            raise ValueError("Invalid initialization method.")
        
    # manual initalization
    def manual_initialization(self, X):
        plt.scatter(X[:, 0], X[:, 1], s=30)
        plt.title("Select initial centroids (click on points)")
        points = plt.ginput(self.n_clusters)  # Get manual points from user
        plt.close()
        return np.array(points)
    
    # assigns clusters by Euclidean distance
    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    # 
    def calculate_centroids(self, X, labels):
        return np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

    # predicts the cluster assignments for new data points
    def predict(self, X):
        return self.assign_clusters(X)