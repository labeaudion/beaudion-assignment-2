import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tolerance=1e-4, init_method='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_method = init_method
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # initializes centroids based on the chosen method
        self.centroids = self.initialize_centroids(X)
        
        for i in range(self.max_iter):
            # assigns labels
            self.labels = self.assign_labels(X)
            
            # calculates new centroids
            new_centroids = self.calculate_centroids(X)
            
            # checks for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break
            
            self.centroids = new_centroids

    # function to initalize centroids
    def initialize_centroids(self, X):
        if self.init_method == 'random':
            return self.random_initialization(X)
        elif self.init_method == 'farthest_first':
            return self.farthest_first_initialization(X)
        elif self.init_method == 'kmeans++':
            return self.kmeans_plus_plus_initialization(X)
        else:
            raise ValueError("Invalid initialization method specified.")

    # function that implements random initialization
    def random_initialization(self, X):
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices]

    # function that implements farthest first implementation
    def farthest_first_initialization(self, X):
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    # function that implements kmeans++ implementation
    def kmeans_plus_plus_initialization(self, X):
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
            probabilities = distances / distances.sum()
            next_centroid = X[np.random.choice(X.shape[0], p=probabilities)]
            centroids.append(next_centroid)
        return np.array(centroids)

    # function that assigns labels to data points based on Euclidean distance
    def assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    # function that calculates centroids given a dataset
    def calculate_centroids(self, X):
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

    # function that assigns labels to new data points based on centroids determined during fit(self, X)
    def predict(self, X):
        return self._assign_labels(X)

# Example usage:
#if __name__ == "__main__":
  #  np.random.seed(42)
  #  X = np.random.rand(100, 2)  # 100 points in 2D

    # Create and fit the model using KMeans++
  #  kmeans = KMeans(n_clusters=3, init_method='kmeans++')
  #  kmeans.fit(X)

    # Print the resulting centroids and labels
  #  print("Centroids:\n", kmeans.centroids)
   # print("Labels:\n", kmeans.labels)
