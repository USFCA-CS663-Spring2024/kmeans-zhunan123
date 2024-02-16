from cluster import cluster
import numpy as np

class KMeans1(cluster):
    def __init__(self, k=5, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        X = np.array(X)
        n_instances, n_features = X.shape
        centroids = X[np.random.choice(n_instances, self.k, replace=False), :]
        centroids_prev = np.zeros(centroids.shape)
        clusters = np.zeros(n_instances, dtype=int)
        centroids_movement = np.linalg.norm(centroids - centroids_prev, axis=1)

        iteration = 0
        while np.any(centroids_movement > 0.0001) and iteration < self.max_iterations:
            for i in range(n_instances):
                distances = np.linalg.norm(X[i] - centroids, axis=1)
                cluster_index = np.argmin(distances)
                clusters[i] = cluster_index
                
            centroids_prev = np.copy(centroids)
            for i in range(self.k):
                points_in_cluster = X[clusters == i]
                if points_in_cluster.size:
                    centroids[i] = np.mean(points_in_cluster, axis=0)
                    
            centroids_movement = np.linalg.norm(centroids - centroids_prev, axis=1)
            iteration += 1
            
        return clusters, centroids.tolist()

