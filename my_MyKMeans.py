import numpy as np

class MyKMeans():
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
    
    def __str__(self):  
        return f"MyKMeans class: n_clusters={self.n_clusters}, max_iter={self.max_iter}, n_init={self.n_init}, random_state={self.random_state}"
    
    def __calc_dis(self, X, centroid):
        distance = np.sqrt(np.sum(np.power(X - centroid, 2), axis=1))
        return distance
        
    def fit(self, X):
        np.random.seed(self.random_state)
        X = X.to_numpy()
        n, m = X.shape
        centroids = np.empty((self.n_clusters, m))
        min = np.min(X, axis=0)
        max = np.max(X, axis=0)
        wcss_arr = np.empty((self.n_init))
        centroids_arr = np.empty((self.n_init, self.n_clusters, m))
        for i in range(self.n_init):
            for item in range(self.n_clusters):
                centroids[item] = np.random.uniform(min, max)
            for _ in range(self.max_iter):
                distanсes = np.empty((self.n_clusters, n))
                for i in range(self.n_clusters):
                    centroid_sample = np.full((n, m), centroids[i])
                    distanсes[i] = self.__calc_dis(X, centroid_sample)
                indx = np.argmin(distanсes)

                centroids_temp = np.empty((self.n_clusters, m))
                wcss = 0
                for item in range(self.n_clusters):
                    cluster = X[np.where(indx == item)]
                    n_cl,m_cl = cluster.shape
                    centroids_temp[item] = np.mean(cluster, axis = 0)
                    centroid_sample = np.full((n_cl, m_cl), centroids_temp[item])
                    dis_sum = np.sum(np.power(self.__calc_dis(cluster, centroid_sample), 2), axis = 0)
                    wcss = wcss + dis_sum
                wcss_arr[i] = wcss
                if np.array_equal(centroids, centroids_temp):
                    centroids = centroids_temp
                    break
                centroids = centroids_temp
                
                centroids_arr[i] = centroids
