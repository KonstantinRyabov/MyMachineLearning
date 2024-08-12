import numpy as np

class MyKNNReg():
    def __init__(self, k = 0, train_size = (), metric = 'euclidean'):
        self.metric = metric
        self.k = k
        self.train_size = train_size
        self.X = []
        self.y = []
        
    def fit(self, X, y):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.train_size = X.shape
        
    def __calc_dis(self, metric, X, X_test_sample):
        distance = 0
        if metric == 'euclidean':
            distance = np.sqrt(np.sum(np.power(X - X_test_sample, 2), axis=1))
        elif metric == 'chebyshev':
            distance = np.max(np.abs(X - X_test_sample), axis=1)
        elif metric == 'manhattan':
            distance = np.sum(np.abs(X - X_test_sample), axis=1)
        elif metric == 'cosine':
            distance = 1 - np.sum(X * X_test_sample, axis=1)/(np.sqrt(np.sum(np.power(X, 2), axis=1)) * np.sqrt(np.sum(np.power(X_test_sample,2), axis=1)))
        return distance

    def __calc_predict(self, X_test):
        X_test = X_test.to_numpy()
        distanсes = np.empty((X_test.shape[0], self.train_size[0]))
        for i in range(X_test.shape[0]):
            X_test_sample = np.full((self.train_size[0], X_test[i].shape[0]), X_test[i])
            distanсes[i] = self.__calc_dis(self.metric, self.X, X_test_sample)
        indx = np.argsort(distanсes)[:, :self.k]
        target = self.y[indx]
        return target

    def predict(self, X_test):
        target = self.__calc_predict(X_test)
        mean_target = np.mean(target, axis = 1)
        return mean_target

    def __str__(self):  
        return f"MyKNNClf class: k={self.k}"