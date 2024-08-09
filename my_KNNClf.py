import numpy as np

class MyKNNClf():
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
        class_predict = self.y[indx]
        class_one = np.sum(class_predict, axis=1)
        return class_one

    def predict(self, X_test):
        class_one = self.__calc_predict(X_test)
        predict = np.where(class_one >= (self.k - class_one), 1, 0)
        return predict
    
    def predict_proba(self, X_test):
        class_one = self.__calc_predict(X_test)
        predict_proba = class_one/self.k
        return predict_proba

    def __str__(self):  
        return f"MyKNNClf class: k={self.k}"