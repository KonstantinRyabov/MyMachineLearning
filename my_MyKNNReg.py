import numpy as np

class MyKNNReg():
    def __init__(self, k = 0, train_size = ()):
        self.k = k
        self.train_size = train_size
        self.X = []
        self.y = []
        
    def fit(self, X, y):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.train_size = X.shape
        
    def __calc_predict(self, X_test):
        X_test = X_test.to_numpy()
        distanсes = np.empty((X_test.shape[0], self.train_size[0]))
        for i in range(X_test.shape[0]):
            X_test_sample = np.full((self.train_size[0], X_test[i].shape[0]), X_test[i])
            distanсes[i] = np.sqrt(np.sum(np.power(self.X - X_test_sample, 2), axis=1))
        indx = np.argsort(distanсes)[:, :self.k]
        target = self.y[indx]
        return target

    def predict(self, X_test):
        target = self.__calc_predict(X_test)
        mean_target = np.mean(target, axis = 1)
        return mean_target

    def __str__(self):  
        return f"MyKNNClf class: k={self.k}"