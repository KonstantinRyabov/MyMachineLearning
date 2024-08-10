import numpy as np

class MyKNNClf():
    def __init__(self, weight, k = 0, train_size = (), metric = 'euclidean'):
        self.weight = weight
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
        distanсes_k = np.sort(distanсes, axis = 1)[:, :self.k]
        return (class_predict, distanсes_k)

    def predict(self, X_test):
        class_predict, distanсes_k = self.__calc_predict(X_test)
        if self.weight == 'uniform':
            class_one = np.sum(class_predict, axis=1)
            predict = np.where(class_one >= (self.k - class_one), 1, 0)
        elif self.weight == 'rank':
            rows = class_predict.shape[0]
            all_rank = np.sum(1 / np.array(range(1, class_predict.shape[1] + 1)))
            predict = np.empty((rows))
            for item in  range(rows):
                weight_one = np.sum(1 / (np.where(class_predict[item] == 1)[0] + 1)) / all_rank
                weight_zero = np.sum(1 / (np.where(class_predict[item] == 0)[0] + 1)) / all_rank
                predict[item] = np.where(weight_one > weight_zero, 1, 0)
        elif self.weight == 'distance':
            rows = distanсes_k.shape[0]
            all_distance = np.sum(1 / distanсes_k, axis = 1)
            predict = np.empty((rows))
            for item in  range(rows):
                weight_one = np.sum(1 / distanсes_k[item][(np.where(class_predict[item] == 1)[0])]) / all_distance[item]
                weight_zero = np.sum(1 / distanсes_k[item][(np.where(class_predict[item] == 0)[0])]) / all_distance[item]
                predict[item] = np.where(weight_one > weight_zero, 1, 0)
        return predict.astype(int)
    
    def predict_proba(self, X_test):
        class_predict, distanсes_k = self.__calc_predict(X_test)
        if self.weight == 'uniform':
            class_one = np.sum(class_predict, axis=1)
            predict_proba = class_one/self.k
        elif self.weight == 'rank':
            rows = class_predict.shape[0]
            all_rank = np.sum(1 / np.array(range(1, class_predict.shape[1] + 1)))
            predict_proba = np.empty((rows))
            for item in  range(rows):
                weight_one = np.sum(1 / (np.where(class_predict[item] == 1)[0] + 1)) / all_rank
                predict_proba[item] = weight_one
        elif self.weight == 'distance':
            rows = distanсes_k.shape[0]
            all_distance = np.sum(1 / distanсes_k, axis = 1)
            predict_proba = np.empty((rows))
            for item in  range(rows):
                weight_one = np.sum(1 / distanсes_k[item][(np.where(class_predict[item] == 1)[0])]) / all_distance[item]
                predict_proba[item] = weight_one
        return predict_proba

    def __str__(self):  
        return f"MyKNNClf class: k={self.k}"