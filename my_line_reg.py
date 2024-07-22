import numpy as np

class MyLineReg():
    def __init__(self, n_iter, learning_rate):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = []
    def fit(self, X, y, verbose = 0):
        X = X.to_numpy()
        y = y.to_numpy()
        
        X = np.hstack([np.ones((X.shape[0],1)), X])
        self.weights = np.ones(X.shape[1])
        
        log_param = verbose
        for x in range(0,self.n_iter):
            y_pred = np.dot(X, self.weights)
            loss = y_pred - y
            m = np.size(y)
            cost = np.sum(loss ** 2) / m
            grad = 2 * np.dot(X.T, loss) / m
            self.weights = self.weights - self.learning_rate * grad
            if(verbose != 0):
                print(f"start | loss: {cost}")
                if x == log_param:
                    print(f"{x} | loss: {cost}")
                    log_param = log_param + verbose
                    
    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0],1)), X])
        y_pred = np.dot(X, self.weights)
        return y_pred
    
    def get_coef(self):
        return self.weights[1:]

    def __str__(self):  
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
