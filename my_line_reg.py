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
        self.weights = np.ones((X.shape[1], 1))
        
        log_param = verbose
        current_weights = self.weights
        for x in range(self.n_iter):
            y_pred = np.dot(X, current_weights)
            loss = np.mean((y_pred - y) ** 2)
            grad = 2 * np.dot(X.T, (y_pred - y)) / np.size(y)
            current_weights = current_weights - self.learning_rate * grad
            if(verbose != 0):
                print(f"start | loss: {loss}")
                if x == log_param:
                    print(f"{x} | loss: {loss}")
                    log_param = log_param + verbose
        self.weights = current_weights
    
    def get_coef(self):
        return self.weights[1:]

    def __str__(self):  
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
