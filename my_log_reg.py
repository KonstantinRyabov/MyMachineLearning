import numpy as np

class MyLogReg():
    def __init__(self, n_iter, learning_rate):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = []
    
    def fit(self, X, y, verbose = 0):
        X = X.to_numpy()
        y = y.to_numpy()
        eps = 1e-15
        
        X = np.hstack([np.ones((X.shape[0],1)), X])
        self.weights = np.ones(X.shape[1])
        
        log_param = verbose
        for x in range(0,self.n_iter):
            z = np.dot(X, self.weights) * -1
            y_pred = 1 / (1 + np.exp(z))
            loss = y_pred - y
            n = np.size(y)
            cost = -1 * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)) / n
            grad = np.dot(X.T, loss) * 1 / n
            self.weights = self.weights - self.learning_rate * grad
            if(verbose != 0):
                print(f"start | loss: {cost}")
                if x == log_param:
                    print(f"{x} | loss: {cost}")
                    log_param = log_param + verbose
    
    def get_coef(self):
        return self.weights[1:]
    
    def predict_proba(self, X):
        X = np.hstack([np.ones((X.shape[0],1)), X])
        z = np.dot(X, self.weights) * -1
        y_pred = 1 / (1 + np.exp(z))
        return y_pred
    
    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0],1)), X])
        z = np.dot(X, self.weights) * -1
        y_pred = 1 / (1 + np.exp(z))
        y_pred_class = np.round(y_pred, dtype = np.integer).astype(int)
        return y_pred_class
        
        
    def __str__(self):  
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"