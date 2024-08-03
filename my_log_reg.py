import numpy as np

class MyLogReg():
    def __init__(self, n_iter, learning_rate, metric = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric 
        self.weights = []
        self.result_metric = 0
        
    def __get_metrics(self, posotive, n):
        if(self.metric == 'accuracy'):
            self.result_metric = posotive / n
        #elif(self.metric == 'precision'):
            
        #elif(self.metric == 'recall'):
            
        #elif(self.metric == 'f1'):
            
        #elif(self.metric == 'roc_auc'):
            
    
    def fit(self, X, y, verbose = 0):
        X = X.to_numpy()
        y = y.to_numpy()
        eps = 1e-15
        
        X = np.hstack([np.ones((X.shape[0],1)), X])
        self.weights = np.ones(X.shape[1])
        
        log_param = verbose
        for x in range(0,self.n_iter + 1):
            z = np.dot(X, self.weights) * -1
            y_pred = 1 / (1 + np.exp(z))
            y_pred_class = np.round(y_pred).astype(int)
            loss = y_pred - y
            n = np.size(y)
            positive = np.sum(np.where(y == y_pred_class, 1, 0))
            self.__get_metrics(positive, n)
            
            if(x == self.n_iter + 1):
                break
            
            cost = -1 * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)) / n
            grad = np.dot(X.T, loss) * 1 / n
            self.weights = self.weights - self.learning_rate * grad
            
            if(verbose != 0):
                if self.metric is None:
                    print(f"start | loss: {cost}")
                else:
                    print(f"start | loss: {cost} | {self.metric}: {self.result_metric}")
                
                if iter == log_param:
                    if self.metric is None:
                        print(f"{iter} | loss: {cost}")
                    else:
                        print(f"{iter} | loss: {cost} | {self.metric}: {self.result_metric}")
                    log_param = log_param + verbose
                    
    def get_best_score(self):
        return self.result_metric
        
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
        y_pred_class = np.round(y_pred).astype(int)
        return y_pred_class
        
        
    def __str__(self):  
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"