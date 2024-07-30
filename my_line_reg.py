import numpy as np

class MyLineReg():
    def __init__(self, n_iter, learning_rate):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.l1_coef = 0
        self.l2_coef = 0
        self.metric = 0
        self.reg = ''
        self.weights = []
        self.result_metric = 0
        
    def __get_metrics(self, y, loss, norm, cost, m):
        if(self.metric == 'mae'):
            self.result_metric = np.sum(abs(loss)) / m
        elif(self.metric == 'mse'):
            self.result_metric = cost
        elif(self.metric == 'rmse'):
            self.result_metric = cost ** 0.5
        elif(self.metric == 'mape'):
            self.result_metric = np.sum(abs(loss / y)) * (100 / m)
        elif(self.metric == 'r2'):
            self.result_metric = 1 - np.sum(loss ** 2) / np.sum(norm ** 2)

    def fit(self, X, y, verbose = 0):
        X = X.to_numpy()
        y = y.to_numpy()
        
        X = np.hstack([np.ones((X.shape[0],1)), X])
        self.weights = np.ones(X.shape[1])
        
            
        log_param = verbose
        for iter in range(1, self.n_iter + 1):
            y_pred = np.dot(X, self.weights)
            loss = y_pred - y 
            norm = y - np.mean(y)
            m = np.size(y)
            
            l1_grad = self.l1_coef * np.sum(np.sign(self.weights))
            l2_grad = self.l2_coef * 2 * np.sum(self.weights)
            elasticnet_grad = l1_grad + l2_grad
        
            l_grad = 0
            if(self.reg == 'l1'):
                l_grad = l1_grad
            elif(self.reg == 'l2'):
                l_grad = l2_grad
            elif(self.reg == 'elasticnet'):
                l_grad = elasticnet_grad
            
            cost = np.sum(loss ** 2) / m
            self.__get_metrics(y, loss, norm, cost, m)
            
            grad = 2 * np.dot(X.T, loss) / m + l_grad
            learning_rate  = self.learning_rate if(isinstance(self.learning_rate, float)) else self.learning_rate(iter)
            self.weights = self.weights - learning_rate * grad
            
            if(verbose != 0):
                if self.metric == '':
                    print(f"start | loss: {cost}")
                else:
                    print(f"start | loss: {cost} | {self.metric}: {self.result_metric}")
                
                if iter == log_param:
                    if self.metric == '':
                        print(f"{iter} | loss: {cost}")
                    else:
                        print(f"{iter} | loss: {cost} | {self.metric}: {self.result_metric}")
                    log_param = log_param + verbose

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0],1)), X])
        y_pred = np.dot(X, self.weights)
        return y_pred
    
    def get_coef(self):
        return self.weights[1:]
    
    def get_best_score(self):
        return self.result_metric

    def __str__(self):  
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
