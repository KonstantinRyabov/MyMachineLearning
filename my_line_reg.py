import numpy as np
import random

class MyLineReg():
    def __init__(self, n_iter, learning_rate, sgd_sample = None, random_state = 42, l1_coef = 0, l2_coef = 0, metric = '', reg = ''):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.metric = metric
        self.reg = reg
        self.random_state = random_state
        self.sgd_sample = sgd_sample
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
        random.seed(self.random_state)
        X = X.to_numpy()
        y_all = y.to_numpy()
        rows = X.shape[0]
        # добавляем вектор единиц слева
        X = np.hstack([np.ones((X.shape[0],1)), X])
        # инициализация весов
        self.weights = np.ones(X.shape[1])
        
        log_param = verbose
        for iter in range(1, self.n_iter + 1):

            if(self.sgd_sample is not None):
                if(isinstance(self.sgd_sample, float)):
                    sample_rows_idx = random.sample(range(X.shape[0]), round(rows * self.sgd_sample))
                    X_grad = X[sample_rows_idx,:]
                    y = y_all[sample_rows_idx]
                else:
                    sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                    X_grad = X[sample_rows_idx,:]
                    y = y_all[sample_rows_idx]
            else:
                X_grad = X
                y = y_all
            
            # регуляризация
            l1_grad = self.l1_coef * np.sign(self.weights)
            l2_grad = self.l2_coef * 2 * self.weights
            elasticnet_grad = l1_grad + l2_grad
        
            l_grad = 0
            if(self.reg == 'l1'):
                l_grad = l1_grad
            elif(self.reg == 'l2'):
                l_grad = l2_grad
            elif(self.reg == 'elasticnet'):
                l_grad = elasticnet_grad
            
            y_pred_all = np.dot(X, self.weights)
            loss_all = y_pred_all - y_all
            norm = y_all - np.mean(y_all)
            m_all = np.size(y_all)
            cost = np.sum(loss_all ** 2) / m_all
            # метрики
            self.__get_metrics(y_all, loss_all, norm, cost, m_all)
            
            y_pred = np.dot(X_grad, self.weights)
            m = np.size(y)
            loss = y_pred - y
            
            # градиент
            grad = 2 * np.dot(X_grad.T, loss) / m + l_grad
            learning_rate  = self.learning_rate if(isinstance(self.learning_rate, float)) else self.learning_rate(iter)
            self.weights = self.weights - learning_rate * grad
            
            # логи
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
