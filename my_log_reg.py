import numpy as np
import random

class MyLogReg():
    def __init__(self, n_iter, learning_rate, sgd_sample = None, random_state = 42, metric = None, reg = None, l1_coef = 0, l2_coef = 0):
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
        
        
    def __get_metrics(self, y_pred, y_pred_class, truepos, falseneg, allpos, alltrue, n):
        if(self.metric == 'accuracy'):
            self.result_metric = alltrue / n
        elif(self.metric == 'precision'):
            self.result_metric =  truepos / allpos
        elif(self.metric == 'recall'):
            self.result_metric = truepos / (truepos + falseneg)
        elif(self.metric == 'f1'):
            precision = truepos / allpos
            recall = truepos / (truepos + falseneg)
            self.result_metric = 2 * precision * recall / (precision + recall)
        elif(self.metric == 'roc_auc'):
            desc_score_indices = np.argsort(y_pred)[::-1]
            y_pred = np.round(y_pred[desc_score_indices], 10)
            y_pred_class = y_pred_class[desc_score_indices]
            pos = np.sum(y_pred_class == 1)
            neg = np.sum(y_pred_class == 0)

            pos_class = y_pred[y_pred_class == 1]
            neg_class = y_pred[y_pred_class == 0]

            total = 0
            for score in neg_class:
                higher = (pos_class > score).sum()
                equal = (pos_class == score).sum()
                total += higher + 0.5 * equal
            self.result_metric = total / (neg * pos)
    
    def fit(self, X, y, verbose = 0):
        random.seed(self.random_state)
        X = X.to_numpy()
        y_all = y.to_numpy()
        rows = X.shape[0]
        eps = 1e-15
        # добавляем вектор единиц слева
        X = np.hstack([np.ones((X.shape[0],1)), X])
        # инициализация весов
        self.weights = np.ones(X.shape[1])
        
        log_param = verbose
        for iter in range(1,self.n_iter + 1):
            
            # для формирования мини-пакет(Стохастический градиентный спуск)
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
                
            # предсказываем значение значение
            z = np.dot(X_grad, self.weights) * -1
            y_pred = 1 / (1 + np.exp(z))
            loss = y_pred - y
            n = np.size(y)
            
            z_all = np.dot(X, self.weights) * -1
            y_pred_all = 1 / (1 + np.exp(z_all))
            y_pred_class_all = np.round(y_pred_all).astype(int)
            n_all = np.size(y_all)
            
            # метрики
            alltrue = np.sum(np.where(y_all == y_pred_class_all, 1, 0))
            allpos = np.sum(np.where(y_pred_class_all == 1, 1, 0))
            truepos = np.sum(np.where((y_pred_class_all == 1) & (y_all == y_pred_class_all), 1, 0))
            falseneg = np.sum(np.where((y_pred_class_all == 0) & (y_all != y_pred_class_all), 1, 0))
            self.__get_metrics(y_pred_all, y_pred_class_all, truepos, falseneg, allpos, alltrue, n)
            
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
            
            # функция потерь и градиент
            cost = -1 * np.sum(y_all * np.log(y_pred_all + eps) + (1 - y_all) * np.log(1 - y_pred_all + eps)) / n_all
            grad = np.dot(X_grad.T, loss) * 1 / n + l_grad
            learning_rate  = self.learning_rate if(isinstance(self.learning_rate, float)) else self.learning_rate(iter)
            self.weights = self.weights - learning_rate * grad
            
            # логи
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