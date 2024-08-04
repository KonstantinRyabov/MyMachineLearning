import numpy as np

class MyLogReg():
    def __init__(self, n_iter, learning_rate, metric = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric 
        self.weights = []
        self.result_metric = 0
        
    def __binary_clf_curve(self, y_true, y_score):
    
        desc_score_indices = np.argsort(y_score)[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        
        distinct_indices = np.where(np.diff(y_score))[0]
        end = np.array([y_true.size - 1])
        
        threshold_indices = np.hstack((distinct_indices, end))
        thresholds = y_score[threshold_indices]
        
        tps = np.cumsum(y_true)[threshold_indices]
        fps = (1 + threshold_indices) - tps
        
        return tps, fps, thresholds
    
    def __roc_auc_score(self, y_true, y_score):
        tps, fps, _ = self.__binary_clf_curve(y_true, y_score)
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        zero = np.array([0])
        tpr_diff = np.hstack((np.diff(tpr), zero))
        fpr_diff = np.hstack((np.diff(fpr), zero))
        auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2
        return auc
        
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
            self.result_metric = self.__roc_auc_score(y_pred_class, np.round(y_pred, 10))
    
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
            alltrue = np.sum(np.where(y == y_pred_class, 1, 0))
            allpos = np.sum(np.where(y_pred_class == 1, 1, 0))
            truepos = np.sum(np.where((y_pred_class == 1) & (y == y_pred_class), 1, 0))
            falseneg = np.sum(np.where((y_pred_class == 0) & (y != y_pred_class), 1, 0))
            self.__get_metrics(y_pred, y_pred_class, truepos, falseneg, allpos, alltrue, n)
            
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