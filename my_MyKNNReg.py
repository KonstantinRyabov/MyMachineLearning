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
    
    def __str__(self):  
        return f"MyKNNReg class: k={self.k}"