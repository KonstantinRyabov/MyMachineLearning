class MyTreeClf():
    def __init__(self, max_depth, min_samples_split, max_leafs):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
    
    def __str__(self):  
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"