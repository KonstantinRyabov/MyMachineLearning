class MyTreeClf():
    def __init__(self, max_depth, min_samples_split, max_leafs):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
    
    def get_best_split(X, y):
        from functools import reduce
        import math
        import numpy as np
        X['target'] = y
        # Функция для подсчета энтропии
        entropy = lambda s:-reduce(lambda x,y:x+y,map(lambda x:(x/len(s))*math.log2(x/len(s)), s.value_counts()))
        # словарь для атрибутов
        attrs = {}
        for attr in X.columns[:-1]:
            attrs[attr] = {"ig":0,"values":0}
            # считаем разделители
            ls = sorted(set(X[attr]))
            split_values = [np.mean(np.array([ls[item], ls[item + 1]])) for item in range(len(ls) - 1)]
            # считаем начальное состояние энтропии
            ent = entropy(X.iloc[:,-1])
            # перебираем все разделители и ищем набольший прирост инфомарции
            for value in split_values:
                df_left = X.query(attr+"<="+str(value))
                df_right = X.query(attr+">"+str(value))

                ent_left = entropy(df_left.iloc[:,-1])*df_left.shape[0]/X.shape[0]
                ent_right = entropy(df_right.iloc[:,-1])*df_right.shape[0]/X.shape[0]
                ig = ent - (ent_left + ent_right)

                if attrs[attr]["ig"] < ig:
                    attrs[attr]["ig"] = ig
                    attrs[attr]["values"] = value
                
        # ключ с максимальным значением
        maxkey = max(attrs, key = lambda x: attrs[x]["ig"])
        col_name, split_value, ig = maxkey, attrs[maxkey]["values"], attrs[maxkey]["ig"]
        return col_name, split_value, ig
    
    def __str__(self):  
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"
    