import numpy as np
import pandas as pd
import math
from collections import Counter
from sklearn.datasets import load_iris

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def _euclid_distance(self, x1, x2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
    
    def _get_neighbors(self, test_point):
        distances = []
        for xi, yi in zip(self.X_train, self.y_train):
            dist = self._euclid_distance(xi, test_point)
            distances.append((dist, yi))
        distances.sort(key=lambda x: x[0])
        return [label for _, label in distances[:self.k]]
    
    def predict_one(self, x):
        neighbors = self._get_neighbors(x)
        most_common = Counter(neighbors).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        correct = sum(p == t for p, t in zip(y_pred, y_test))
        return correct / len(y_test)

def split_data(X, y, test_size = None, shuffle = True, stratify = True, random_state = None):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame):
        y = y.values
    
    np.random.seed(random_state)
    
    if stratify:
        classes, _ = np.unique(y, return_counts=True)
        X_train, X_test, y_train, y_test = [], [], [], []
        
        for cls in classes:
            idx = np.where(y == cls)[0]
            if shuffle:
                np.random.shuffle(idx)
            
            split_idx = int(len(idx) * (1 - test_size))
            train_idx, test_idx = idx[:split_idx], idx[split_idx:]
            
            X_train.extend(X[train_idx])
            X_test.extend(X[test_idx])
            y_train.extend(y[train_idx])
            y_test.extend(y[test_idx])
        
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    
    else:
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)
            
        split_idx = int(len(X) * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
def solve_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_df = pd.DataFrame(X, columns=iris.feature_names)
    y_sr = pd.Series(y)
    X_train, X_test, y_train, y_test = split_data(X_df, y_sr, random_state=42)
    
    sqrt_y_test = int(math.sqrt(len(y_test)))
    k = sqrt_y_test if sqrt_y_test % 2 != 0 else sqrt_y_test - 1

    knn = KNNClassifier(k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")