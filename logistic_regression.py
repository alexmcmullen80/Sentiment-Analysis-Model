import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import preprocess
from cross_validate import cross_validate

print("-----SKLEARN MODEL-----")
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test  = preprocess(test_size = 0.2)

model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:}")
print(f"Precision: {precision:}")
print(f"Recall: {recall:}")
print(f"F1 score: {f1:}")

print("-----FROM SCRATCH MODEL-----")
class lr():
    def __init__(self, learning_rate=0.1, epoch=5000):
        self.lr = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # cross entropy loss
    def loss(self, actual, pred):
        y1 = actual * np.log(pred + 1e-9)
        y2 = (1 - actual) * np.log(1 - pred + 1e-9)
        return -np.mean(y1 + y2)
    
    def feed_forward(self,x):
        z = np.dot(x, self.weights) + self.bias
        A = self.sigmoid(z)
        return A
    
    # gradient descent
    def fit(self, x, y):
        samples, features = x.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for e in range(self.epoch):
            A = self.feed_forward(x)
            loss = self.loss(y,A)
            dz = A - y
            dw = (1 / samples) * np.dot(x.T, dz)
            db = (1 / samples) * np.sum(dz)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        y_hat = np.dot(x, self.weights) + self.bias
        y_predicted = self.sigmoid(y_hat)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        
        return np.array(y_predicted_cls)
    
model_two = lr()
cross_validate(X_train, X_test, y_train, y_test, model_two, num_folds=2)