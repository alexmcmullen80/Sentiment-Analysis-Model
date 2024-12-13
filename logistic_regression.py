import numpy as np
from preprocess import preprocess, preprocess_with_sbert, load_numpy_arrays
from cross_validate import cross_validate

class lr():
    
    # initialize variables
    def __init__(self, learning_rate=0.1, epoch=10000):
        self.lr = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None
    
    # perform sigmoid operation
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
            #print(loss)
            dz = A - y
            dw = (1 / samples) * np.dot(x.T, dz)
            db = (1 / samples) * np.sum(dz)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # predict positive/negative labels
    def predict(self, x):
        y_hat = np.dot(x, self.weights) + self.bias
        y_predicted = self.sigmoid(y_hat)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

print("-----LOGISTIC REGRESSION-----")

# perform preprocessing using preprocess.py
#preprocess_with_sbert(test_size = 0.2)
X_train, X_test, y_train, y_test  = load_numpy_arrays()
# create instance of model class
model_two = lr()
# perform cross validation and other evaluation using cross_validate.py
cross_validate(X_train, X_test, y_train, y_test, model_two, num_folds=2)