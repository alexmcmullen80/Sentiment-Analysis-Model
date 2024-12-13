import numpy as np
from preprocess import preprocess
from cross_validate import cross_validate

# NEW PER MILESTONE 3 - GRAPHS
import matplotlib.pyplot as plt

class lr():
    
    # initialize variables
    # NEW PER MILESTONE 3 - IMPROVED MODEL PARAMETERS
    def __init__(self, learning_rate=0.15, epoch=8000):
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

        # NEW PER MILESTONE 3 - EARLY STOPPING AND GRAPHS
        stoppingepoch = 0
        prevloss = 0
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches((15,8))
        train_plots=[]
        epochs=[]

        for e in range(self.epoch):
            A = self.feed_forward(x)
            loss = self.loss(y,A)
            dz = A - y
            dw = (1 / samples) * np.dot(x.T, dz)
            db = (1 / samples) * np.sum(dz)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # NEW PER MILESTONE 3 - EARLY STOPPING AND GRAPHS
            # plot every 500 epoch
            if e%500 == 0:
                train_plots.append(loss)
                epochs.append(e)

            # early stopping with a threshold of 0.5 and patience value of 100
            if e > 100 and stoppingepoch == 0 and loss < 0.45 and prevloss - loss < 1/10000:
                stoppingepoch = e
                print("Epoch: {}".format(e))
                print("Previous Loss: {}".format(prevloss))
                print("Loss: {}".format(loss))
                break
            
            if e % 100 == 0:
                prevloss = loss
        # plot points
        self.ax.plot(epochs,train_plots, color = "red", label = "Training Loss")
        # set the x-labels
        self.ax.set_xlabel("Epoch")
        # set the y-labels
        self.ax.set_ylabel("Loss")
        self.ax.set_ylim([0,1])
        # set the title
        self.ax.set_title("Loss vs Epoch for Training the Logistic Regression Model")
        plt.show()

    # predict positive/negative labels
    def predict(self, x):
        y_hat = np.dot(x, self.weights) + self.bias
        y_predicted = self.sigmoid(y_hat)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

print("-----LOGISTIC REGRESSION-----")

# perform preprocessing using preprocess.py
X_train, X_test, y_train, y_test  = preprocess(test_size = 0.2)
# create instance of model class
model_two = lr()
# perform cross validation and other evaluation using cross_validate.py
cross_validate(X_train, X_test, y_train, y_test, model_two, num_folds=2)