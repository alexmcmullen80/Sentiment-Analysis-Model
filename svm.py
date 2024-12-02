from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from preprocess import preprocess

class SVMClassifier:
    def __init__(self, learning_rate=0.001, c_value=0.001, epoch=1000):
        self.learning_rate = learning_rate
        self.C = c_value
        self.epoch = epoch
        self.weights = None

    def compute_gradient(self, X, Y):
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y * np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y * X_[0])

        return total_distance
    
    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)
        self.weights = np.zeros(X.shape[1])

        for epoch in range(self.epoch):
            features, output = shuffle(X, y)

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)


    def predict(self, X):
        return [1 if pred > 0 else 0 for pred in np.dot(X, self.weights)]

# Load data
X_train, X_test, y_train, y_test = preprocess(test_size=0.2)

#y_train = np.where(y_train == 0, -1, 1)

C = 1
learning_rate = 0.0001
epoch = 100

# Create SVM classifier
svm_classifier = SVMClassifier(learning_rate=learning_rate,epoch=epoch, c_value=C)

# Train model
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 score: {f1:.3f}")