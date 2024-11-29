# svm.py
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess

class SVMClassifier:
    def __init__(self):
        self.model = svm.SVC()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# Load data
X_train, X_test, y_train, y_test = preprocess(test_size=0.2)

# Create SVM classifier
svm_classifier = SVMClassifier()

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