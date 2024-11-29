import pandas as pd
import numpy as np

from preprocess import preprocess

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_score


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.likelihoods = {}
        self.classes = None

    def fit(self, X, y):
        """Fit the model to the training data."""
        self.classes = np.unique(y)
        num_samples, num_features = X.shape

        # Initialize dictionaries
        self.class_priors = {}
        self.likelihoods = {}

        for cls in self.classes:
            # Select all samples of this class
            X_cls = X[y == cls]
            total_samples_cls = X_cls.shape[0]
            
            # Calculate P(Class)
            self.class_priors[cls] = total_samples_cls / num_samples
            
            # Calculate likelihoods P(Feature | Class) with Laplace smoothing
            likelihood_cls = (np.sum(X_cls, axis=0) + 1) / (np.sum(X_cls) + num_features)
            self.likelihoods[cls] = likelihood_cls
        

    def predict(self, X):
        """Predict the class labels for input samples X."""
        predictions = []
        for x in X:
            posteriors = {}
            for cls in self.classes:
                log_prior = np.log(self.class_priors[cls])
                log_likelihood = np.sum(np.log(self.likelihoods[cls]) * x)
                posteriors[cls] = log_prior + log_likelihood

            # Choose the class with the highest posterior probability
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)



X_train, X_test, y_train, y_test  = preprocess(test_size = 0.2)


nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:}")
print(f"Precision: {precision:}")
print(f"Recall: {recall:}")
print(f"F1 score: {f1:}")

# from sklearn.naive_bayes import MultinomialNB
# MNB = MultinomialNB()
# k_folds = KFold(n_splits = 10)
# scores = cross_val_score(MNB, X_train, y_train, cv = k_folds)


# MNB.fit(X_train, y_train)

# y_pred = MNB.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"Accuracy: {accuracy:}")
# print(f"Precision: {precision:}")
# print(f"Recall: {recall:}")
# print(f"F1 score: {f1:}")


#print("Cross Validation Scores: ", scores)
#print("Average CV Score: ", scores.mean())