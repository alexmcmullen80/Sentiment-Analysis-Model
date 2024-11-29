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

        #initialize
        self.class_priors = {}
        self.likelihoods = {}

        for cls in self.classes:
            #select all samples of this class
            X_cls = X[y == cls]
            total_samples_cls = X_cls.shape[0]
            
            #calculate P(Class)
            self.class_priors[cls] = total_samples_cls / num_samples
            
            #calculate likelihoods P(Feature | Class) with Laplace smoothing
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

            #choose class with highest posterior probability
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)
    


def cross_validate(X, y, num_folds):
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for train_idx, val_idx in kfold.split(X):
            #split the data into train and validation sets
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            #initialize and train classifier
            cv_nb = NaiveBayesClassifier()
            cv_nb.fit(X_train, y_train)

            #make predictions on the validation data
            y_pred = cv_nb.predict(X_val)

            #append model performances to list
            accuracies.append(accuracy_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred, average='weighted'))
            recalls.append(recall_score(y_val, y_pred, average='weighted'))
            f1_scores.append(f1_score(y_val, y_pred, average='weighted'))

        #print average metrics across all folds
        print(f"CV Average Accuracy: {np.mean(accuracies):.3f}")
        print(f"CV Average Precision: {np.mean(precisions):.3f}")
        print(f"CV Average Recall: {np.mean(recalls):.3f}")
        print(f"CV Average F1 Score: {np.mean(f1_scores):.3f}")

X_train, X_test, y_train, y_test  = preprocess(test_size = 0.2)

cross_validate(X_train, y_train, 10)
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

#predict
y_pred = nb_classifier.predict(X_test)

#evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:}")
print(f"Precision: {precision:}")
print(f"Recall: {recall:}")
print(f"F1 score: {f1:}")

