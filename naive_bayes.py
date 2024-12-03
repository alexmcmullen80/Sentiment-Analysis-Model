
import numpy as np
from preprocess import preprocess
from cross_validate import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




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

#split into train and test
X_train, X_test, y_train, y_test  = preprocess(test_size = 0.2, technique = 'tf-idf variance', percentile = 10)

#initialize model
nb_classifier = NaiveBayesClassifier()
#call cross validation function
cross_validate(X_train, X_test, y_train, y_test, nb_classifier)

# #train
# nb_classifier.fit(X_train, y_train)
# #predict
# y_pred = nb_classifier.predict(X_test)

# #evaluate performance
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"Accuracy: {accuracy:}")
# print(f"Precision: {precision:}")
# print(f"Recall: {recall:}")
# print(f"F1 score: {f1:}")

