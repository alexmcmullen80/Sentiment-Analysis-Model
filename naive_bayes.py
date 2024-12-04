
import numpy as np
from preprocess import preprocess
from cross_validate import cross_validate



class NaiveBayesClassifier:
    def __init__(self):
        #initialize
        self.class_priors = {}
        self.likelihoods = {}
        self.classes = None

    def fit(self, X, y):
        #get each class
        self.classes = np.unique(y)
        num_samples, num_features = X.shape

        #initialize
        self.priors = {}
        self.likelihoods = {}

        for unique_class in self.classes:
            #select all samples of this class
            X_class = X[y == unique_class]
            total_samples_cls = X_class.shape[0]
            
            #calculate P(class) aka priors
            self.priors[unique_class] = total_samples_cls / num_samples
            
            #calculate P(feature|class) aka likelihoods with smoothening
            self.likelihoods[unique_class] = (np.sum(X_class, axis=0) + 1) / (np.sum(X_class) + num_features)
        

    def predict(self, X_test):
        #initialize empty predictions
        predictions = []

        #loop through test set
        for data in X_test:
            #initialize empty posteriors
            posteriors = {}

            #compute log priors, log likelihoods and posteriors for each class
            for unique_class in self.classes:
                log_prior = np.log(self.priors[unique_class])
                log_likelihood = np.sum(np.log(self.likelihoods[unique_class]) * data)
                posteriors[unique_class] = log_prior + log_likelihood

            #choose class with highest posterior probability and append it to predictions
            predictions.append(max(posteriors, key=posteriors.get))

        return np.array(predictions)

#split into train and test
#X_train, X_test, y_train, y_test  = preprocess(test_size = 0.2, technique = 'tf-idf variance', percentile = 10)
X_train, X_test, y_train, y_test  = preprocess(test_size = 0.2)

#initialize model
nb_classifier = NaiveBayesClassifier()
#call cross validation function
cross_validate(X_train, X_test, y_train, y_test, nb_classifier)



