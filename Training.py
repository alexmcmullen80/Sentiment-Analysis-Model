import numpy as np
import pandas as pd
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pickle

def preprocess(test_size=0.2, technique = 'none', percentile = 0):

    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    response = []
    score = []

    # parse sentences and scores from txt files
    files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
    for file in files:
        with open('sentiment_labelled_sentences/' + file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                temp = line.split('\t') 
                response.append(temp[0])
                score.append(temp[1].strip())

    # sentences preprocessing
    data_without_stopwords = []
    for i in range(0, len(response)):

        # convert uppercase into lowercase
        doc = re.sub('[^a-zA-Z]', ' ', response[i])
        doc = doc.lower()
        doc = doc.split()

        # stopword removal and lemmatization
        doc = [lemmatizer.lemmatize(word) for word in doc if not word in set(stopwords)]
        doc = ' '.join(doc)
        data_without_stopwords.append(doc)

    data = pd.DataFrame(list(zip(data_without_stopwords, score))) 
    data.columns = ['response', 'score']

    # vectorization
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(score) 
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data_without_stopwords).toarray() 

    # additional preprocess techniques given 'technique'
    # removing reviews with the lowest tf-idf sum (low information content)
    if technique == 'information content':
        vectors = vectorizer.fit_transform(data_without_stopwords).toarray()
        document_sums = np.sum(vectors, axis=1)
        threshold = np.percentile(document_sums, percentile)
        vectors = vectors[document_sums > threshold]
        encoded_labels = np.array(encoded_labels)[document_sums > threshold]
    
    #removing reviews with low tf-idf variance
    elif technique == 'tf-idf variance':
        vectors = vectorizer.fit_transform(data_without_stopwords).toarray()
        document_variances = np.var(vectors, axis=1)
        threshold = np.percentile(document_variances, percentile)
        vectors = vectors[document_variances > threshold]
        encoded_labels = np.array(encoded_labels)[document_variances > threshold]

    # perform train test split given 'test_size'
    return train_test_split(vectors, encoded_labels, test_size=test_size, random_state=42)

def cross_validate(X, y, model, num_folds=10):
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        training_errors = []
        validation_errors = []

        for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print("Fold {}".format(i+1))

            #split the data into train and validation sets
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            #initialize and train model
            first_model = model
            first_model.fit(X_train, y_train)

            #compute training errors (1-accuracy)
            y_train_pred = first_model.predict(X_train)
            train_error = 1 - accuracy_score(y_train, y_train_pred)
            training_errors.append(train_error)

            #compute validation errors (1-accuracy)
            y_val_pred = first_model.predict(X_val)
            val_error = 1 - accuracy_score(y_val, y_val_pred)
            validation_errors.append(val_error)

        #build new model to train on whole train set and predict on test set
        model_final = model
        model_final.fit(X, y)

        #compute average training and validation errors
        avg_train_error = np.mean(training_errors)
        avg_val_error = np.mean(validation_errors)

        #print errors
        print('--------------------------------------------')
        print(f"Average Training Error: {100 * avg_train_error:.2f}%")
        print(f"Average Cross-Validation Error: {100 * avg_val_error:.2f}%")
        print('--------------------------------------------')

        #do bias-variance analysis
        if avg_train_error > 0.1 and abs(avg_train_error - avg_val_error) < 0.05:
            print("High Bias: The model underfits the data.")
        elif avg_train_error < 0.05 and avg_val_error > 0.15:
            print("High Variance: The model overfits the training data.")
        else:
            print("Good balance of bias and variance")
        print('--------------------------------------------')

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
            features, output = shuffle(X, y, random_state=42)

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)


    def predict(self, X):
        return [1 if pred > 0 else 0 for pred in np.dot(X, self.weights)]

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
            if (e % 1000 == 0):
                print(" Epoch: " + str(e) + " Loss: " + loss)
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

def main():
    X_train, X_test, y_train, y_test = preprocess(test_size=0.2)

    # SVM training and validation
    C = 1
    learning_rate = 0.0005
    epoch = 100

    # Create SVM classifier
    svm_classifier = SVMClassifier(learning_rate=learning_rate,epoch=epoch, c_value=C)

    print("Training svm model")
    # train model and cross validate
    cross_validate(X_train, y_train, svm_classifier, num_folds=10)
    with open('svm.pkl', 'wb') as f:
        pickle.dump(svm_classifier, f)

    # Logistic regression training and validation

    # create instance of model class
    lr_classifier = lr()
    print("Training logistic regression model")
    print("warning, num_folds set to 10. This may take a while... (feel free to reduce number of folds)")
    # perform cross validation and other evaluation using cross_validate.py
    cross_validate(X_train, y_train, lr_classifier, num_folds=10)
    with open('lr.pkl', 'wb') as f:
        pickle.dump(lr_classifier, f)

    # Naive Bayes training and validation

    #initialize model
    nb_classifier = NaiveBayesClassifier()
    print("Training naive bayes model")
    #call cross validation function
    cross_validate(X_train, y_train, nb_classifier, num_folds=10)
    with open('nb.pkl', 'wb') as f:
        pickle.dump(nb_classifier, f)

if __name__ == '__main__':
    main()