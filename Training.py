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

# NEW PER MILESTONE 3 - GRAPHS
import matplotlib.pyplot as plt

# NEW PER MILESTONE 3
# load features as npy files saved from pretrained model
def load_numpy_arrays():
    X_train = np.load('preprocessed_data/train_features.npy')
    X_test = np.load('preprocessed_data/test_features.npy')
    y_train = np.load('preprocessed_data/train_labels.npy')
    y_test = np.load('preprocessed_data/test_labels.npy')
    return X_train, X_test, y_train, y_test

# NEW PER MILESTONE 3
# perform preprocess using SBERT, save features as npy files
def preprocess_with_sbert(test_size=0.2):
    #import SBERT model
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    
    response = []
    score = []

    # parse sentences and scores from txt files
    files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
    for file in files:
        with open('sentiment_labelled_sentences/' + file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                temp = line.split('\t')
                response.append(temp[0].strip())
                score.append(temp[1].strip())

    # sentences preprocessing
    sentence_embeddings = sbert_model.encode(response)

    # label encoding
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(score)

    # perform train test split given 'test_size'
    X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, encoded_labels, test_size=test_size, random_state=42)

    # save features and labels as numpy files
    np.save('preprocessed_data/train_features.npy', X_train)
    np.save('preprocessed_data/test_features.npy', X_test)
    np.save('preprocessed_data/train_labels.npy', y_train)
    np.save('preprocessed_data/test_labels.npy', y_test)

def preprocess(test_size=0.2, technique = 'none'):

    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    response = []
    score = []

    # parse sentences and scores from txt files
    files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
    for file in files:
        with open('sentiment_labelled_sentences/' + file, 'r', encoding='utf-8') as f:
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
    # NEW PER MILESTONE 3
    # selecting the best features using the Chi-Squared test
    if technique == 'chi-squared':
        from sklearn.feature_selection import SelectKBest, chi2
        # Select top 1000 features based on Chi-Squared
        selector = SelectKBest(chi2, k=2100)
        #print("Original shape:", vectors.shape)
        vectors = selector.fit_transform(vectors, encoded_labels)
        #print("Shape after dimensionality reduction:", vectors.shape)

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
        print("full train")
        model_final = model
        if isinstance(model_final,lr) or isinstance(model_final,SVMClassifier):
            model_final.fit(X, y, True)
        else:
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
    
    def compute_loss(self, X, Y):
        X_ = np.array([X])
        return np.sum(1 - (Y * np.dot(X_,self.weights)))

    def fit(self, X, y, makeGraph=False):
        y = np.where(y == 0, -1, 1)
        self.weights = np.zeros(X.shape[1])
        losses = []

        for epoch in range(self.epoch):
            features, output = shuffle(X, y, random_state=42)

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)
            
            loss = self.compute_loss(features, output)
            losses.append(loss)
            if epoch > 0 and abs(losses[epoch] - losses[epoch - 1]) < 0.001:
                print(f"early stopping at epoch {epoch}")
                break

            print(f"epoch: {epoch}, loss: {loss}")

        if makeGraph:
            self.fig, self.ax = plt.subplots()
            self.fig.set_size_inches((15,8))
            # plot points
            self.ax.plot([i+1 for i in range(self.epoch)], losses, color = "red", label = "Training Loss")
            # set the x-labels
            self.ax.set_xlabel("Epoch")
            # set the y-labels
            self.ax.set_ylabel("Loss")
            # set the title
            self.ax.set_title("Loss vs Epoch for Training the SVM Model")
            plt.show()

    def predict(self, X):
        return [1 if pred > 0 else 0 for pred in np.dot(X, self.weights)]

class lr():
    
    # initialize variables
    def __init__(self, learning_rate=0.1, epoch=5000):
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
    def fit(self, x, y, makeGraph=False):
        samples, features = x.shape
        self.weights = np.zeros(features)
        self.bias = 0

        # NEW PER MILESTONE 3 - EARLY STOPPING AND GRAPHS
        stoppingepoch = 0
        prevloss = 0
        if makeGraph:
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
            # plot every 100 epoch
            if makeGraph and e%100 == 0:
                train_plots.append(loss)
                epochs.append(e)

            # early stopping with a threshold of 1/10000
            if e > 0 and stoppingepoch == 0 and prevloss - loss < 1/10000:
                stoppingepoch = e
                train_plots.append(loss)
                epochs.append(e)
                print("Stopping Epoch: {}".format(e))
                print("Previous Loss: {}".format(prevloss))
                print("Loss: {}".format(loss))
                break
            prevloss = loss

        if makeGraph:
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

def main():
    #preprocess_with_sbert(test_size = 0.2)
    X_train, X_test, y_train, y_test = load_numpy_arrays()

    # SVM training and validation
    # initialize model
    C = 1
    learning_rate = 0.0005
    epoch = 10 # decided based on early stopping result
    svm_classifier = SVMClassifier(learning_rate=learning_rate,epoch=epoch, c_value=C)
    print("Training svm model")

    # train model and cross validate
    cross_validate(X_train, y_train, svm_classifier, num_folds=10)

    # save as pickle
    with open('svm.pkl', 'wb') as f:
        pickle.dump(svm_classifier, f)

    # Logistic Regression training and validation
    # initialize model
    lr_classifier = lr()
    print("Training logistic regression model")

    # perform cross validation and other evaluation using cross_validate.py
    cross_validate(X_train, y_train, lr_classifier, num_folds=10)

    # save as pickle
    with open('lr.pkl', 'wb') as f:
        pickle.dump(lr_classifier, f)

    # Naive Bayes training and validation
    # do not use SBERT for NB
    X_train, X_test, y_train, y_test = preprocess(test_size=0.2, technique = 'chi-squared')

    # initialize model
    nb_classifier = NaiveBayesClassifier()
    print("Training naive bayes model")

    # call cross validation function
    cross_validate(X_train, y_train, nb_classifier, num_folds=10)

    # save as pickle
    with open('nb.pkl', 'wb') as f:
        pickle.dump(nb_classifier, f)

if __name__ == '__main__':
    main()