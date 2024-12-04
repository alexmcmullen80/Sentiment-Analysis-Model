import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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