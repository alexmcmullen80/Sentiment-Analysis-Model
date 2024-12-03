import pandas as pd
import numpy as np
import re
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Stopword removal, converting uppercase into lower case, and lemmatization
def preprocess(test_size=0.2, technique = 'none', percentile = 0):

    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    response = []
    score = []
    files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']

    for file in files:
        with open('sentiment_labelled_sentences/' + file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                temp = line.split('\t') 
                response.append(temp[0])
                score.append(temp[1].strip())

    data_without_stopwords = []
    for i in range(0, len(response)):
        doc = re.sub('[^a-zA-Z]', ' ', response[i])
        doc = doc.lower()
        doc = doc.split()
        doc = [lemmatizer.lemmatize(word) for word in doc if not word in set(stopwords)]
        doc = ' '.join(doc)
        data_without_stopwords.append(doc)

    data = pd.DataFrame(list(zip(data_without_stopwords, score))) 
    data.columns = ['response', 'score']

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(score) 
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data_without_stopwords).toarray() 

    if technique == 'information content':
        #--------------------------------------------------------------#
        #removing reviews with the lowest tf-idf sum (low information content)
        vectors = vectorizer.fit_transform(data_without_stopwords).toarray()
        document_sums = np.sum(vectors, axis=1)  # Use toarray() if sparse
        threshold = np.percentile(document_sums, percentile)  # Remove the lowest 10%
        vectors = vectors[document_sums > threshold]
        encoded_labels = np.array(encoded_labels)[document_sums > threshold]
        #--------------------------------------------------------------#
    elif technique == 'tf-idf variance':
        #--------------------------------------------------------------#
        #removing reviews with low tf-idf variance
        vectors = vectorizer.fit_transform(data_without_stopwords).toarray()
        document_variances = np.var(vectors, axis=1)  # Use toarray() if sparse
        threshold = np.percentile(document_variances, percentile)  # Remove the lowest 10%
        vectors = vectors[document_variances > threshold]
        encoded_labels = np.array(encoded_labels)[document_variances > threshold]
        #--------------------------------------------------------------#
    elif technique == 'cosine similarity':
        #--------------------------------------------------------------#
        #remove reviews with low cosine similarity to others (outliers)
        all_vectors = vectorizer.fit_transform(data_without_stopwords)
        threshold = 1/100 * percentile
        similarity_matrix = cosine_similarity(all_vectors)
        mean_similarities = np.mean(similarity_matrix, axis=1)

        filtered_data_and_labels = [
            (data_without_stopwords[i], score[i]) 
            for i in range(len(data_without_stopwords)) 
            if mean_similarities[i] > threshold
        ]

        # Separate the filtered data and labels
        data_without_stopwords, filtered_labels = zip(*filtered_data_and_labels)

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(filtered_labels) 
        vectors = vectorizer.fit_transform(data_without_stopwords).toarray()
        #---------------------------------------------------------------#

    #vectors = vectorizer.fit_transform(data_without_stopwords).toarray()
    return train_test_split(vectors, encoded_labels, test_size=test_size, random_state=42)
    #return train_test_split(vectors, score, test_size=test_size, random_state=42)