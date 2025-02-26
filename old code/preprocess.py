import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_numpy_arrays():
    X_train = np.load('preprocessed_data/train_features.npy')
    X_test = np.load('preprocessed_data/test_features.npy')
    y_train = np.load('preprocessed_data/train_labels.npy')
    y_test = np.load('preprocessed_data/test_labels.npy')
    return X_train, X_test, y_train, y_test

def preprocess_with_sbert(test_size=0.2):
    # Load SBERT model    
    from sentence_transformers import SentenceTransformer
    # sbert_model = SentenceTransformer('all-distilroberta-v1')  # A lightweight SBERT model
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    response = []
    score = []

    # Parse sentences and scores from text files
    files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
    for file in files:
        with open('sentiment_labelled_sentences/' + file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                temp = line.split('\t')
                response.append(temp[0].strip())
                score.append(temp[1].strip())

    # Encode sentences into SBERT embeddings
    sentence_embeddings = sbert_model.encode(response)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(score)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, encoded_labels, test_size=test_size, random_state=42)

    np.save('preprocessed_data/train_features.npy', X_train)
    np.save('preprocessed_data/test_features.npy', X_test)
    np.save('preprocessed_data/train_labels.npy', y_train)
    np.save('preprocessed_data/test_labels.npy', y_test)

def preprocess(test_size=0.2, feature_extraction = 'tf-idf',technique = 'none'):

    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    response = []
    score = []

    # parse sentences and scores from txt files
    files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
    for file in files:
        with open('sentiment_labelled_sentences/' + file, 'r', encoding="utf-8") as f:
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

    #added fole milestone 3
    if(feature_extraction == 'bow'):
        vectorizer = CountVectorizer()
    else:
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

    elif technique == 'variance':
        from sklearn.feature_selection import VarianceThreshold
        # Initialize with a threshold (e.g., remove features with variance < 0.01)
        selector = VarianceThreshold(threshold=0.0001)
        print("Original shape:", vectors.shape)
        vectors = selector.fit_transform(vectors)
        print("New shape:", vectors.shape)

    elif technique == 'sparsity':
        sparsity = np.count_nonzero(vectors, axis=0) / vectors.shape[0]

        # Keep features with sparsity > threshold (e.g., >1% of documents)
        threshold = 0.0005
        keep_indices = np.where(sparsity > threshold)[0]
        print("Original shape:", vectors.shape)
        vectors = vectors[:, keep_indices]
        print("New shape:", vectors.shape)

    elif technique == 'chi-squared':
        from sklearn.feature_selection import SelectKBest, chi2
        # Select top 1000 features based on Chi-Squared
        selector = SelectKBest(chi2, k=2100)
        print("Original shape:", vectors.shape)
        vectors = selector.fit_transform(vectors, encoded_labels)
        print("Shape after dimensionality reduction:", vectors.shape)


    # perform train test split given 'test_size'
    return train_test_split(vectors, encoded_labels, test_size=test_size, random_state=42)