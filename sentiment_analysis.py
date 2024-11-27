import pandas as pd
import re
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Stopword removal, converting uppercase into lower case, and lemmatization
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()



response = []
score = []
files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
for file in files:
    with open('sentiment_labelled_sentences/' + file, 'r') as f:
        lines = f.readlines()
        columns = lines[0].split('\t')

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

print(data.head(10))


vectorizer = TfidfVectorizer() 
vectors = vectorizer.fit_transform(data_without_stopwords)
# print("n_samples: %d, n_features: %d" % vectors.shape)

X_train, X_test, y_train, y_test = train_test_split(vectors, score, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train, y_train)

y_pred = MNB.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:}")
print(f"Precision: {precision:}")
print(f"Recall: {recall:}")
print(f"F1 score: {f1:}")