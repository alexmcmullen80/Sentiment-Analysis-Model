from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, GRU, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os


max_words = 5000  #number of words to keep based on frequency
max_sequence_length = 50  #max sequence length after padding

#load GloVe embeddings
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

embeddings_index = load_glove_embeddings('glove/glove.6B.300d.txt')

#prepare the sentiment data
files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
data = []

for file in files:
    with open('sentiment_labelled_sentences/' + file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split('\t')
            data.append(temp)  #store as tuple of (text, label)

df = pd.DataFrame(data, columns=["text", "label"])

#tokenizer setup
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=max_sequence_length)

#encode the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

#split into train and test set (same state as preprocessing)
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

#create the embedding matrix
embedding_dim = 300  #using GloVe 300 dimensional embeddings
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

#build the RNN
class SimpleTextRNN:
    def __init__(self):
        self.model = None

    def build_model(self, hidden_units, output_units, activation):
        self.model = Sequential()
        
        #embedding layer with GloVe embeddings
        #self.model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length, weights=[embedding_matrix], trainable=True))
        self.model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True))
        
        #GRU layer to prevent vanishing/exploding gradient
        #use bidirectional and l2 regularization to improve performance
        self.model.add(Bidirectional(GRU(hidden_units, activation=activation[0], return_sequences=False, kernel_regularizer=l2(0.015))))
    
        
        #dropout Layer
        self.model.add(Dropout(0.7))
        
        #dense layer
        self.model.add(Dense(output_units, activation=activation[1]))
        
        #use Adam function as optimizer and SparseCategoricalCrossentropy as they are optimal for text processing with binary labels
        optimizer = Adam(learning_rate=0.0001)
        self.model.compile(loss=SparseCategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
        return self.model

    #train model with early stopping (verbose=2 displays train and validation accuracy and loss for each epoch)
    def train_model(self, X_train, y_train, epochs=30, batch_size=32):
        #check if validation accuracy has not gotten better for 3 straight epochs
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.2, callbacks=[early_stopping])
        #return history

    #evaluate model with accuracy and loss 
    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy

    #predict labels
    def predict(self, X_input):
        return np.argmax(self.model.predict(X_input), axis = 1)


#initialize RNN
rnn_model = SimpleTextRNN()
rnn_model.build_model(hidden_units=64, output_units=2, activation=['tanh', 'softmax'])
#lstm_model.build_model(hidden_units=64, output_units=2, activation=['relu', 'softmax'])

#train
rnn_model.train_model(X_train, y_train)

#evaluate and print test accuracy
#loss, accuracy = lstm_model.evaluate_model(X_test, y_test)
y_test_pred = rnn_model.predict(X_test)

#compute and print accuracy, precision, recall, f1 score
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
print(f"Accuracy: {accuracy:}")
print(f"Precision: {precision:}")
print(f"Recall: {recall:}")
print(f"F1 score: {f1:}")
print('--------------------------------------------')

