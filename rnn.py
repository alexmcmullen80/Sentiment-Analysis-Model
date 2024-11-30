from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, GRU, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os



# Paths and hyperparameters
glove_file_path = 'glove/glove.6B.300d.txt'  # Set the path to your GloVe file
max_words = 10000  # Number of words to keep based on frequency
max_sequence_length = 300  # Max sequence length after padding

# Step 1: Load the GloVe embeddings
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index

# Load GloVe embeddings
embeddings_index = load_glove_embeddings(glove_file_path)

# Step 2: Prepare the text data
files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
data = []

for file in files:
    with open('sentiment_labelled_sentences/' + file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split('\t')
            data.append(temp)  # store as tuple of (text, label)

df = pd.DataFrame(data, columns=["text", "label"])

# Tokenizer setup
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=max_sequence_length)

# Label encode the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Step 3: Create the embedding matrix
embedding_dim = 300  # Using 100-dimensional embeddings from GloVe
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Step 4: Build the LSTM model with GloVe embeddings
class SimpleTextRNN:
    def __init__(self):
        self.model = None

    def build_model(self, hidden_units, output_units, activation):
        self.model = Sequential()
        
        # Embedding Layer with GloVe embeddings
        self.model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, 
                                 input_length=max_sequence_length, weights=[embedding_matrix], trainable=True))
        
        # LSTM Layer
        self.model.add(Bidirectional(GRU(hidden_units, activation=activation[0], return_sequences=False, kernel_regularizer=l2(0.015))))
        #self.model.add(BatchNormalization())
        
        # Dropout Layer
        self.model.add(Dropout(0.7))
        
        # Output Dense Layer
        self.model.add(Dense(output_units, activation=activation[1]))
        
        # Compile the model
        optimizer = Adam(learning_rate=0.0001)
        self.model.compile(loss=SparseCategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
        return self.model

    def train_model(self, X_train, y_train, epochs=30, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.2, callbacks=[early_stopping])
        #history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.2)
        return history

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy

    def predict(self, X_input):
        return self.model.predict(X_input)


# Step 5: Create and compile the LSTM model
lstm_model = SimpleTextRNN()
lstm_model.build_model(hidden_units=64, output_units=2, activation=['tanh', 'softmax'])

# Step 6: Train the model
history = lstm_model.train_model(X_train, y_train)

# Step 7: Evaluate the model
loss, accuracy = lstm_model.evaluate_model(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 8: Plot the training and validation accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
