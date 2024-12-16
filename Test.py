from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np
from Training import load_numpy_arrays,preprocess, SVMClassifier, lr, NaiveBayesClassifier

test_sentences = np.load('preprocessed_data/test_labels_as_sentences.npy')

def test_model(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    incorrectly_predicted_labels = []
    for i in range(len(y_test_pred)):
        if y_test_pred[i] != y_test[i]:
            incorrectly_predicted_labels.append(str(test_sentences[i]))

    #compute and print performance metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"Test Error: {100 * test_error:.2f}%")
    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision: {100 * precision:.2f}%")
    print(f"Recall: {100 * recall:.2f}%")
    print(f"F1 score: {100 * f1:.2f}%")

    return incorrectly_predicted_labels

def load_and_test_model(model_path, X_test, y_test):
    model = None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return test_model(model, X_test, y_test)

X_train, X_test, y_train, y_test = load_numpy_arrays()
print('--------------------------------------------')
print("SVM model test (SBERT, 10 epoch)")
svm_incorrect_labels = load_and_test_model('svm.pkl', X_test, y_test)
print('--------------------------------------------')
print("Logistic Regression model test (SBERT, 5000 epoch)")
lr_incorrect_labels = load_and_test_model('lr.pkl', X_test, y_test)
X_train, X_test, y_train, y_test = preprocess(test_size=0.2, technique='chi-squared')
print('--------------------------------------------')
print("Naive Bayes model test (TF-IDF, Chi Squared)")
nb_incorrect_labels = load_and_test_model('nb.pkl', X_test, y_test)

print('--------------------------------------------')
print("sentences predicted incorrectly by all models:")
for sentence in list(set(svm_incorrect_labels) & set(lr_incorrect_labels) & set(nb_incorrect_labels)):
    print("- " + sentence)