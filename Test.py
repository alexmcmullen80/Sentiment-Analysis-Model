from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from Training import load_numpy_arrays,preprocess, SVMClassifier, lr, NaiveBayesClassifier


def test_model(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    #compute and print performance metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    print('--------------------------------------------')
    print(f"Test Error: {100 * test_error:.2f}%")
    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision: {100 * precision:.2f}%")
    print(f"Recall: {100 * recall:.2f}%")
    print(f"F1 score: {100 * f1:.2f}%")

def load_and_test_model(model_path, X_test, y_test):
    model = None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    test_model(model, X_test, y_test)

X_train, X_test, y_train, y_test = load_numpy_arrays()
print("SVM model test")
load_and_test_model('svm.pkl', X_test, y_test)
print("Logistic Regression model test")
load_and_test_model('lr.pkl', X_test, y_test)
X_train, X_test, y_train, y_test = preprocess(test_size=0.2)
print("Naive Bayes model test")
load_and_test_model('nb.pkl', X_test, y_test)