
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold


def cross_validate(X, X_test, y, y_test, model, num_folds=10):
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
        y_test_pred = model.predict(X_test)
        test_error = 1 - accuracy_score(y_test, y_test_pred)

        #compute average training and validation errors
        avg_train_error = np.mean(training_errors)
        avg_val_error = np.mean(validation_errors)

        #compute and print performance metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        print('--------------------------------------------')
        print(f"Accuracy: {100 * accuracy:.2f}%")
        print(f"Precision: {100 * precision:.2f}%")
        print(f"Recall: {100 * recall:.2f}%")
        print(f"F1 score: {100 * f1:.2f}%")
        
        #print errors
        print('--------------------------------------------')
        print(f"Average Training Error: {100 * avg_train_error:.2f}%")
        print(f"Average Cross-Validation Error: {100 * avg_val_error:.2f}%")
        print(f"Test Error: {100 * test_error:.2f}%")
        print('--------------------------------------------')

        #do bias-variance analysis
        if avg_train_error > 0.1 and abs(avg_train_error - avg_val_error) < 0.05:
            print("High Bias: The model underfits the data.")
        elif avg_train_error < 0.05 and avg_val_error > 0.15:
            print("High Variance: The model overfits the training data.")
        else:
            print("Good balance of bias and variance")
        print('--------------------------------------------')



