import pickle
import sys
import time

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def load_data(data_path):
    """Load DataFrame previously engineered for machine learning. """
    df = pd.read_csv(data_path)
    X = df.drop('successful_offer', axis=1)
    y = df['successful_offer']
    return X, y


def train_and_test_model(model, X_train, X_test, y_train, y_test, print_time=True):
    """Train and test a machine learning model"""
    start = time.time()
    model = model.fit(X_train, y_train)
    end = time.time()
    if print_time:
        print('The model took {:.2f} seconds to train.'.format(end - start))

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(train_predictions, y_train)
    test_accuracy = accuracy_score(test_predictions, y_test)

    print(f'The train accuracy of the model is {train_accuracy}')
    print(f'The test accuracy of the model is {test_accuracy}')

    return model


def build_model(random_state=42, cross_validation_folds=3):
    """ Build an GradientBoostingClassifier model. """

    gbc = GradientBoostingClassifier(random_state=random_state)

    parameters = {'loss': ['deviance', 'exponential'],
                  'learning_rate': [0.1, 0.2, 0.3],
                  'n_estimators': [20, 50, 100, 150],
                  'max_depth': [3, 4, 5],
                  'min_samples_split': [2, 3, 4]}

    grid_object = GridSearchCV(gbc, parameters, cv=cross_validation_folds)

    return grid_object


def save_model(model, model_filepath):
    """ Save the model as a pickle file. """
    with open(model_filepath, 'wb') as pickle_model:
        pickle.dump(model, pickle_model)


def main():
    """ Main function that trains, evaluates, and saves the model. """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    FILEPATH: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        train_and_test_model(model, X_train, X_test, y_train, y_test, print_time=False)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the cleaned data CSV  '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/cleaned_data.csv classifier.pkl')


if __name__ == '__main__':
    main()
