import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """ Load data from given database path. """
    engine_path = "sqlite:///{}".format(database_filepath)
    engine = create_engine(engine_path)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df[df.columns[4:]]
    return X, y, y.columns.tolist()


def tokenize(text):
    """ Tokenize text into a list of stemmatized words. """
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words("english")]
    
    
def build_model():
    """ Build an AdaBoostClassifier model. """
    
    pipeline = Pipeline([
    ('vec', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    random_state = 42
    parameters = {'classifier__estimator__n_estimators':[20, 50, 100], 
              'classifier__estimator__learning_rate':[0.5, 1.0], 
              'classifier__estimator__algorithm': ('SAMME', 'SAMME.R'),
              'classifier__estimator__random_state': [random_state]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """ Evaluate the performance of the model on the testing dataset. """
    y_preds = model.predict(X_test)
    
    average_f1_score, average_precision, average_recall = [], [], []
    for idx, category in enumerate(category_names):
        print("Category: " + category + "\n")
        f1 = f1_score(y_test.values[:, idx], y_preds[:, idx], average='weighted')
        average_f1_score.append(f1)
        precision = precision_score(y_test.values[:, idx], y_preds[:, idx], average='weighted')
        average_precision.append(precision)
        recall = recall_score(y_test.values[:, idx], y_preds[:, idx], average='weighted')
        average_recall.append(recall)
        print("F1 score: {}, Precision: {}, Recall: {}".format(f1, precision, recall))
        print("\n")
    
    print("Average F1 score: {}".format(np.mean(average_f1_score)))
    print("Average precision score: {}".format(np.mean(average_precision)))
    print("Average recall score: {}".format(np.mean(average_recall)))
        

def save_model(model, model_filepath):
    """ Save the model as a pickle file. """
    with open(model_filepath, 'wb') as pickle_model:
        pickle.dump(model, pickle_model)


def main():
    """ Main function that trains, evaluates, and saves the model. """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
