from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
import pandas as pd
import pickle
import sys
import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """Loads dataset from database with read_sql_table

    :param database_filepath: (str) Path to SQL database.
    :return:
        X: (pandas DataFrame) Features.
        Y: (pandas DataFrame) Target Variables.
        category_names: (list) Possible categories of variables.
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM messages", engine)

    X = df['message']
    Y = df.iloc[:, 4:]  # Ignore the first 4 columns from df: [id, message, original, genre]
    categories = Y.columns.tolist()

    return X, Y, categories


def tokenize(text):
    """Extracts features from text.

    :param text: (str) Text to have heatures extracted.
    :return:
        clean_tokens: (list) Cleaned tokens.
    """

    # replace any url to 'url_place_holder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'url_place_holder')

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds machine learning pipeline

    :return:
        cv: (estimator) GridSearchCV with Pipeline.
    """
    pipeline = Pipeline(
        [
            (
                'count_vectorizer', CountVectorizer(tokenizer=tokenize),
            ),
            (
                'tfidf_transformer', TfidfTransformer()),
            (
                'classifier', MultiOutputClassifier(AdaBoostClassifier()),
            ),
        ]
    )

    # parameters_grid = {'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
    #                    'classifier__estimator__n_estimators': [10, 20, 40]}
    parameters_grid = {'classifier__estimator__learning_rate': [0.01, 0.05],
                       'classifier__estimator__n_estimators': [10, 20]}

    cv = GridSearchCV(pipeline, cv=5, param_grid=parameters_grid, scoring='f1_micro', n_jobs=-1)

    return cv


def build_simple_model():
    """Builds machine learning pipeline

    :return:
        pipeline: (estimator)
    """

    pipeline = Pipeline(
        [
            (
                "vect",
                CountVectorizer(
                    tokenizer=tokenize,
                    max_df=1.0,
                    max_features=None,
                    ngram_range=(1, 2),
                ),
            ),
            ("tfidf", TfidfTransformer(use_idf=True)),
            (
                "clf",
                MultiOutputClassifier(
                    SGDClassifier(
                        loss="hinge",
                        alpha=0.00005,
                        tol=0.01,
                        n_iter_no_change=5,
                        penalty="l2",
                    )
                ),
            ),
        ]
    )
    return pipeline


def build_model_with_feature_union():
    """Builds machine learning pipeline

    :return:
        cv: (estimator) GridSearchCV with Pipeline.
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ()  # other pipeline

        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters_grid = {'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
                       'classifier__estimator__n_estimators': [10, 20, 40]}

    cv = GridSearchCV(pipeline, param_grid=parameters_grid, scoring='f1_micro', n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Report the f1 score, precision and recall for each output category of the dataset.

    :param model: (estimator)
    :param X_test: (pandas DataFrame) Features.
    :param Y_test: (pandas DataFrame) Target Variables.
    :param category_names: (list) Possible categories of variables.
    """

    y_pred = model.predict(X_test)

    # Calculate the accuracy for each output category of the dataset.
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))


def save_model(model, model_filepath):
    """Exports model as a pickle file.

    :param model: (estimator)
    :param model_filepath: (str) Path to where model will be saved.
    :return:
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_simple_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py data/disaster_response.db classifier.pkl')


if __name__ == '__main__':
    main()
