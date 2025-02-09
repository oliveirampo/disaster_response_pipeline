import sys
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
import joblib
from sqlalchemy import create_engine

sys.path.append("../models")

import plot
import train_classifier


app = Flask(__name__)


def tokenize(text):
    """Extracts features from text.

    :param text: (str) Text to have features extracted.
    :return:
        clean_tokens: (list) Cleaned tokens.
    """

    clean_tokens = train_classifier.tokenize(text)
    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    graphs = plot.get_graphs(df)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # for sample message display
    messages_df = df[["message", "genre"]].copy()
    messages_df["shortened"] = messages_df.message.str[:250]
    sample_messages = messages_df.sample(3).values
    
    # render web page with plotly graphs
    return render_template(
        'master.html',
        ids=ids,
        graphJSON=graphJSON,
        sample_messages=sample_messages,
    )


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()