import json

import pandas as pd
import plotly
from flask import Flask
from flask import render_template
from plotly.graph_objs import Bar
from sklearn.externals import joblib

app = Flask(__name__)

# Load data
df = pd.read_csv('../data/data_for_analysis.csv')

# load model
model = joblib.load("../models/classifier.pkl")


def age_interval(age):
    """Put age into an interval."""
    if age < 10:
        interval = '[0, 10['
    elif 10 <= age < 20:
        interval = '[10, 20['
    elif 20 <= age < 30:
        interval = '[20, 30['
    elif 30 <= age < 40:
        interval = '[30, 40['
    elif 40 <= age < 50:
        interval = '[40, 50['
    elif 50 <= age < 60:
        interval = '[50, 60['
    elif 60 <= age < 70:
        interval = '[60, 70['
    elif age >= 70:
        interval = '[70+, ]'
    else:
        raise ValueError(f"Unknow value for age : {age}")
    return interval


def income_interval(income):
    """Put income into an interval."""
    income = income / 1000
    if 30 <= income < 40:
        interval = '[30, 40['
    elif 40 <= income < 50:
        interval = '[40, 50['
    elif 50 <= income < 60:
        interval = '[50, 60['
    elif 60 <= income < 70:
        interval = '[60, 70['
    elif 70 <= income < 80:
        interval = '[70, 80['
    elif 80 <= income < 90:
        interval = '[80, 90['
    elif 90 <= income < 100:
        interval = '[90, 100['
    elif income >= 100:
        interval = '[100+, ]'
    else:
        raise ValueError(f"Unknow value for income : {income}")
    return interval


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    offer_types = df.groupby('offer_type')['successful_offer'].mean()
    offer_types_df = pd.DataFrame({'offer_type': offer_types.index, '% of success': offer_types.values})

    df['age_interval'] = df['age'].apply(age_interval)
    grouped_ages = df.groupby('age_interval')['successful_offer'].mean()
    grouped_ages_df = pd.DataFrame({'age': grouped_ages.index, '% of success': grouped_ages.values})

    difficulties = df.groupby('difficulty')['successful_offer'].mean()
    difficulties_df = pd.DataFrame({'difficulty': difficulties.index, '% of success': difficulties.values})

    df['income_interval'] = df['income'].apply(income_interval)
    incomes = df.groupby('income_interval')['successful_offer'].mean()
    incomes_df = pd.DataFrame({'income': incomes.index, '% of success': incomes.values})

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=offer_types_df['offer_type'],
                    y=offer_types_df['% of success']
                )
            ],

            'layout': {
                'title': 'Probability of success by offer type',
                'yaxis': {
                    'title': "% of success"
                },
                'xaxis': {
                    'title': "Offer type"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=grouped_ages_df['age'],
                    y=grouped_ages_df['% of success']
                )
            ],

            'layout': {
                'title': 'Probability of success by age',
                'yaxis': {
                    'title': "% of success"
                },
                'xaxis': {
                    'title': "Age"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=difficulties_df['difficulty'],
                    y=difficulties_df['% of success']
                )
            ],

            'layout': {
                'title': 'Probability of success by offer difficulty',
                'yaxis': {
                    'title': "% of success"
                },
                'xaxis': {
                    'title': "Difficulty"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=incomes_df['income'],
                    y=incomes_df['% of success']
                )
            ],

            'layout': {
                'title': 'Probability of success by customer income',
                'yaxis': {
                    'title': "% of success"
                },
                'xaxis': {
                    'title': "Income (in thousands USD)"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


@app.route('/machine-learning/')
def machine_learning():
    """Machine learning page."""
    return render_template('machine-learning.html')



#
# # web page that handles user query and displays model results
# @app.route('/go')
# def go():
#     # save user input in query
#     query = request.args.get('query', '')
#
#     # use model to predict classification for query
#     classification_labels = model.predict([query])[0]
#     classification_results = dict(zip(df.columns[4:], classification_labels))
#
#     # This will render the go.html Please see that file.
#     return render_template(
#         'go.html',
#         query=query,
#         classification_result=classification_results
#     )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
