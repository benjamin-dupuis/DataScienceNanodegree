import json

import pandas as pd
import plotly
from flask import Flask, render_template, request, redirect, url_for, jsonify
from plotly.graph_objs import Bar
from flask_wtf import FlaskForm
import numpy as np
from wtforms.fields import SelectField
import pickle

app = Flask(__name__)


def load_model(model_filepath):
    """Load trained machine learning model."""
    with open(model_filepath, 'rb') as pickle_model:
        ml_model = pickle.load(pickle_model)
    pickle_model.close()
    return ml_model


# Load model
model = load_model('../models/classifier.pkl')

# Load data
df = pd.read_csv('../data/data_for_analysis.csv')
portfolio = pd.read_csv('../data/engineered_portfolio.csv')
profile = pd.read_csv('../data/engineered_profile.csv')
app.config['SECRET_KEY'] = 'secret'


def get_offer_and_user_attributes(portfolio_engineered, profile_engineered, offer_id, user_id):
    """Get the offer and user attributes from DataFrames."""
    assert offer_id in list(portfolio_engineered['offer_id']), f"Offer ID {offer_id} not found in portfolio DataFrame."
    assert user_id in list(profile_engineered['id']), f"User ID {user_id} not found in profile DataFrame."

    offer_values = list(
        portfolio_engineered[portfolio_engineered['offer_id'] == offer_id].drop('offer_id', axis=1).values[0])
    user_values = list(profile_engineered[profile_engineered['id'] == user_id].drop('id', axis=1).values[0])

    return np.array(user_values + offer_values)


def predict(ml_model, values):
    """
    User machine learning model to make a prediction for given values.
    :param ml_model: Machine learning model.
    :param values: List of values.
    :return: The prediction (int).
    """
    prediction = ml_model.predict(values.reshape(-1, 1).T)
    return prediction[0]


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


class ChoiceForm(FlaskForm):
    user_ids = list(df['id'].unique())
    offer_ids = list(df['offer_id'].unique())
    user_id = SelectField('User ID', choices=list(zip(user_ids, user_ids)))
    offer_id = SelectField('Offer ID', choices=list(zip(offer_ids, offer_ids)))


@app.route('/machine-learning',  methods=['GET', 'POST'])
def machine_learning():
    """Machine learning page."""
    form = ChoiceForm()

    if request.method == 'POST':
        user_id_value = form.user_id.data
        offer_id_value = form.offer_id.data
        attributes = get_offer_and_user_attributes(portfolio_engineered=portfolio,
                                                   profile_engineered=profile,
                                                   user_id=user_id_value,
                                                   offer_id=offer_id_value)
        prediction = predict(ml_model=model, values=attributes)
        #data = {'Offer ID': offer_id_value, 'User ID': user_id_value}
        data = {'Prediction': prediction}
        return render_template('machine-learning.html', data=data, form=form)
    data = {}
    return render_template('machine-learning.html', data=data, form=form)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
