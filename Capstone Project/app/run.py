import json
import pickle

import numpy as np
import pandas as pd
import plotly
from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from plotly.graph_objs import Bar, Pie
from wtforms.fields import SelectField

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

    offer_values_dict = portfolio_engineered[portfolio_engineered['offer_id'] == offer_id].drop('offer_id',
                                                                                                axis=1).to_dict(
        'records')[0]
    user_values_dict = profile_engineered[profile_engineered['id'] == user_id].drop('id', axis=1).to_dict('records')[0]

    return offer_values_dict, user_values_dict


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


# Index webpage displays cool visuals and receives user input text for model.
@app.route('/')
@app.route('/index')
def index():
    labels = ['Successul', 'Not-successful']
    success = np.mean(df['successful_offer'])
    values = [success, 1 - success]

    offer_types = df.groupby('offer_type')['successful_offer'].mean()
    offer_types_df = pd.DataFrame({'offer_type': offer_types.index, '% of success': offer_types.values})

    df['age_interval'] = df['age'].apply(age_interval)
    grouped_ages = df.groupby('age_interval')['successful_offer'].mean()
    grouped_ages_df = pd.DataFrame({'age': grouped_ages.index, '% of success': grouped_ages.values})

    df['income_interval'] = df['income'].apply(income_interval)
    incomes = df.groupby('income_interval')['successful_offer'].mean()
    incomes_df = pd.DataFrame({'income': incomes.index, '% of success': incomes.values})

    graphs = [
        {
            'data': [
                Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=['#006341', 'black'])
                )
            ],
            'layout': {
                'title': "Distribution of offers' success",
                'titlefont': {
                    'size': 18,
                }
            }
        },

        {
            'data': [
                Bar(
                    x=offer_types_df['offer_type'],
                    y=offer_types_df['% of success'],
                    marker=dict(color='#006341')
                )
            ],

            'layout': {
                'title': 'Probability of success by offer type',
                'yaxis': {
                    'title': "% of success",
                    'titlefont': {
                        'size': 18,
                    }
                },
                'xaxis': {
                    'title': "Offer type",
                    'titlefont': {
                        'size': 18,
                    }
                },
                "titlefont": {
                    "size": 28
                },
            }
        },
        {
            'data': [
                Bar(
                    x=grouped_ages_df['age'],
                    y=grouped_ages_df['% of success'],
                    marker=dict(color='#006341')
                )
            ],

            'layout': {
                'title': 'Probability of success by age',
                'yaxis': {
                    'title': "% of success",
                    'titlefont': {
                        'size': 18,
                    }
                },
                'xaxis': {
                    'title': "Age",
                    'titlefont': {
                        'size': 18,
                    }
                },
                "titlefont": {
                    "size": 28
                },
            }
        },
        {
            'data': [
                Bar(
                    x=incomes_df['income'],
                    y=incomes_df['% of success'],
                    marker=dict(color='#006341')
                )
            ],

            'layout': {
                'title': 'Probability of success by customer income',
                'yaxis': {
                    'title': "% of success",
                    'titlefont': {
                        'size': 18,
                    }
                },
                'xaxis': {
                    'title': "Income (in thousands USD)",
                    'titlefont': {
                        'size': 18,
                    }
                },
                "titlefont": {
                    "size": 28
                },
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


@app.route('/machine-learning', methods=['GET', 'POST'])
def machine_learning():
    """Machine learning page."""
    form = ChoiceForm()
    if request.method == 'POST':
        user_id_value = form.user_id.data
        offer_id_value = form.offer_id.data
        offer_values_dict, user_values_dict = get_offer_and_user_attributes(portfolio_engineered=portfolio,
                                                                            profile_engineered=profile,
                                                                            user_id=user_id_value,
                                                                            offer_id=offer_id_value)
        data = {'user_info': user_values_dict, 'offer_info': offer_values_dict}
        attributes = np.array(list(user_values_dict.values()) + list(offer_values_dict.values()))
        prediction = predict(ml_model=model, values=attributes)
        data.update({'Prediction': prediction})
        return render_template('machine-learning.html', data=data, form=form)

    user_id_value = form.user_id.choices[0][0]
    offer_id_value = form.offer_id.choices[0][0]
    offer_values_dict, user_values_dict = get_offer_and_user_attributes(portfolio_engineered=portfolio,
                                                                        profile_engineered=profile,
                                                                        user_id=user_id_value,
                                                                        offer_id=offer_id_value)
    data = {'user_info': user_values_dict, 'offer_info': offer_values_dict}

    return render_template('machine-learning.html', data=data, form=form)


def convert_values_to_string(values):
    for key, value in values.items():
        values[key] = str(value)
    return values


@app.route('/update/<user_id>')
def update(user_id):
    """Update information about user ID."""
    form = ChoiceForm()
    _, user_values_dict = get_offer_and_user_attributes(portfolio_engineered=portfolio,
                                                        profile_engineered=profile,
                                                        user_id=user_id,
                                                        offer_id=form.offer_id.choices[0][0])

    data = {'user_info': convert_values_to_string(user_values_dict)}
    return jsonify(data)


@app.route('/update-offer/<offer_id>')
def update_offer(offer_id):
    """Update information about offer ID."""
    form = ChoiceForm()
    offer_values_dict, _ = get_offer_and_user_attributes(portfolio_engineered=portfolio,
                                                         profile_engineered=profile,
                                                         user_id=form.user_id.choices[0][0],
                                                         offer_id=offer_id)

    data = {'offer_info': convert_values_to_string(offer_values_dict)}
    return jsonify(data)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
