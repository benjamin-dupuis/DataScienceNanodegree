# Link to Web app: [https://udacity-starbucks-capstone.herokuapp.com/](https://udacity-starbucks-capstone.herokuapp.com/)


# Starbucks Capstone Project

This project uses machine learning to predict if a Stabucks customer will respond postively to an marketing offer.
For this project, we had a dataset containing the offers sent to the clients via different sources, i.e 'web', 'social', 'email', and 'mobile'. There were three different kind of offers sent to the clients : 
1. "BOGO": Buy One Get One Free
2. "Discount": A discount given to the customer for a purchase.
3. "Informational": An advertisement.

Another important point was that the offers were valid for a given number of days.

In the dataset, all of the transactions made by the different clients were recorded. Therefore, the task was to determine from those transactions which offers were successful, and from there, try to predict whether or not a client would respond positively to an offer.

For a company like Starbucks, that has millions of customers, that data insight can save them thousands of dollars every year. Indeed, knowing which users to send offers (and which user to not send offer to) can not only create more revenues, but also reduce costs.
This project includes a data analysis of that dataset, a machine learning model trained to predict if an offer is going to be successful when sent to a given customer, and a web application to gather and present the results.


### Installation

To download the repository, open a cmd prompt and execute 
```
git clone https://github.com/benjamin-dupuis/DataScienceNanodegree.git
```

Move into the Capstone project folder:

```
cd Capstone Project
```

Create a virtual environment and activate it. For infos, click [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/).


Download the necessary libraries:

```
pip install -r requirements.txt
```

### Usage:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and saves the data in a CSV file:
        `python data/process_data.py`
    - To run ML pipeline that trains classifier and saves it:
        `python models/train_classifier.py data/cleaned_data.csv models/classifier.pkl`

2. Move into the app folder and run the application:
    ```
    cd app
    python run.py
    ```

3. Open a browser, and go to `localhost:3001`


### Insights

That project was challenging for several reasons. First of all, the cleaning and data engineering process had to be well thought and meticulously done. For each event in the dataset, we had to take into consideration different factors: the order of the preceding and following events, the result of that event, as well as the time it occured. Given the fact that a customer can make several purchases, not necessarly related to an previously, sent offer added difficulties to this data analysis. Also, the different offers had different definitions of 'success'. In particular, the 'Informational' offers, i.e advertisement, had no reward associated to it, nor an event tagged 'event completed' associated to it. 

Also, after deducing which offers to tag as successful, there was a high disbalance in the dataset. About 22 % of the offers were successful, the rest being not successful. The strategy that I opted was to do an "oversampling" of the data, meaning that I duplicated some of the datapoints so that the final dataset had balance in its target classes.

For the machine learning part, I tested several algorithms. The best model was a RandomForestClassifier, with an accuracy (on the balanced dataset) of 81 %. I also fine-tuned the model using [scikit-learn GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). 

All of the data engineering process and machine learning experimentations can be found in the [jupyter notebook](https://github.com/benjamin-dupuis/DataScienceNanodegree/blob/master/Capstone%20Project/Starbucks_Capstone_notebook.ipynb).


### Improvements

Several adjustements could be done to improve the results. First of all, he way that I deduced which offers were successful could be refined. For each offer received, I looked at the very next offers, and verified if the order of the events following the offer receiving matched a predefined sequence. However, between the time the user receives the offer, and makes a purchase that completes the offer, several unrelated events can happen: the user makes another purcharse, receives another offer, views an offer sent several days ago... Therefore, an adjustment would be to think in more broad terms, i.e look at the timestamps of the different actions and make sure that the required sequence is contained, in order, inside a valid time interval, not necessary one directly after another.

Also, more feature engineering could have been done. For example, for every user, look at the mean amount they spent for their different purchases, or how many offers they received.


