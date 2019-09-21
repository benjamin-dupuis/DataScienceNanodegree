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

In this project, the strategy that I chose to address the challenge mentionned above was to gather all the transactions data, and linking that to the offers characteristics, as well as the users attributes. By doing so, I was able to deduce which offers the different users responded to. Then, I combined various attributes from the dataset to create new attributes; that process is called *feature engineering*. From the engineered dataset, I built different machine learning models, and compared them using various metrics; thhose metrics are described in more details in the "Insights" section. After chosing the best performing model, I fine-tuned that model, i.e choose the best parameters to make that specific model perform the best. After training this model I saved it and used it to build a Web application. That Web application allows the user to select an offer ID and an offer ID, an use the machine learning model to predict if the offer will be successful. The strategy to build a machine learning model was used for its direct applicability and its rusability for future offers and future clients.

This project includes a data analysis of that dataset, a machine learning model trained to predict if an offer is going to be successful when sent to a given customer, and a web application to gather and present the results.


## Installation

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

## Usage:
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


## Files

- `app`
    - `run.py` : Contains the main python file that runs the Flask application.
    - `templates` : Folder that contains the HTML files used in the Web app.
    
- `data`
    - `portfolio.json`: JSON file containing the information about the offers.
    - `profile.json`: JSON file containing the information about the users.
    - `transcript.json`: JSON file containing all events related to the users and the offers.
    - `process_data.py`: Python file that loads, cleans, engineers, and saves the data into CSV files.

- `models`
    - `train_classifier.py`: Python file that loads the engineered data, trains a machine leaning model, and saves the trained model.
    
- `README.md`: File that describes the project and how to use it.

- `Starbucks_Capstone_Notebook.ipynb`: Jupyter Notebook that contains the data engineeering and data analysis, as well as the code used to train and test the selected machine learning models

- `requirements.txt`: Text file that contains all the Python packages necessary to run the code and the Web application.


## Libraries Used

-[Flask: Web Development framework](https://palletsprojects.com/p/flask/)

-[scikit-learn: Machine Learning](https://scikit-learn.org/stable/)  

-[Pandas: Data Analysis](https://pandas.pydata.org/)  

-[Numpy: Scientific Computing](http://www.numpy.org/)  

-[Matplotlib: Data Visualization](https://matplotlib.org/)  

-[seaborn: Data Visualization](https://seaborn.pydata.org/) 

-[jupyter: Interactive Notebook](https://jupyter.org/)

-[WTForms: Forms API for Flask Web Development](https://pypi.org/project/WTForms/)

-[imbalanced-learn: Data Analysis library to deal with imbalanced dataset](https://imbalanced-learn.readthedocs.io/en/stable/)

-[xgboost: Gradient Boosting library](https://xgboost.readthedocs.io/en/latest/)



## Insights

### Preprocessing

That data preprocessing was challenging for several reasons. First of all, the cleaning and data engineering process had to be well thought and meticulously done. For each event in the dataset, we had to take into consideration different factors: the order of the preceding and following events, the result of that event, as well as the time it occured. Given the fact that a customer can make several purchases, not necessarly related to an previously, sent offer added difficulties to this data analysis. Also, the different offers had different definitions of 'success'. In particular, the 'Informational' offers, i.e advertisement, had no reward associated to it, nor an event tagged 'event completed' associated to it. My strategy to deduce which offers were successful was the following:

As mentionned, for offers of type "informational", it is just an advertisement, so there is no "reward", nor a "difficulty". Also, for this type of offer, their is no "offer completed" event. 

Therefore, here are the steps necessary for the different offer types to be considered "successful": 

"BOGO" and "discount: 

   1. "offer received"
   2. "offer viewed"
   3. "offer completed"
   4. "transaction"
    
    
"informational":

   1. "offer received"
   2. "offer viewed"
   3. "transaction"


### Data Engineering

Here are the data engineering actions that I executed on the combined dataset. 

- Profile

    For the 'became_member_on' column, which indicated the date that the user started using the application, I performed 2 steps:
        1. I extracted the day of the week the user became memeber. In some cases, the fact that the user became a member on a weekday or on weekend can tell us if that user will be a recurrent buyer or not.
        2. I extracted the year the user became a member, and I "dummied" that variable.
        
    I also 'dummied' the gender column.
    

- Portoflio

    On the part of the dataset containing information about the different offers, I made the following actions:
    
    - I dummied the offer type.
    - I extracted the list of sources used to send the offers, created 4 different columns from those sources where the value is 1 or 0, indicating if the source was used or not for a given offer ID.
    - I also multiplied the duration by 24 to transform the value from days to hours.
    
    
- Transcript

    On the dataset contaning the events, I made the following data engineering actions:
    
    - I only took the data related to users in the profile section of the dataset. Indeed, we did not have any information about the addtional users, so we could not really use those in our analysis process.
    - I created a column named 'offer_id', containing values of the offer ID extracted from the event. However, the events of type 'transaction' did not have any offer ID, so we assigned a temporary value of "null" to those.
    - As mentionned above, we use a specific order of events for the different offers types to deduce whether or not a given offer was successful. Also, it's important to mention that the transaction had to be done in a given timeframe to consider the offer as successful.
    - I finally dropped several columns that did not give use any important information, like the offer_id or the time of the event (that value had already been used to deduce the offers success). I also dropped any duplicated rows from our dataset.
    
    
I ended up with the following schema for the dataset:

    age:                        Age of the user (in years)

    income                      Income of the user (in USD)

    weekday_membership          Weekday (value from 1 to 7) were the user became member

    gender_F                    1 if the user is a female, 0 otherwise.

    gender_M                    1 if the user is a male, 0 otherwise.

    gender_O                    1 if the gender is 'other', 0 otherwise. 

    became_member_on_2013       1 if the user became member in 2013, 0 otherwise.

    became_member_on_2014       1 if the user became member in 2014, 0 otherwise.

    became_member_on_2015       1 if the user became member in 2015, 0 otherwise.

    became_member_on_2016       1 if the user became member in 2016, 0 otherwise.

    became_member_on_2017       1 if the user became member in 2017, 0 otherwise.

    became_member_on_2018       1 if the user became member in 2018, 0 otherwise.

    difficulty                  Amount the user has to spend for the offer.

    reward                      Reward given to the user if he completes the offer.

    email                       1 if the offer was sent by email, 0 otherwise.

    mobile                      1 if the offer was sent by mobile, 0 otherwise.

    social                      1 if the offer was sent on social media, 0 otherwise.

    web                         1 if the offer was sent on the web, 0 otherwise.

    duration_hours              Duration in hours where the offer is active.

    offer_type_bogo             1 if the offer is "BOGO", 0 otherwise.

    offer_type_discount         1 if the offer is a discount, 0 otherwise.

    offer_type_informational    1 if the offer is an advertisement, 0 otherwise.

    successful_offer            1 if the offer is considered successful, 0 otherwise.
    



### Post Processing

After deducing which offers to tag as successful, there was a high disbalance in the dataset. About 22 % of the offers were successful, the rest being not successful. The strategy that I opted was to do an "oversampling" of the data, meaning that I duplicated some of the datapoints so that the final dataset had balance in its target classes.


### Machine Learning

For the machine learning part, I tested an compared the performance of several algorithms. Those experiements were made on the balanced dataset, ie that the two target classes, successful or not successful, were equally distributed. Here are the results:

<table>
  <tr>
    <th rowspan="2"><br>Algorithm</th>
    <th colspan="3">Training </th>
    <th colspan="3">Testing</th>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>Precision</td>
    <td>Recall</td>
    <td>Accuracy</td>
    <td>Precision</td>
    <td>Recall</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>59%</td>
    <td>70%</td>
    <td>58%</td>
    <td>59%</td>
    <td>70%</td>
    <td>57%</td>
  </tr>
  <tr>
    <td>Decision Tree</td>
    <td><b>94%</b></td>
    <td>98%</td>
    <td><b>91%</b></td>
    <td>80%</td>
    <td>90%</td>
    <td>72%</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td><b>94%</b></td>
    <td><b>99%</b></td>
    <td>90%</td>
    <td><b>82%</b></td>
    <td><b>92%</b></td>
    <td><b>76%</b></td>
  </tr>
  <tr>
    <td>Gradient Boosting</td>
    <td>60%</td>
    <td>70%</td>
    <td>59%</td>
    <td>60%</td>
    <td>70%</td>
    <td>58%</td>
  </tr>
  <tr>
    <td>AdaBoost</td>
    <td>59%</td>
    <td>67%</td>
    <td>58%</td>
    <td>59%</td>
    <td>67%</td>
    <td>57%</td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>60%</td>
    <td>70%</td>
    <td>59%</td>
    <td>60%</td>
    <td>70%</td>
    <td>59%</td>
  </tr>
</table>


As we can see above, the model that performs the best is by far the Random Forest Classifier. However, it does overfitting, because its training accuracy is around 94%, and its testing accuracy is around 81%. We will do hyperparameters tuning to find a better model. Also, in this particcular problem, we favorise recall over precision. Indeed, we think it's better to send the offers to all the clients who are susceptible to be influenced by it, even though it implies at the same time to send offers to clients that will not repond positively to those offers. In a long period of time, that strategy should be increase revenues more that it increases the expenses related to additional sent offers. Once again, the model that had the best recall metric was the Random Forest Classifier.

The Random Forest Classifier can be desribed as follows:

An group of decision trees are brought together to work as an *ensemble*. In each iteration, each individual tree in the random forest makes a class prediction. The most popular prediction among the ensemble is used as the final prediction of the model. The random part of that model resides in the fact that a random subset of features is used to make the predictions. The reason behind that is that if the model would use all of the features for every tree, somee trees would become highly correlated, and therefore the model would tend to nevery adjust its predictions. 


### Optimization

The best model was a RandomForestClassifier, with an accuracy (on the balanced dataset) of 94% on training and 81% on testing. That difference between trainining and testing accuracy indicates that the model does *overfitting*. To improve the performance of the model we used [scikit-learn GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). In particular, I tried different values of the "depth" of the tree, indicating how far the three can go the separate the different data points. That parameter in particular is important when trying to reduce overfitting. Indeed, the more splitting the algorithm does, the more it will tend to make over represent the data, or learn the data "by heart", meaning that the accuracy on the training will be very high, but when dealing with unknown data, due to that over specialization. 

The hyper-parameters that I chose to optimize are the following:

    "n_estimators" :        The number of trees in the Random Forest. Values used: [50, 60, 70, 100, 200, 400]
    
    "min_samples_split":    The minimum number of samples required to split an internal node. Values used: [2, 3, 5, 7, 9, 15]
    
    "min_samples_leaf":     The minimum number of samples required to be at a leaf node. Values used: [1, 3, 5, 7]
    
    "max_depth":            The maximum depth of the tree. Values used: [1, 2, 3, 5, 7, 9, 15, 20, 25]
    
    
Also, it's important that this algorithm has a random part to it, as mentionned above. Therefore, we chose a fixed random state for our training, so that the model would give the same results each time.

We also chose a Cross Validation equal to 5. Cross validation is the process of separating the dataset into subsets, (for training and testing dataset), and performing the validation on each of those. That process is repeated "k" times (in our case k=5). The final performance is determined by the average performance obtained on each subset. Also, the metric that we used in the fine-tuning process was the area under ROC curve, so that the balance between the accuracy and the recall is optimized.


<table>
  <tr>
    <th rowspan="2"><br>Algorithm</th>
    <th colspan="3">Training </th>
    <th colspan="3">Testing</th>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>Precision</td>
    <td>Recall</td>
    <td>Accuracy</td>
    <td>Precision</td>
    <td>Recall</td>
  </tr>
  <tr>
  <tr>
    <td>Random Forest</td>
    <td>94%</td>
    <td>99%</td>
    <td>90%</td>
    <td>82%</td>
    <td>92%</td>
    <td>76%</td>
  </tr>
    
   <tr>
    <td>Random Forest Optimized</td>
    <td>95%</td>
    <td>99%</td>
    <td>90%</td>
    <td>81%</td>
    <td>92%</td>
    <td>76%</td>
  </tr>

 
</table>



## Improvements

Several adjustements could be done to improve the results. First of all, he way that I deduced which offers were successful could be refined. For each offer received, I looked at the very next offers, and verified if the order of the events following the offer receiving matched a predefined sequence. However, between the time the user receives the offer, and makes a purchase that completes the offer, several unrelated events can happen: the user makes another purcharse, receives another offer, views an offer sent several days ago... Therefore, an adjustment would be to think in more broad terms, i.e look at the timestamps of the different actions and make sure that the required sequence is contained, in order, inside a valid time interval, not necessary one directly after another.

Also, more feature engineering could have been done. For example, for every user, look at the mean amount they spent for their different purchases, or how many offers they received.


## Refrerences
- [Udacity DSND Term 1](https://github.com/udacity/DSND_Term1)
- [Udacity DSND Term 2](https://github.com/udacity/DSND_Term2)
- [Understanding Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)
- [Flask Boilerplate](https://github.com/MaxHalford/flask-boilerplate)
- [Bootstrap in Flask](https://john.soban.ski/pass-bootstrap-html-attributes-to-flask-wtforms.html)
