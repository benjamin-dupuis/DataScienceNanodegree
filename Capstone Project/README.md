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


That data preprocessing was challenging for several reasons. First of all, the cleaning and data engineering process had to be well thought and meticulously done. For each event in the dataset, we had to take into consideration different factors: the order of the preceding and following events, the result of that event, as well as the time it occured. Given the fact that a customer can make several purchases, not necessarly related to an previously, sent offer added difficulties to this data analysis. Also, the different offers had different definitions of 'success'. In particular, the 'Informational' offers, i.e advertisement, had no reward associated to it, nor an event tagged 'event completed' associated to it. 


### Data Engineering


### Post Processing

After deducing which offers to tag as successful, there was a high disbalance in the dataset. About 22 % of the offers were successful, the rest being not successful. The strategy that I opted was to do an "oversampling" of the data, meaning that I duplicated some of the datapoints so that the final dataset had balance in its target classes.


### Machine Learning

For the machine learning part, I tested an compared the performance of several algorithms. Those experiements were made on the balanced dataset, ie that the two target classes, successful or not successful, were equally distributed. Here the results:

- Logistic Regression
    - Accuracy Metrics
    
        Training accuracy: 59%, Testing accuracy: 59%
 
    - Recall metrics:
    
        Training recall: 58%,  Testing recall: 57%

    - Precision Metrics
    
        Training precision: 70%,  Testing precision: 70%
       
- Decision Tree
    - Accuracy Metrics
    
        Training accuracy: **94%**,  Testing accuracy: 80%
       
    - Recall metrics:
    
        Training recall: **91%**,  Testing recall: 72%
       
    - Precision Metrics
    
        Training precision: 98%, Testing precision: 90%
        
    
- Random Forest
    - Accuracy Metrics
    
        Training accuracy: **94%**, Testing accuracy: **82%**
        
    - Recall metrics:
    
        Training recall: 90%, Testing recall: **76%**
        
    - Precision Metrics
    
        Training precision: **99%**,  Testing precision: **92%**
       
   

- Gradient Boosting
    - Accuracy Metrics
    
        Training accuracy: 60%,  Testing accuracy: 60%
       
    - Recall metrics:
    
        Training recall: 59%, Testing recall: 58%
        
    - Precision Metrics
    
        Training precision: 70%, Testing precision: 70%
        
        

- AdaBoost
    - Accuracy Metrics
    
        Training accuracy: 59%, Testing accuracy: 59%
        
    - Recall metrics:
    
        Training recall: 58%,  Testing recall: 57%
       
    - Precision Metrics
    
        Training precision: 67%,   Testing precision: 67%
      


- XGBoost
    - Accuracy Metrics
    
        Training accuracy: 60%, Testing accuracy: 60%
        
    - Recall metrics:
    
        Training recall: 59%,   Testing recall: 59%
      
    - Precision Metrics
    
        Training precision: 70%,   Testing precision: 70%
      


### Optimization

The best model was a RandomForestClassifier, with an accuracy (on the balanced dataset) of 81 %. I also fine-tuned the model using [scikit-learn GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). 



## Improvements

Several adjustements could be done to improve the results. First of all, he way that I deduced which offers were successful could be refined. For each offer received, I looked at the very next offers, and verified if the order of the events following the offer receiving matched a predefined sequence. However, between the time the user receives the offer, and makes a purchase that completes the offer, several unrelated events can happen: the user makes another purcharse, receives another offer, views an offer sent several days ago... Therefore, an adjustment would be to think in more broad terms, i.e look at the timestamps of the different actions and make sure that the required sequence is contained, in order, inside a valid time interval, not necessary one directly after another.

Also, more feature engineering could have been done. For example, for every user, look at the mean amount they spent for their different purchases, or how many offers they received.


