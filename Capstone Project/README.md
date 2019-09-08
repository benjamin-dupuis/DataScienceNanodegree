# Starbucks Capstone Project

This project uses machine learning to predict if a Stabucks customer will respond postively to an marketing offer.


### Installation

To download the repository, open a cmd prompt and execute 
```
git clone https://github.com/benjamin-dupuis/DataScienceNanodegree.git
```

Move into the Disaster Response project folder:

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


### Machine Learning

The model that was selected was GradientBoostingClassifier. The model was able to predict whether or not an offer was going to be successful with an accuracy of **75%**.
The model was fine-tuned using [scikit-learn GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). 
