# Kickstarter projects analysis

This project used the data collected from the 2018 Kickstarter projects. The dataset is available here: 
[Kaggle](https://www.kaggle.com/kemical/kickstarter-projects)

I did an Medium article describing my data analysis. This article is available [here](https://medium.com/@benjamindupuis/this-is-what-it-takes-to-have-a-successful-kickstarter-project-5806d28b6fab).



## Dependencies

The code uses Python, as well as the following packages:
- [jupyter](https://jupyter.org/install)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [seaborn](https://github.com/mwaskom/seaborn)


## Installation

To be able to run the code, do the following steps:

1. Clone the repo and move into the project's folder. For this, open a terminal and run the following commands:
```
git clone https://github.com/benjamin-dupuis/DataScienceNanodegree.git
cd DataScienceNanodegree
cd Intro to Data Science
```

2. Open the notebook with Jupyter:
```
jupyter notebook
```
3. Run the cells inside the notebook.


## Project motivation

The goal of this project was to analyse the key aspects of a succesful Kickstarter project. 
More specifically, I wanted to answer the following questions: 

- Do high money goals lead to a higher degree of success ?
- Is it better to have a short or a long deadline ?
- Does the name of a project has an influence on its probability of success ?
- Can we use machine learning to predict if a project will be successful ?


## Project takeways

From our analysis, we arrived at the following conclusions:

- Small money goals are much more likely to lead to a successful project.
- It is better for a project to have short deadline, approximately 2 months.
- A name with a medium length has the best chance of success.
- We were able to use a machine learning model to predict the outcome of a project with an accuracy of 70 %.

