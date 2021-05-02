# Disaster Response Pipeline

This project uses machine learning to categorize messages
that were sent during disaster events.

It also contains a webapp where a user can send
messages and have them classified.



![](/app/static/images/webapp_screen_shot.png)


## Getting Started

### Prerequisites

Python 3.6.13

The list of packages
and recommended versions are listed in file app/requirements.txt

### Installation

Install the packages listed in app/requirements.txt
preferably in a virtual environment.

```python
pip install -r app/requirements.txt
```

Then, run the following command from the repository's root

```python
python app/run.py
```

Go to
[http://0.0.0.0:3001/](http://0.0.0.0:3001/)
to see the web app live on your local machine.

### File Description

* /app - files needed to run the flask app (including python, html and css files.)
* /data - file with draining data and script to clean and process the data.
* /models - script with the pipeline to train the model.

## Training New Data

All input files used to train the model are already provided
in /data, and the corresponding trained model is located in /models .

It is possible to train a new model,
with the same data or with new data with the following steps:

* Clean the data and store it in a database

```python
python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
```

* Train a new model using the newly cleaned data.

```python
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

### The Model

For the training of the model Stochastic Gradient Descent (SGD)
combined with grid search
was used with the help of 
[Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
toolkits.

Several models were tested, but only the one with the highest F1 score
was included.

<!---
#### Model Performance

Comparison of F1-score results for models trained
for train and test sets.

| Model | Cross-validation Train | Train | Test |
| ------------- |:-------------:| -----:| -----:|
| SDG | XXX | XXX | XXX |
| Logistic regression | XXX | XXX | XXX |
| KNN | XXX | XXX | XXX |
| Naive Bayes | XXX | XXX | XXX |
| XGBoost | XXX | XXX | XXX |
--->

## Acknowledgement

Credit to Udacity and
[rjjfox](https://github.com/rjjfox/disaster-response-classification/tree/435a3c0dd67e7409dc0454601ba9560b9a8810de) 
for providing the data and/or serving as template for this project.
