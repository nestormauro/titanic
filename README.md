# Titanic
This package solves the Titanic Kaggle competition as part of a technical assessment. 

# Installing

In order to install the packages needed for this project you need to install `poetry` first. You can install it using this command:

`pip install poetry`

Once `poetry` be installed you can install the required packages with this command:

`poetry install`

# Package usage

You can use the  `main.py` script to manage the package. This script receives two arguments: `program-option` and `algorithm`, all required.

* The available options for `program-option` are: `Train`, `Predict`, and `Test`
* The available options for `algorithm` are: `Dummy`, `RandomForest`, and `GradientBoosting`

## Train

This option will train an ML model to solve the Titanic dataset and then will create a `joblib` file to save the trained model. The model will be saved in the path `model/models/`.

This is an example of training the Random Forest algorithm `main.py "Train" "RandomForest"`
**Note:** You can use the other available algorithms

## Predict

This option will use a trained model `joblib` file located in `model/models/` to predict over the data in file `data/test.csv`. 
**Notes:**
* You can add or delete rows to the file `data/test.csv` to get other predictions.
* In case the latest model has not been trained previously you won't be able to get predictions. You must train the model before

This is an example of predicting over `data/test.csv` and using the Gradient Boosting algorithm `main.py "Predict" "GradientBoosting"`

## Test

This option will run the `pytest` command and then will create a coverage test. The coverage test will be showing on the screen, something similar to this:

```
---------- coverage: platform linux, python 3.9.16-final-0 -----------
Name                             Stmts   Miss  Cover
----------------------------------------------------
__init__.py                          0      0   100%
config/__init__.py                   0      0   100%
config/settings.py                   9      9     0%
data/__init__.py                     0      0   100%
etl/__init__.py                      0      0   100%
etl/data_loader.py                  19     19     0%
etl/feature_engineer.py            100    100     0%
main.py                             21     21     0%
model/__init__.py                    0      0   100%
model/dummy_classifier.py            6      6     0%
model/gradient_boosting.py           6      6     0%
model/kfold_cross_validator.py      29     29     0%
model/models/__init__.py             0      0   100%
model/random_forest.py               6      6     0%
predict/__init__.py                  0      0   100%
predict/predictor.py                24     24     0%
test_data/__init__.py                0      0   100%
test_titanic.py                     69      0   100%
train/__init__.py                    0      0   100%
train/trainer.py                    36     36     0%
----------------------------------------------------
TOTAL                              325    256    21%
Coverage HTML written to dir htmlcov
```

And then in the directory `htmlcov`. You can use a web browser to see the file `htmlconv/index.html`.

# Analysis

You can see the analysis of this technical assessment in the next jupyter notebook [analysis](https://github.com/nestormauro/titanic/blob/main/notebooks/analysis.ipynb)
