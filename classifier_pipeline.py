from pathlib import Path

import pandas as pd
import numpy as np
import random as rnd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, RobustScaler, StandardScaler, Imputer
from sklearn.model_selection import cross_val_score

from selector import NumberSelector, TextSelector, LabelBinarizerPipelineFriendly

dirname = os.path.dirname(__file__)
train_df = pd.read_csv(dirname / Path("./data/train.csv"))
test_df = pd.read_csv(dirname / Path("./data/test.csv"))
combine = [train_df, test_df]

feature_names = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
feature_drops = ["Survived", "PassengerId"]
label_drops = ["Pclass", "PassengerId", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
X_train = train_df.drop(feature_drops, axis=1)
y_train = train_df.drop(label_drops, axis=1)
X_test = test_df.drop("PassengerId", axis=1).copy()
print("{}, {}, {}".format(X_train.shape, y_train.shape, X_test.shape))

#print(X_train.head())
print(y_train.head())

pclass = make_pipeline(
    NumberSelector(key='Pclass'))
sex = make_pipeline(
    TextSelector(key='Sex'),
    LabelBinarizerPipelineFriendly())
age = make_pipeline(
    NumberSelector(key="Age"),
    Imputer(strategy="median"))
embarked = make_pipeline(
    TextSelector(key="Embarked"),
    LabelBinarizerPipelineFriendly())
fare = make_pipeline(
    NumberSelector(key="Fare"),
    Imputer(strategy="median"),
    StandardScaler())
sibsp = make_pipeline(
    NumberSelector(key="SibSp"),
    Imputer(strategy="median"))
parch = make_pipeline(
    NumberSelector(key="Parch"),
    Imputer(strategy="median"))
#ticket
#cabin
#name

features = FeatureUnion([
    ('sex', sex),
    ('pclass', pclass),
    ('age', age),
    #('embarked', embarked),
    ('fare', fare),
    ('sibsp', sibsp),
    ('parch', parch),
    #('ticket', ticket),
    #('cabin', cabin)
    #('name', name)
])
features.fit_transform(X_train, y_train)
print(X_train.head())

pipe = make_pipeline(features, RandomForestClassifier())
scores = cross_val_score(pipe, X_train, y_train.values.ravel(), cv=10, scoring='accuracy')
print(scores.mean())  # Score = 80.0%
