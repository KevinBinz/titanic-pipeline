import pandas as pd
import os
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

dirname = os.path.dirname(__file__)
train_df = pd.read_csv(dirname / Path("./data/train.csv"))
test_df = pd.read_csv(dirname / Path("./data/test.csv"))

Xtrain_drop = ["PassengerId", "Survived", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
Xtest_drop = ["PassengerId", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
ytrain_drop = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

X_train = train_df.drop(Xtrain_drop, axis=1)  # Only remaining column: PClass
X_test = test_df.drop(Xtest_drop, axis=1)  # Only remaining column: PClass
y_train = train_df.drop(ytrain_drop, axis=1)  # Only remaining column: Survived

print("{}, {}, {}".format(X_train.shape, y_train.shape, X_test.shape))

clf = LogisticRegression()
scores = cross_val_score(clf, X_train, y_train.values.ravel(), cv=10, scoring='accuracy')
print(scores.mean())  # Score: 67.9%
