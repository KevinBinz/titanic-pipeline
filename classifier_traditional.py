import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from logger import logger

dirname = os.path.dirname(__file__)
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

feature_list = ["Pclass"]
label = "Survived"
X_train = train_df[feature_list].copy()
X_test = test_df[feature_list].copy()
y_train = train_df[label].values.ravel()

logger.info("train X/Y = {}/{}, test X={}".format(X_train.shape, y_train.shape, X_test.shape))

clf = LogisticRegression(solver="lbfgs")
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
logger.info("Validation Accuracy: {:.3f} ± {:.3f}".format(np.mean(scores), 2 * np.std(scores)))