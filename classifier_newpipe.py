import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from pipeline_lib import Debug

from logger import logger


class Headers:
    P_CLASS = "pclass"
    NAME = "name"
    SEX = "sex"
    AGE = "age"
    SIBLING_SPOUSE = "sibsp"
    PARENT_CHILDREN = "parch"
    TICKET = "ticket"
    FARE = "fare"
    CABIN = "cabin"
    EMBARKED = "embarked"
    SURVIVED = "survived"
    PASSENGER_ID = "passengerid"

    # Derived Columns
    FAMILY_SIZE = "familysize"


class NewPipeline:
    def get_data(self, fn):
        df = pd.read_csv(fn)
        label = df.survived.values
        features = df[[Headers.EMBARKED, Headers.SEX, Headers.P_CLASS, Headers.AGE, Headers.FARE]]
        X_train, X_test, y_train, y_test = train_test_split(features, label)

        return X_train, X_test, y_train, y_test

    def get_pipe(self):
        numeric_features = [Headers.AGE, Headers.FARE]
        categorical_features = [Headers.EMBARKED, Headers.SEX, Headers.P_CLASS]
        # to_be_removed = [Headers.NAME, Headers.CABIN, Headers.TICKET, Headers.PASSENGER_ID]

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value="<unk>")),
            ('encoder', OneHotEncoder())
        ])

        preprocessor = ColumnTransformer(
            remainder='passthrough',
            transformers=[
                ('numeric', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features),
                # ('remove', 'drop', to_be_removed)
            ]
        )

        pipe = make_pipeline(preprocessor, Debug(), LGBMClassifier())

        #  logger.debug(features.get_params().keys())
        # pipe.fit(X_train, y_train)
        # scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        # logger.info("Validation Accuracy: {:.3f} ± {:.3f}".format(np.mean(scores), 2 * np.std(scores)))

        # clf = pipe.steps[1][1]
        # self.print_importances(clf, X_train)
        # logger.info(format_as_text(explain_weights_lightgbm(lgb=clf, vec=features)))
        return pipe

    def predict(self, pipe, X_test, k=3):
        """

        :param pipe:
        :param X_test:
        :return:
        """
        y_test = pipe.predict(X_test)
        return y_test


if __name__ == '__main__':
    fn = "https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/master/notebooks/datasets/titanic3.csv"
    model = NewPipeline()
    X_train, X_test, y_train, y_test = model.get_data(fn)
    pipe = model.get_pipe()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    logger.info(accuracy_score(y_pred, y_test))

