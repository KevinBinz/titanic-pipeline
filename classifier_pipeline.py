import pandas as pd
import numpy as np
from eli5.lightgbm import explain_weights_lightgbm
from eli5.formatters import format_as_text
from eli5 import transform_feature_names
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from logger import logger
from pipeline_lib import NumberSelector, TextSelector, CustomLabelBinarizer, TextImputer, ModifiedCountVec


class Headers:
    P_CLASS = "Pclass"
    NAME = "Name"
    SEX = "Sex"
    AGE = "Age"
    SIBLING_SPOUSE = "SibSp"
    PARENT_CHILDREN = "Parch"
    TICKET = "Ticket"
    FARE = "Fare"
    CABIN = "Cabin"
    EMBARKED = "Embarked"
    SURVIVED = "Survived"
    PASSENGER_ID = "PassengerId"

    # Derived Columns
    FAMILY_SIZE = "FamilySize"


@transform_feature_names.register(NumberSelector)
def odd_feature_names(transformer, in_names=None):
    if in_names is None:
        from eli5.sklearn.utils import get_feature_names
        # generate default feature names
        in_names = get_feature_names(transformer, num_features=transformer.n_features_)
    # return a list of strings derived from in_names
    return in_names[1::2]

class ClassifierPipeline:
    def print_importances(self, clf, X_train):
        # TODO: would be nice to convert this to matplotlib bar chart.
        values = sorted(zip(X_train.columns, clf.feature_importances_), key=lambda x: x[1] * -1)
        for column, imp in values:
            logger.info("Feature {0}: {1}".format(column, imp))

    def drop_nonfeatures(self, df, features):
        transformers = features.transformer_list
        feature_list = list()
        for transformer in transformers:
            (name, _) = transformer
            feature_list.append(name)
        df_columns = df.columns.values

        shape_before = df.shape
        drop_cols = list()
        for column in df_columns:
            if column not in feature_list:
                drop_cols.append(column)
                df.drop(labels=column, axis=1, inplace=True)
        logger.info("drop_nonfeatures(). Shape Changed: {} -> {}".format(shape_before, df.shape))
        logger.info("... columns dropped: {}".format(drop_cols))
        logger.info("... columns remaining: {}".format(df.columns.values.tolist()))
        assert sorted(feature_list) == sorted(df.columns.values)
        return df

    def get_data(self):
        # Data Ingestion
        X_train = pd.read_csv("./data/train.csv")
        y_train = X_train[Headers.SURVIVED].copy().values.ravel()
        X_train.drop(Headers.SURVIVED, axis=1, inplace=True)

        X_test = pd.read_csv("./data/test.csv")

        # Data Transformation
        X_train[Headers.FAMILY_SIZE] = X_train[Headers.SIBLING_SPOUSE] + X_train[Headers.PARENT_CHILDREN]
        X_test[Headers.FAMILY_SIZE] = X_test[Headers.SIBLING_SPOUSE] + X_test[Headers.PARENT_CHILDREN]

        # X_train["CabinNull"] = X_train["Embarked"].notnull().astype('int')
        # X_test["CabinNull"] = X_test["Embarked"].notnull().astype('int')

        # Data Synchronization
        pipe = self.get_pipe()
        X_train = self.drop_nonfeatures(X_train, pipe)
        X_test = self.drop_nonfeatures(X_test, pipe)

        logger.info("train X/Y = {}/{}, test X={}".format(X_train.shape, y_train.shape, X_test.shape))
        logger.info(X_train.isnull().sum())
        logger.info(X_train.dtypes)

        return X_train, y_train, X_test

    def get_pipe(self):
        return FeatureUnion([
            # Categorical Features
            (Headers.SEX, Pipeline([
                ('selector', TextSelector(key=Headers.SEX)),
                ('imputer',  TextImputer()),
                ('encoder', CustomLabelBinarizer())
            ])),
            (Headers.EMBARKED, Pipeline([
                ('selector', TextSelector(key=Headers.EMBARKED)),
                ('imputer',  TextImputer()),
                ('encoder', CustomLabelBinarizer())
            ])),
            # (Headers.NAME, Pipeline([
            #     ('selector', TextSelector(key=Headers.NAME)),
            #     ('imputer',  TextImputer()),
            #     ('encoder', ModifiedCountVec(max_features=10))
            # ])),
            # Numeric Features
            (Headers.P_CLASS, Pipeline([
                ('selector', NumberSelector(key=Headers.P_CLASS)),
                # ('imputer',  SimpleImputer(strategy="median"))
            ])),
            (Headers.AGE, Pipeline([
                ('selector', NumberSelector(key=Headers.AGE)),
                ('imputer',  SimpleImputer(strategy="median"))
            ])),
            (Headers.FAMILY_SIZE, Pipeline([
                ('selector', NumberSelector(key=Headers.FAMILY_SIZE)),
                ('imputer',  SimpleImputer(strategy="median"))
            ])),
            (Headers.FARE, Pipeline([
                ('selector', NumberSelector(key=Headers.FARE)),
                ('imputer',  SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])),
        ])

    def train(self, X_train, y_train):
        features = self.get_pipe()
        # logger.info(features.get_feature_names())
        features.fit_transform(X_train, y_train)
        pipe = Pipeline([
            ('features', features),
            ('model', LGBMClassifier()),
        ])
        logger.debug(features.get_params().keys())
        pipe.fit(X_train, y_train)
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        logger.info("Validation Accuracy: {:.3f} ± {:.3f}".format(np.mean(scores), 2 * np.std(scores)))

        clf = pipe.steps[1][1]
        self.print_importances(clf, X_train)
        # logger.info(format_as_text(explain_weights_lightgbm(lgb=clf, vec=features)))
        return pipe

    def predict(self, pipe, X_test):
        y_test = pipe.predict(X_test)
        return y_test


if __name__ == '__main__':
    model = ClassifierPipeline()
    X_train, y_train, X_test = model.get_data()
    pipe = model.train(X_train, y_train)
    y_test = model.predict(pipe, X_test)
