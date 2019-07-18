import pandas as pd
import numpy as np
from eli5.lightgbm import explain_weights_lightgbm
from eli5.formatters import format_as_text
from eli5 import transform_feature_names
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


from logger import logger
from pipeline_lib import NumberSelector, TextSelector, ColumnSelector, TypeSelector, CustomLabelBinarizer, TextImputer, ModifiedCountVec
from df_utils import DataframeStats

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

    def drop_nonfeatures(self, df):
        snapshot = DataframeStats.snapshot(df)
        drop_headers = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']
        keep_headers = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']

        shape_before = df.shape
        for header in drop_headers:
            df.drop(labels=header, axis=1, inplace=True)
        logger.info("drop_nonfeatures(). Shape Changed: {} -> {}".format(shape_before, df.shape))
        logger.info("... columns dropped: {}".format(drop_headers))
        logger.info("... columns remaining: {}".format(df.columns.values.tolist()))
        DataframeStats.show_delta(snapshot, DataframeStats.snapshot(df))
        return df

    def get_data(self):
        # Data Ingestion
        X_train = pd.read_csv("./data/train.csv")
        y_train = X_train[Headers.SURVIVED].copy().values.ravel()
        X_train.drop(Headers.SURVIVED, axis=1, inplace=True)

        X_test = pd.read_csv("./data/test.csv")

        X_train = self.process(X_train)
        X_train = self.drop_nonfeatures(X_train)
        DataframeStats.snapshot(X_train).display()


        X_test = self.process(X_test)
        X_test = self.drop_nonfeatures(X_test)
        DataframeStats.snapshot(X_test).display()

        # X_train["CabinNull"] = X_train["Embarked"].notnull().astype('int')
        # X_test["CabinNull"] = X_test["Embarked"].notnull().astype('int')

        logger.info("train X/Y = {}/{}, test X={}".format(X_train.shape, y_train.shape, X_test.shape))

        return X_train, y_train, X_test

    def process(self, df):
        snapshot = DataframeStats.snapshot(df)
        df[Headers.SEX] = df[Headers.SEX].notnull().astype('category')
        df[Headers.EMBARKED] = df[Headers.EMBARKED].notnull().astype('category')
        df[Headers.FAMILY_SIZE] = df[Headers.SIBLING_SPOUSE] + df[Headers.PARENT_CHILDREN]
        DataframeStats.show_delta(snapshot, DataframeStats.snapshot(df))
        return df

    def get_pipe(self):
        ftunion = make_union(
            # Categorical Features
            # make_pipeline(TextSelector(key=Headers.SEX), TextImputer(), CustomLabelBinarizer()),
            # make_pipeline(TextSelector(key=Headers.EMBARKED), TextImputer(), CustomLabelBinarizer()),

            # Numeric Features
            make_pipeline(NumberSelector(key=Headers.P_CLASS)),
            make_pipeline(NumberSelector(key=Headers.AGE), SimpleImputer(strategy="median")),
            make_pipeline(NumberSelector(key=Headers.FAMILY_SIZE), SimpleImputer(strategy="median")),
            make_pipeline(NumberSelector(key=Headers.FARE), SimpleImputer(strategy="median")),
        )

        cols = [Headers.P_CLASS, Headers.AGE, Headers.SEX, Headers.FAMILY_SIZE, Headers.FARE, Headers.EMBARKED]
        ftunion2 = make_pipeline(
            ColumnSelector(columns=cols),
            FeatureUnion(transformer_list=[
                ("numeric_features", make_pipeline(
                    TypeSelector(np.number),
                    SimpleImputer(strategy="median"),
                    StandardScaler()
                )),
                ("categorical_features", make_pipeline(
                    TypeSelector("category"),
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder()
                )),
                # ("object_features", make_pipeline(
                #     TypeSelector("object"),
                #     SimpleImputer(strategy="most_frequent"),
                #     CustomLabelBinarizer()
                # )),
                # ("boolean_features", make_pipeline(
                #     TypeSelector("bool"),
                #     SimpleImputer(strategy="most_frequent")
                # ))
            ])
        )
        #logger.info(ftunion2.get_feature_names())
        return ftunion2

    def train(self, X_train, y_train):
        logger.info("\n{}".format(X_train.dtypes))
        features = self.get_pipe()
        # logger.info(features.get_feature_names())
        # features.fit_transform(X_train)
        pipe = make_pipeline(features, LGBMClassifier())
        logger.info(features.get_params().keys())
        pipe.fit(X_train, y_train)
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        logger.info("Validation Accuracy: {:.3f} ± {:.3f}".format(np.mean(scores), 2 * np.std(scores)))

        clf = pipe.steps[1][1]
        ft = pipe.steps[0][1]
        self.print_importances(clf, X_train)
        logger.info(format_as_text(explain_weights_lightgbm(lgb=clf, vec=ft)))
        return pipe

    def predict(self, pipe, X_test):
        y_test = pipe.predict(X_test)
        return y_test


if __name__ == '__main__':
    model = ClassifierPipeline()
    X_train, y_train, X_test = model.get_data()
    pipe = model.train(X_train, y_train)
    y_test = model.predict(pipe, X_test)
