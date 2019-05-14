import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from eli5.sklearn.utils import get_feature_names
from logger import logger

# class CustomPipeline(BaseEstimator, TransformerMixin):
#     """
#     Transformer to select a single column from the data frame to perform additional transformations on
#     Use on text columns in the data
#     """
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return X
#
#     def get_feature_names(self, X):
#         """Assuming a single transformer in the `Pipeline` has `get_feature_names`,
#         call it and transform its result through the remainder of the pipeline."""
#         names = None
#         for name, step in transformers:
#             if hasattr(step, 'get_feature_names'):
#                 if names is not None:
#                     raise ValueError('Multiple steps with get_feature_names')
#                 names = step.get_feature_names()
#             elif names is not None:
#                 names = step.transform(names)
#         if names is None:
#             raise ValueError('No step with get_feature_names')
#         return names


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        out_df = X.select_dtypes(include=[self.dtype])
        logger.info("TypeSelector.transform(dtype={} shape={} columns={})".format(self.dtype, out_df.shape, out_df.columns.values))
        return out_df


class StringIndexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

    def get_feature_names(self, X):
        return self.key


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

    def get_feature_names(self, X):
        return self.key


class CustomLabelBinarizer(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(CustomLabelBinarizer, self).fit(X)

    def transform(self, X, y=None):
        return super(CustomLabelBinarizer, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(CustomLabelBinarizer, self).fit(X).transform(X)


class TextImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("TextImputer.transform(X shape={}, dtypes={} nulls={})".format(X.shape, X.dtypes, X.isna().sum()))
        X.fillna('UNK', inplace=True)
        return X


class CategoryImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        unk_token = 'UNK'
        X = X.cat.add_categories(unk_token)
        logger.info("CategoryImputer.transform(X shape={}, dtypes={} nulls={})".format(X.shape, X.dtypes, X.isna().sum()))
        X.fillna(unk_token, inplace=True)
        return X


class ModifiedCountVec(BaseEstimator, TransformerMixin):
    def __init__(self, **params):
        self.vect = CountVectorizer(**params)

    def transform(self, X):
        # print(X.unique())
        p = self.vect.transform(X)
        return p

    def fit(self, X, y=None, **fit_params):
        self.vect.fit(X)
        return self


class Debug(BaseEstimator, TransformerMixin):
    def transform(self, X):
        #print(X.shape)
        print(X[0:5, :])
        print("End")
        # what other output you want
        return X

    def fit(self, X, y=None, **fit_params):
        return self