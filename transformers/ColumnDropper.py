import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):

    """
        Custom transformer to drop unneed columns in the pipeline
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        cols = self.columns
        if cols is None:
            self.to_drop_ = []
        else:
            if isinstance(cols, str):
                cols = [cols]
            self.to_drop_ = [c for c in cols if c in X.columns]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")