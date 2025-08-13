import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEnginnering(BaseEstimator, TransformerMixin):
    """
        Custom transformer for feature enginnering
    """

    def __init__(self, q_high: float = 0.90):
        self.q_high = q_high
        self.amount_threshold = None
        self.score_threshold  = None
        self.score_scale_max = 100.0

    def fit(self, X, y=None):
        m = pd.to_numeric(X["monto"], errors="coerce").clip(lower=0)
        self.amount_threshold = float(np.quantile(m.dropna(), self.q_high))

        sc = pd.to_numeric(X["score"], errors="coerce")
        sc01 = (sc / self.score_scale_max).clip(0, 1)
        self.score_threshold = float(np.quantile(sc01.dropna(), self.q_high))
        return self

    def _transform_date_time_values(self , X: pd.DataFrame) :
        """transform and extract date time values and features

        Args:
            X (pd.DataFrame): dataframe with the data

        Returns:
            pd.DataFrame
        """
        date = pd.to_datetime(X["fecha"], errors="coerce", utc=True)
        X["hour"]  = date.dt.hour.astype("Int64")
        X["dow"]   = date.dt.dayofweek.astype("Int64")
        X["day"] = date.dt.day.astype("Int64")
        X["night"] = X["hour"].isin([22,23,0,1,2,3,4,5]).astype("int8")
        return X

    def transform(self, X):
        X = X.copy()
        X = self._transform_date_time_values(X)
        X["p_bin"] = (
            X["p"].astype("string").str.strip().str.upper()
            .map({"Y":1, "N":0}).astype("Int8")
        )

        m = pd.to_numeric(X["monto"], errors="coerce").clip(lower=0)
        X["monto_log1p"] = np.log1p(m)
        X["high_amount"] = (m >= self.amount_threshold).astype("int8")

        sc01 = (pd.to_numeric(X["score"], errors="coerce") / self.score_scale_max).clip(0, 1)
        X["score_pct"]  = sc01
        X["high_score"] = (sc01 >= self.score_threshold).astype("int8")

        for c in ("o", "p"):
            if c in X.columns:
                X.drop(columns=c, inplace=True, errors="ignore")

        return X
