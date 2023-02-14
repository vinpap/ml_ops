import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized

class DistanceTransformer(TransformerMixin, BaseEstimator):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        
    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
    
        X_ = X.copy()
        distance = haversine_vectorized(X)
        X_["distance"] = distance
        return X_[["distance"]]


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self, time_column_name="pickup_datetime"):
        self.time_column_name = time_column_name
        self.utc_time_column = time_column_name
        self.NY_time_column = time_column_name + "_NY_time"
    
    def __extract_time_features(self, df):
        
        df[self.utc_time_column] = pd.to_datetime(df[self.utc_time_column], 
                                               infer_datetime_format=True
                                          )
        df[self.NY_time_column] = df[self.utc_time_column].dt.tz_convert("US/Eastern")
        df["hour"] = df[self.NY_time_column].dt.hour
        df["dow"] = df[self.NY_time_column].dt.dayofweek
        df["month"] = df[self.NY_time_column].dt.month
        df["year"] = df[self.NY_time_column].dt.year
        df.set_index(self.utc_time_column, inplace=True)
        return df

    def fit(self, X, y=None): return self
        

    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = self.__extract_time_features(X_)
        return X_[["hour", "dow", "month", "year"]]