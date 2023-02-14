import pandas as pd


def get_data(nrows=10000):
    """Retrieves the data from the hard drive"""
    df = pd.read_csv("TaxiFareModel/data/train.csv", nrows=nrows)
    return df



def clean_data(df, test=False):
    """returns a DataFrame without outliers and missing values"""
    df.dropna(inplace=True)
    df = df[df["passenger_count"].between(0, 12)]
    df = df[df["pickup_latitude"].between(left = 40, right = 42 )]
    df = df[df["pickup_longitude"].between(left = -74.3, right = -72.9 )]
    df = df[df["dropoff_latitude"].between(left = 40, right = 42 )]
    df = df[df["dropoff_longitude"].between(left = -74, right = -72.9 )]
    df = df[df["fare_amount"] > 0]
    return df