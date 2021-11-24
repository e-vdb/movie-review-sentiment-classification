"""Data-preprocessing"""
import pandas as pd


def setup_datatrain(df):
    X = df.review
    y = df.sentiment
    X_clean = pd.Series([rev.replace("<br />", " ") for rev in X], name='review')
    return X_clean, y