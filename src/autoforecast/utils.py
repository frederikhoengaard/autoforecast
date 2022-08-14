from datetime import timedelta

import pandas as pd
from pandas import DataFrame


def load_dataset(fname: str) -> pd.DataFrame:
    df = pd.read_csv(fname,sep=';')
    dates = df['datetime'].tolist()
    vals = {}
    for col in df.columns[1:]:
        vals[col] = df[col].tolist()
    df = pd.DataFrame({
        col: vals[col]
    }, index=pd.DatetimeIndex(dates))
    return df


def featuremaker(df: DataFrame, target: str, lags=365) -> DataFrame:
    dates = df.index
    day = []
    month = []
    weekdays = []
    weekend = []
    features = {
        'day': [],
        'month': [],
        'weekday': [],
        'is_weekend': []
    }

    for i in range(1,lags+1):
        features['lag_'+str(i)] = []

    current_day_as_int = lags
    for date in dates[lags:]:
        dtdate = pd.to_datetime(date)
        features['day'].append(date.day)
        features['month'].append(date.month)
        features['weekday'].append(dtdate.weekday())
        features['is_weekend'].append(1 if dtdate.weekday() > 4 else 0)

        for i in range(1, lags+1):
            lag_day = date - timedelta(days=i)
            features['lag_'+str(i)].append(df[target][lag_day])

    df = df[lags:]

    for feature in features:
        df_feature = pd.DataFrame({
            feature: features[feature]
        }, index=dates[lags:])

        df = pd.merge(df, df_feature, left_index=True, right_index=True)

    df = pd.get_dummies(df, columns=['day', 'month', 'weekday'])

    return df
