from datetime import timedelta

import pandas as pd
from pandas import DataFrame


def train_test_split_time(df, test_periods):
    n_obs = len(df)
    train = df[:n_obs-test_periods]
    test = df[n_obs-test_periods:]
    return train, test


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


def get_inverse_weights(errors: list) -> list:
    errors = [abs(error) for error in errors]
    total_errors = sum(errors)
    weights = [num/total_errors for num in errors]
    weights_reciprocal = [1/w for w in weights]
    total_reciprocals = sum(weights_reciprocal)

    return [num/total_reciprocals for num in weights_reciprocal]


def make_features_for_next(df, target):
    last_obs = df.iloc[-1]
    next_date = df.index[-1] + timedelta(days=1)

    date_features = {
        'day': next_date.day,
        'month':next_date.month,
        'weekday':next_date.weekday(),
        'is_weekend': 1 if next_date.weekday() > 4 else 0
    }

    n_lags = len([item for item in df.columns if "lag" in item])

    feature_dict = {
        "is_weekend": date_features["is_weekend"]
    }

    for i in range(1, n_lags+1):
        if i == 1:
            feature_dict[f"lag_{i}"] = last_obs[target]
        else:
            feature_dict[f"lag_{i}"] = last_obs[f"lag_{i-1}"]

    for i in range(1,32):
        if date_features["day"] == i:
            feature_dict[f"day_{i}"] = 1
        else:
            feature_dict[f"day_{i}"] = 0

    for i in range(1,13):
        if date_features["month"] == i:
            feature_dict[f"month_{i}"] = 1
        else:
            feature_dict[f"month_{i}"] = 0

    for i in range(7):
        if date_features["weekday"] == i:
            feature_dict[f"weekday_{i}"] = 1
        else:
            feature_dict[f"weekday_{i}"] = 0

    return pd.DataFrame(feature_dict, index=[next_date])