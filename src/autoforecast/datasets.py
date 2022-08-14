import pandas as pd
from sys import argv
from utils import load_dataset


def load_denmark(resample=None) -> pd.DataFrame:
    df = load_dataset('sample_data/dk.csv')
    if resample is not None:
        if resample.lower() == 'daily':
            resampled_df = df.resample('D').sum()
            return resampled_df
        elif resample.lower() == "weekly":
            return df.resample("W").sum()
        elif resample.lower() == "monthly":
            return df.resample("M").sum()
    else:
        return df


def load_finland(resample=None) -> pd.DataFrame:
    df = load_dataset('sample_data/fi.csv')
    if resample is not None:
        if resample.lower() == 'daily':
            resampled_df = df.resample('D').sum()
            return resampled_df
    else:
        return df


def load_germany(resample=None) -> pd.DataFrame:
    df = load_dataset('sample_data/de.csv')
    if resample is not None:
        if resample.lower() == 'daily':
            resampled_df = df.resample('D').sum()
            return resampled_df
    else:
        return df


def load_norway(resample=None) -> pd.DataFrame:
    df = load_dataset('sample_data/no.csv')
    if resample is not None:
        if resample.lower() == 'daily':
            resampled_df = df.resample('D').sum()
            return resampled_df
    else:
        return df


def load_sweden(resample=None) -> pd.DataFrame:
    df = load_dataset('sample_data/se.csv')
    if resample is not None:
        if resample.lower() == 'daily':
            resampled_df = df.resample('D').sum()
            return resampled_df
    else:
        return df


def main():
    if '-t' in argv:
        print(load_denmark())
        print(load_sweden())
        print(load_norway())
        print(load_germany())
        print(load_finland())


if __name__ == '__main__':
    main()
