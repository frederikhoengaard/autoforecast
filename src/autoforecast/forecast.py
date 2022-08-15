import pandas as pd
from pandas import DataFrame
from xgboost import XGBRegressor
from typing import Optional, Tuple, List
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from utils import *
from datasets import load_denmark


class AutoModel:
    """
    This class implements the automated pipeline for autoforecast.
    The main methods to consider for interfacing are:
        - fit()
        - predict()
        - recursive_forecast()
    """

    def __init__(self, time_allowance=3600, verbose=0):
        self.param_grid = {
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
            'gamma': [0, 0.25, 0.5, 1.0],
            'n_estimators': [100, 200, 300, 500]
        }
        self.data = None
        self._train = None
        self.target = None
        self.time_allowance = time_allowance
        self.time_left = time_allowance
        self.holdout = None
        self.ensemble = None
        self.ensemble_weights = None
        self._is_fitted = False
        self._verbose = verbose

    def _set_model_weights(self, models: List[Tuple]) -> List:
        """
        Private method to set the weight of each model for
        the blender part of the pipeline.

        :param models: List of Tuples of a model name and its instance
        :returns: List of weights for each model
        """
        out = []

        scores = []
        for name, model in models:
            scores.append(model.best_score_)

        weights = get_inverse_weights(scores)

        for i in range(len(models)):
            out.append((models[i][0], models[i][1], weights[i]))
        return out

    def _train_xgb_regressor(
            self,
            features: DataFrame,
            labels: DataFrame,
            params: dict,
            n_folds=5
    ) -> RandomizedSearchCV:
        """
        Private method to optimize hyperparameters for a number of XGBregressors

        :param features: DataFrame of features
        :param labels: DataFrame of labels
        :param params: Search grid for XGBRegressor
        :n_folds: Number of cross-validation folds to make
        :returns: A RandomizedSearchCV object containing the optimized regressor
        """
        xgbtuned = XGBRegressor(verbosity=self._verbose)

        tscv = TimeSeriesSplit(n_splits=n_folds)  # time series cross validation split
        xgbtunedreg = RandomizedSearchCV(
            xgbtuned,
            param_distributions=params,
            scoring='neg_mean_squared_error',
            n_iter=25,
            n_jobs=-1,
            cv=tscv,
            verbose=self._verbose,
        )
        xgbtunedreg.fit(features, labels)
        return xgbtunedreg

    def _fit_ensemble(self, features, labels, param_grid: dict, n_regressors=5) -> List[Tuple]:
        """
        Private method to fit a number of regressors

        :param features: DataFrame of features
        :param labels: DataFrame of labels
        :param param_grid: Search grid for the XGBRegressors
        :n_regressors: Number of regressors to fit
        :returns: List of Tuples of a model name and its instance
        """
        models = []

        for i in range(n_regressors):
            models.append((f"xgb_{i}", self._train_xgb_regressor(features, labels, params=param_grid)))
        return models

    def _ensemble_predict(self, models, features) -> DataFrame:
        predictions = {}

        for name, model, weight in models:
            predictions[name] = model.best_estimator_.predict(features) * weight

        predictions = pd.DataFrame(predictions, index=features.index)
        predictions[self.target] = predictions.sum(axis=1)

        return predictions[self.target]

    def fit(self, data: DataFrame, target: str, holdout_size=None):
        """
        Fit the AutoModel predictor

        :param data: DataFrame of both features and labels
        :param target: name of target column/variable
        :param holdout_size: int of number of observations to set aside from data
            for holdout
        """
        self.data = featuremaker(data, target)
        self.target = target

        if holdout_size:
            self._train, self.holdout = train_test_split_time(self.data, holdout_size)
        else:
            self._train = self.data

        train_x, train_y = self._train.iloc[:, 1:], self._train.iloc[:, :1]

        self.ensemble = self._fit_ensemble(train_x, train_y, self.param_grid, n_regressors=5)
        self.ensemble_weights = self._set_model_weights(self.ensemble)

        self._is_fitted = True

    def predict(self, features: DataFrame) -> DataFrame:
        """
        Predict target value given a feature vector

        :param features: DataFrame of test set features
        :returns: DataFrame of predictions
        """
        assert self._is_fitted

        return self._ensemble_predict(self.ensemble_weights, features)

    def recursive_forecast(self, horizon: int) -> Tuple[DataFrame, DataFrame]:
        """
        This method returns recursive forecasts from the predictor i.e. a sequence of
        predictions where each succesive predictions is based on lag values of previous
        predictions.

        :param horizon: number of recursive forecast to perform
        :param verbose: verbosity level
        :returns: Tuple containing full dataset appended with predictions and
            DataFrame of predictions isolated
        """
        assert self._is_fitted
        results = {"date": [], "prediction": []}

        train_data = self._train

        for i in range(horizon):
            x = make_features_for_next(train_data, self.target)
            next_date = x.index[0]
            prediction = float(self.predict(x))
            x[self.target] = prediction
            train_data = pd.concat([train_data, x])

            results["date"].append(next_date)
            results["prediction"].append(prediction)
            if self._verbose > 2:
                print(f"Predicted {prediction} for time {next_date}")

        results = pd.DataFrame(results, index=results["date"]).iloc[:, 1:]

        return train_data, results
