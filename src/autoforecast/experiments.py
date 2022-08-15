from sklearn.metrics import mean_absolute_percentage_error

from datasets import load_denmark
from forecast import AutoModel
from utils import *


def predict_danish_power_consumption_2015(recursive=False):

    dk_daily = load_denmark(resample="daily")
    train, test = train_test_split_time(dk_daily, 365)

    am = AutoModel(verbose=2)
    print("fitting model")
    am.fit(dk_daily, "consumption_MW", holdout_size=365)

    y_pred = am.recursive_forecast(365)

    print(mean_absolute_percentage_error(test, y_pred[1]))


predict_danish_power_consumption_2015()