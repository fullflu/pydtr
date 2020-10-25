import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from pandas._testing import assert_frame_equal
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pydtr.iqlearn.regression import IqLearnReg


class RegWrapperRule1(BaseEstimator, RegressorMixin):
    def __init__(self, thres: int, fit_value: int) -> None:
        self.thres = thres
        self.fit_value = fit_value

    def fit(self, X: pd.DataFrame, y: np.array):
        return self

    def predict(self, X):
        flag = X["L1"] < self.thres
        q_val = pd.Series(np.zeros(X.shape[0]))
        # if L1 >= thres;  then q(L1, 1) > q(L1, 0); else q(L1, 1) < q(L1, 0)
        q_val[flag] = X.loc[flag, "A1"] * 2 + (1 - X.loc[flag, "A1"]) * 1
        q_val[~flag] = X.loc[~flag, "A1"] * 1 + (1 - X.loc[~flag, "A1"]) * 2
        return q_val


class RegWrapperRule2(BaseEstimator, RegressorMixin):
    def __init__(self, thres: int, fit_value: int) -> None:
        self.thres = thres
        self.fit_value = fit_value

    def fit(self, X: pd.DataFrame, y: np.array):
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        flag = X["L1"] < self.thres
        q_val = pd.Series(np.zeros(X.shape[0]))
        # if L1 >= thres;  then q(L1, 1) < q(L1, 0); else q(L1, 1) > q(L1, 0)
        q_val[flag] = X.loc[flag, "A2"] * 1 + (1 - X.loc[flag, "A2"]) * 2
        q_val[~flag] = X.loc[~flag, "A2"] * 2 + (1 - X.loc[~flag, "A2"]) * 1
        return q_val


def test_iqlearn_regwrapper_rule():
    # setup params
    n = 10
    thres = int(n / 2)
    # sample rule base models
    model1 = RegWrapperRule1(thres, 1)
    model2 = RegWrapperRule2(thres, 2)
    # sample dataframe
    df = pd.DataFrame()
    df["L1"] = np.arange(n)
    df["A1"] = [0, 1] * int(n / 2)
    df["A2"] = [0] * int(n / 2) + [1] * int(n / 2)
    df["Y1"] = np.zeros(n)
    df["Y2"] = np.zeros(n)
    # set model info
    model_info = [
        {
            "model": model1,
            "action_dict": {"A1": [0, 1]},
            "feature": ["L1", "A1"],
            "outcome": "Y1"
        },
        {
            "model": model2,
            "action_dict": {"A2": [0, 1]},
            "feature": ["L1", "A1", "Y1", "A2"],
            "outcome": "Y2"
        }
    ]
    # fit model (dummy)
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info
    )
    dtr_model.fit(df)
    assert dtr_model.model_all[0].fit_value == 1
    assert dtr_model.model_all[1].fit_value == 2
    # predict optimal atcions
    action_1 = dtr_model.predict(df, 0)
    action_2 = dtr_model.predict(df, 1)
    action_all = dtr_model.predict_all_stages(df)
    # stage 1 test
    true_action_1 = [1] * int(n / 2) + [0] * int(n / 2)
    assert all([a == b for a, b in zip(action_1["A1"].tolist(), true_action_1)])
    # stage 2 test
    true_action_2 = [0] * int(n / 2) + [1] * int(n / 2)
    assert all([a == b for a, b in zip(action_2["A2"].tolist(), true_action_2)])
    # all stage test
    assert action_all.shape[0] == action_1.shape[0] * 2
    a1 = action_all.query("stage == 0")[["A1", "val"]].reset_index(drop=True)
    a2 = action_all.query("stage == 1")[["A2", "val"]].reset_index(drop=True)
    assert_frame_equal(action_1, a1)
    assert_frame_equal(action_2, a2)
    # function test
    q_val_1 = dtr_model._get_max_val_df(dtr_model.model_all[0], df, 0)
    assert all([a == b for a, b in zip(q_val_1["A1"].tolist(), true_action_1)])
    q_val_2 = dtr_model._get_max_val_df(dtr_model.model_all[1], df, 1)
    assert all([a == b for a, b in zip(q_val_2["A2"].tolist(), true_action_2)])
    # fit bootstrap model (dummy)
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info,
        n_bs=2
    )
    dtr_model.fit(df)
    assert dtr_model.model_all[0].fit_value == 1
    assert dtr_model.model_all[1].fit_value == 2
    # check bootstrap model
    assert dtr_model.models[0][0].fit_value == 1
    assert dtr_model.models[1][0].fit_value == 2
    assert len(dtr_model.models[0]) == 2


def test_iqlearn_rf():
    # setup params
    n = 10
    thres = int(n / 2)
    # rf models
    model1 = RandomForestRegressor()
    model2 = RandomForestRegressor()
    # sample dataframe
    df = pd.DataFrame()
    df["L1"] = np.arange(n)
    df["A1"] = [0, 1] * int(n / 2)
    df["A2"] = [0] * int(n / 2) + [1] * int(n / 2)
    df["Y1"] = np.zeros(n)
    df["Y2"] = np.zeros(n)
    # set model info
    model_info = [
        {
            "model": model1,
            "action_dict": {"A1": [0, 1]},
            "feature": ["L1", "A1"],
            "outcome": "Y1"
        },
        {
            "model": model2,
            "action_dict": {"A2": [0, 1]},
            "feature": ["L1", "A1", "Y1", "A2"],
            "outcome": "Y2"
        }
    ]
    # fit model
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info
    )
    dtr_model.fit(df)
    # predict optimal atcions
    action_1 = dtr_model.predict(df, 0)
    action_2 = dtr_model.predict(df, 1)
    action_all = dtr_model.predict_all_stages(df)
    # stage 1 test
    assert action_1.shape[0] == df.shape[0]
    # stage 2 test
    assert action_2.shape[0] == df.shape[0]
    # all stage test
    assert action_all.shape[0] == action_1.shape[0] * 2
    a1 = action_all.query("stage == 0")[["A1", "val"]].reset_index(drop=True)
    a2 = action_all.query("stage == 1")[["A2", "val"]].reset_index(drop=True)
    assert_frame_equal(action_1, a1)
    assert_frame_equal(action_2, a2)
    # fit bootstrap model
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info,
        n_bs=2
    )
    dtr_model.fit(df)
    assert len(dtr_model.models[0]) == 2


def test_iqlearn_rf_multiple_actions():
    # setup params
    n = 10
    thres = int(n / 2)
    # rf models
    model1 = RandomForestRegressor()
    model2 = RandomForestRegressor()
    # sample dataframe
    df = pd.DataFrame()
    df["L1"] = np.arange(n)
    df["A1"] = [0, 1] * int(n / 2)
    df["A2"] = [0] * int(n / 2) + [1] * int(n / 2)
    df["Y1"] = np.zeros(n)
    df["Y2"] = np.zeros(n)
    # set model info
    model_info = [
        {
            "model": model1,
            "action_dict": {"A1": [0, 1]},
            "feature": ["L1", "A1"],
            "outcome": "Y1"
        },
        {
            "model": model2,
            "action_dict": {"A1": [0, 1], "A2": [0, 1]},
            "feature": ["L1", "A1", "Y1", "A2"],
            "outcome": "Y2"
        }
    ]
    # fit model
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info
    )
    dtr_model.fit(df)
    # predict optimal atcions
    action_1 = dtr_model.predict(df, 0)
    action_2 = dtr_model.predict(df, 1)
    action_all = dtr_model.predict_all_stages(df)
    # stage 1 test
    assert action_1.shape[0] == df.shape[0]
    # stage 2 test
    assert action_2.shape[0] == df.shape[0]
    # all stage test
    assert action_all.shape[0] == action_1.shape[0] * 2
    a1 = action_all.query("stage == 0")[["A1", "val"]].reset_index(drop=True)
    a2 = action_all.query("stage == 1")[["A1", "A2", "val"]].reset_index(drop=True)
    assert_frame_equal(action_1, a1)
    assert_frame_equal(action_2, a2)
    # fit bootstrap model
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info,
        n_bs=2
    )
    dtr_model.fit(df)
    assert len(dtr_model.models[0]) == 2


def test_iqlearn_rf_ordinalencoder():
    # setup params
    n = 30
    thres = int(n / 2)
    # rf models
    model1 = RandomForestRegressor()
    model2 = RandomForestRegressor()
    # sample dataframe
    df = pd.DataFrame()
    df["L1"] = np.arange(n)
    df["A1"] = [0, 1, 2] * int(n / 3)
    df["A2"] = [0] * int(n / 3) + [1] * int(n / 3) + [3] * int(n / 3)
    df["Y1"] = np.zeros(n)
    df["Y2"] = np.zeros(n)
    # set model info
    model_info = [
        {
            "model": model1,
            "action_dict": {"A1": [0, 1, 2]},
            "feature": ["L1", "A1"],
            "outcome": "Y1"
        },
        {
            "model": model2,
            "action_dict": {"A2": [0, 1, 3]},
            "feature": ["L1", "A1", "Y1", "A2"],
            "outcome": "Y2"
        }
    ]
    # fit model
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info
    )
    dtr_model.fit(df)
    # predict optimal atcions
    action_1 = dtr_model.predict(df, 0)
    action_2 = dtr_model.predict(df, 1)
    action_all = dtr_model.predict_all_stages(df)
    # stage 1 test
    assert action_1.shape[0] == df.shape[0]
    # stage 2 test
    assert action_2.shape[0] == df.shape[0]
    # all stage test
    assert action_all.shape[0] == action_1.shape[0] * 2
    a1 = action_all.query("stage == 0")[["A1", "val"]].reset_index(drop=True)
    a2 = action_all.query("stage == 1")[["A2", "val"]].reset_index(drop=True)
    assert_frame_equal(action_1, a1)
    assert_frame_equal(action_2, a2)
    # fit bootstrap model
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info,
        n_bs=2
    )
    dtr_model.fit(df)
    assert len(dtr_model.models[0]) == 2


def test_iqlearn_pipeline_category_encoder():
    # setup params
    n = 30
    thres = int(n / 2)
    # statsmodels
    model1 = Pipeline(
        [
            ("ce0", ce.OneHotEncoder(cols=["A1"])),
            ("scale", StandardScaler()),
            ("model", Ridge())
        ]
    )
    model2 = Pipeline(
        [
            ("ce0", ce.OneHotEncoder(cols=["A1", "A2"])),
            ("scale", StandardScaler()),
            ("model", Ridge())
        ]
    )
    # sample dataframe
    df = pd.DataFrame()
    df["L1"] = np.arange(n)
    df["A1"] = ["A", "B", "C"] * int(n / 3)
    df["A2"] = ["A"] * int(n / 3) + ["C"] * int(n / 3) + ["D"] * int(n / 3)
    df["Y1"] = np.zeros(n)
    df["Y2"] = np.zeros(n)
    # set model info
    model_info = [
        {
            "model": model1,
            "action_dict": {"A1": ["A", "B", "C"]},
            "feature": ["L1", "A1"],
            "outcome": "Y1"
        },
        {
            "model": model2,
            "action_dict": {"A2": ["A", "C", "D"]},
            "feature": ["L1", "A1", "Y1", "A2"],
            "outcome": "Y2"
        }
    ]
    # fit model
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info
    )
    dtr_model.fit(df)
    # predict optimal atcions
    action_1 = dtr_model.predict(df, 0)
    action_2 = dtr_model.predict(df, 1)
    action_all = dtr_model.predict_all_stages(df)
    # stage 1 test
    assert action_1.shape[0] == df.shape[0]
    # stage 2 test
    assert action_2.shape[0] == df.shape[0]
    # all stage test
    assert action_all.shape[0] == action_1.shape[0] * 2
    a1 = action_all.query("stage == 0")[["A1", "val"]].reset_index(drop=True)
    a2 = action_all.query("stage == 1")[["A2", "val"]].reset_index(drop=True)
    assert_frame_equal(action_1, a1)
    assert_frame_equal(action_2, a2)
