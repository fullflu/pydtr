import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from pandas._testing import assert_frame_equal

from pydtr.iqlearn.regression import IqLearnReg


def test_iqlearn_sm():
    # setup params
    n = 10
    thres = int(n / 2)
    # sample rule base models
    model1 = "p_outcome ~ L1 * A1"
    model2 = "p_outcome ~ L1 + A1 + Y1 * A2"
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
    # fit bootstrap model (dummy)
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info,
        n_bs=2
    )
    dtr_model.fit(df)
    assert len(dtr_model.models[0]) == 2


def test_iqlearn_sm_multiple_actions():
    # setup params
    n = 10
    thres = int(n / 2)
    # sample rule base models
    model1 = "p_outcome ~ L1 * A1"
    model2 = "p_outcome ~ L1 + A1 + Y1 * A2"
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
    # fit model (dummy)
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
    # fit bootstrap model (dummy)
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info,
        n_bs=2
    )
    dtr_model.fit(df)
    assert len(dtr_model.models[0]) == 2


def test_iqlearn_sm_multinomial_action():
    # setup params
    n = 30
    thres = int(n / 2)
    # sample rule base models
    model1 = "p_outcome ~ L1 * C(A1)"
    model2 = "p_outcome ~ L1 + A1 + Y1 * C(A2)"
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
    # fit model (dummy)
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
    # fit bootstrap model (dummy)
    dtr_model = IqLearnReg(
        n_stages=2,
        model_info=model_info,
        n_bs=2
    )
    dtr_model.fit(df)
    assert len(dtr_model.models[0]) == 2
