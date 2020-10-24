import itertools
from typing import Dict
from abc import ABCMeta, abstractmethod


import pandas as pd
import numpy as np
from sklearn.utils import resample


class IqLearnBase(object):
    """
    Base class of iterative q learning

    Attributes
    ----------
    n_stages : int
        the number of total stages
    model_info : list of dict
        list of model information dictionary of each stage
          - model: string or sklearn-based instance
              model object
          - action_dict: dict
              action varible -> domain of the variable
          - feature: str
              feature variable names
          - outcome: str
              a outcome variable name
    n_bs : int
        the number of bootstrap sampling (0 means no bootstrap)
    """
    def __init__(self, n_stages: int, model_info: list, n_bs: int = 0) -> None:
        """
        Parameters
        ----------
        n_stages : int
            The number of total stages
        model_info : list of dict
            List of model information dictionary of each stage
            - model: string or sklearn-based instance
                Model object
            - action_dict: dict
                Action varible -> domain of the variable
            - feature: str
                Feature variable names
            - outcome: str
                An outcome variable name
        n_bs : int
            The number of bootstrap sampling (0 means no bootstrap)
        """
        assert n_stages == len(model_info), "n_stages must be the same as the length of model_info"
        self.n_stages = n_stages
        self.model_info = model_info
        self.n_bs = n_bs
        self._init_set()

    def _get_max_val_df(self, model, X: pd.DataFrame, t: int) -> pd.DataFrame:
        """
        Get maximum value of Q function of stage t (ref: max_{A} Q(X, A))
        """
        action_dict = self.model_info[t]["action_dict"]
        keys = list(action_dict.keys())
        # dataframe to store pseudo outcomes
        max_val_df = pd.DataFrame(columns=keys)
        # initialize maximum value
        max_val_df["val"] = [-1 * np.inf] * X.shape[0]
        tmp_df = X.reset_index(drop=True).copy()
        for tmp_dict in [dict(zip(action_dict, vs)) for vs in itertools.product(*action_dict.values())]:
            # consider multiple action variables
            tmp_keys = list(tmp_dict.keys())
            tmp_values = list(tmp_dict.values())
            if len(tmp_values) == 1:
                tmp_values = tmp_values[0]
            # raise ValueError
            # set action values
            tmp_df[tmp_keys] = tmp_values
            # predict q value for the action values
            val_a = model.predict(tmp_df)
            # update optimal actions and q values
            flag = max_val_df["val"] < val_a
            max_val_df.loc[flag.values, tmp_keys] = tmp_values
            max_val_df.loc[flag.values, "val"] = val_a[flag.values]
            # reset independent variables
            tmp_df = X.reset_index(drop=True).copy()
        return max_val_df

    def _get_p_outcome(self, model: object, X: pd.DataFrame, y: pd.Series, t: int) -> pd.Series:
        """
        Get pseudo-outcome of stage t
        """
        # return pseudo outcome
        max_val_df = self._get_max_val_df(model, X, t + 1)
        return max_val_df["val"].values + y.values

    @staticmethod
    def _sample_bs(df: pd.DataFrame, size_bs: int = -1) -> pd.DataFrame:
        """
        bootstrap sampling
        """
        # bootstrap sampling
        if size_bs == -1:
            size_bs = df.shape[0]
        return resample(df, n_samples=size_bs)

    def fit(self, df: pd.DataFrame):
        """
        Fit dtr models

        Parameters
        ----------
        df: pandas.dataframe
            input data (each row contains all stage information of each individual)

        Returns
        -------
        self
        """
        # fit models using all data
        for t in reversed(range(self.n_stages)):
            print("Stage: {}".format(t))
            X = df[self.model_info[t]["feature"]]
            y = df[self.model_info[t]["outcome"]]
            if t == self.n_stages - 1:
                p_outcome = y.values
            else:
                X2 = df[self.model_info[t + 1]["feature"]]
                y2 = df[self.model_info[t + 1]["outcome"]]
                p_outcome = self._get_p_outcome(self.model_all[t + 1], X2, y2, t)
            self._fit_model_all_data(X, p_outcome, t)

        # fit models using bootstrap
        for i in range(self.n_bs):
            df_i = self._sample_bs(df)
            for t in reversed(range(self.n_stages)):
                # extract feature and outcome
                X = df_i[self.model_info[t]["feature"]]
                y = df_i[self.model_info[t]["outcome"]]
                if t == self.n_stages - 1:
                    p_outcome = y.values
                else:
                    X2 = df_i[self.model_info[t + 1]["feature"]]
                    y2 = df_i[self.model_info[t + 1]["outcome"]]
                    p_outcome = self._get_p_outcome(self.model_all[t + 1], X2, y2, t)
                # fit model of stage t
                self._fit_model(X, p_outcome, t, i)
        return self

    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, p_outcome: np.array, t: int, i: int) -> None:
        pass

    @abstractmethod
    def _fit_model_all_data(self, X: pd.DataFrame, p_outcome: np.array, t: int) -> None:
        pass

    def _init_set(self) -> None:
        self.model_all = {}
        self.models = {}
        for t in range(self.n_stages):
            # TODO: check model type
            # self._check_model_type(self.model_info[t]["model"])
            self.model_all[t] = None
            if self.n_bs:
                self.models[t] = np.array([None] * (self.n_bs))

    def get_params(self) -> pd.DataFrame:
        # get estimated parameters
        params = pd.DataFrame()
        for t in reversed(range(self.n_stages)):
            if type(self.model_info[t]["model"]) == str:
                tmp_df = pd.melt(pd.DataFrame([i.params for i in self.models[t]]))
                tmp_df["stage"] = t
                params = pd.concat([params, tmp_df])
        return params

    def predict(self, df: pd.DataFrame, t: int) -> pd.DataFrame:
        """
        Predict optimal treatment of stage t for each row

        Parameters
        ----------
        df: pandas.dataframe
            input data (each row contains all stage information of each individual)
        t: int
            stage

        Returns
        -------
        max_val_df: pandas.dataframe
            optimal treatments and their q-values
        """
        # return optimal actions for the specified stage
        X = df[self.model_info[t]["feature"]].copy()
        max_val_df = self._get_max_val_df(self.model_all[t], X, t)
        return max_val_df

    def predict_all_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict optimal treatment of all stages for each row

        Parameters
        ----------
        df: pandas.dataframe
            input data (each row contains all stage information of each individual)
        t: int
            stage

        Returns
        -------
        max_df: pandas.dataframe
            optimal treatments, their q-values, and stage information
        """
        # return optimal actions for all stages
        max_df = pd.DataFrame()
        for t in range(self.n_stages):
            X = df[self.model_info[t]["feature"]]
            max_val_df = self._get_max_val_df(self.model_all[t], X, t)
            max_val_df["stage"] = t
            max_val_df["row_id"] = np.arange(X.shape[0])
            max_df = pd.concat([max_df, max_val_df], sort=True).reset_index(drop=True)
        return max_df
