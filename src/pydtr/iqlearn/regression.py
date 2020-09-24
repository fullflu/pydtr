import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from .base import IqLearnBase


class IqLearnReg(IqLearnBase):
    def _fit_model(self, X: pd.DataFrame, p_outcome: np.array, t: int, i: int) -> None:
        """
        Fit dtr model of stage t and bootstrap i

        Parameters
        ----------
        X: pandas.dataframe
            input data (each row contains feature of each individual)
        p_outcome: np.array
            pseudo outcome of stage t
        t: int
            stage
        i: int
            bootstrap index

        Returns
        -------
        None
        """
        if type(self.model_info[t]["model"]) == str:
            df = X.copy()
            df["p_outcome"] = p_outcome
            self.models[t][i] = smf.ols(formula=self.model_info[t]["model"], data=df).fit()
        else:
            self.models[t][i] = self.model_info[t]["model"].fit(X, p_outcome)

    def _fit_model_all_data(self, X: pd.DataFrame, p_outcome: np.array, t: int) -> None:
        """
        Fit dtr model of stage t

        Parameters
        ----------
        X: pandas.dataframe
            input data (each row contains feature of each individual)
        p_outcome: np.array
            pseudo outcome of stage t
        t: int
            stage

        Returns
        -------
        None
        """
        if type(self.model_info[t]["model"]) == str:
            df = X.copy()
            df["p_outcome"] = p_outcome
            self.model_all[t] = smf.ols(formula=self.model_info[t]["model"], data=df).fit()
        else:
            self.model_all[t] = self.model_info[t]["model"].fit(X, p_outcome)
