# pydtr

[![CircleCI](https://circleci.com/gh/fullflu/pydtr.svg?style=shield)](https://app.circleci.com/pipelines/github/fullflu/pydtr)
[![codecov](https://codecov.io/gh/fullflu/pydtr/branch/master/graph/badge.svg)](https://codecov.io/gh/fullflu/pydtr)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


## Description

This is a python library to conduct a dynamic treatment regime ([DTR](https://en.wikipedia.org/wiki/Dynamic_treatment_regime)), `pydtr`.

A DTR is a paradigm that attempts to select optimal treatments adaptively for individual patients.

Pydtr enables you to implement DTR methods easily by using sklearn-based interfaces.

|                Method                 |  Single binary treatment   |  Multiple treatments  |    Multinomial treatment   |  Continuous treatment  |  Modeling flexibility  |  Interpretability  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
|  IqLearnReg <br> (with sklearn)      |  :white_check_mark:   |   :white_check_mark:  |   :white_check_mark: <br>(with pipeline)  |    |   :white_check_mark: <br>(with arbitrary regression models)  |       |
|  IqLearnReg <br> (with statsmodels)  |  :white_check_mark:   |   :white_check_mark:  |   :white_check_mark:       |    | limited to OLS   |    :white_check_mark: <br>(with confidence intervals)  |
| GEstimation | WIP | | WIP | WIP | WIP | WIP |

`IqLearnReg` means a regression method of iterative q-learning.

When there are categorical independent variables and you use a sklearn model as a regression function, you need to encode the categorical variables before using the model.

We recommend to encode categorical variables by `category_encoders` and combine the encoders with the sklearn model by `sklearn.pipeline`.

G-estimation, a famous method of DTR, is now unavailable.

## Requirements

- python>=3.6
- pandas>=1.1.2
- scikit-learn>=0.23.2
- numpy>=1.19.2
- statsmodels>=0.12.0

## Installation

### From pypi

```
pip install pydtr
```

### From source

```
git clone https://github.com/fullflu/pydtr.git
cd pydtr
python setup.py install
```

## Usage

### Iterative Q Learning (IqLearnReg)

You need to import libraries and prepare data.

```python
# import
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pydtr.iqlearn.regression import IqLearnReg

# create sample dataframe
n = 10
thres = int(n / 2)
df = pd.DataFrame()
df["L1"] = np.arange(n)
df["A1"] = [0, 1] * int(n / 2)
df["A2"] = [0] * int(n / 2) + [1] * int(n / 2)
df["Y1"] = np.zeros(n)
df["Y2"] = np.zeros(n)
```

You can use sklearn-based models.

```python
# set model info
model_info = [
    {
        "model": RandomForestRegressor(),
        "action_dict": {"A1": [0, 1]},
        "feature": ["L1", "A1"],
        "outcome": "Y1"
    },
    {
        "model": RandomForestRegressor(),
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
opt_action_stage_1 = dtr_model.predict(df, 0)
opt_action_stage_2 = dtr_model.predict(df, 1)
opt_action_all_stages = dtr_model.predict_all_stages(df)
```

You can also use statsmodels-based models.

```python
# set model info
model_info = [
    {
        "model": "p_outcome ~ L1 * A1",
        "action_dict": {"A1": [0, 1]},
        "feature": ["L1", "A1"],
        "outcome": "Y1"
    },
    {
        "model": "p_outcome ~ L1 + A1 + Y1 * A2",
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
opt_action_stage_1 = dtr_model.predict(df, 0)
opt_action_stage_2 = dtr_model.predict(df, 1)
opt_action_all_stages = dtr_model.predict_all_stages(df)
```

Please see [examples](https://github.com/fullflu/pydtr/blob/master/examples/) to get more information.

## Authors

- [@fullflu](https://github.com/fullflu) 

## Contributors

Please feel free to create issues or to send pull-requests!

If all checkes have passed in pull-requests, I will merge and release them.

## License

[BSD](https://github.com/fullflu/pydtr/blob/master/LICENSE)


## Structure

```
├── .circleci
│   ├── config.yml
├── .github
│   ├── CODEOWNERS
├── LICENSE
├── MANIFEST.IN
├── Makefile
├── README.md
├── setup.cfg
├── setup.py
├── src
│   ├── pydtr
│   │   ├── __init__.py
│   │   └── iqlearn
│   │       ├── __init__.py
│   │       ├── base.py
│   │       └── regression.py
└── tests
    ├── test_iqlearn_sklearn_predict.py
    └── test_iqlearn_sm_predict.py
```

## References

- Chakraborty, B, Moodie, EE. *Statistical Methods for Dynamic Treatment Regimes.* Springer, New York, 2013.
