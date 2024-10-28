# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Text, Tuple, Union
from ...model.base import ModelFT
from ...data.dataset import DatasetH
# from ...data.dataset.handler import DataHandlerLP
# from ...model.interpret.base import LightGBMFInt
# from ...data.dataset.weight import Reweighter
# from qlib.workflow import R
from ...model.base import Model
from ...model.interpret.base import FeatureInt

class LGBModel(Model, FeatureInt):
    """LightGBM Model"""

    def __init__(self, objective="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs):
        
        self.params = {"objective": objective}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def fit(
        self,
        dataset: DatasetH,
        **kwargs,
    ):
        # prepare dataset for lgb training and evaluation
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key="learn"
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        # Lightgbm need 1D array as its label
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        dtrain = lgb.Dataset(x_train.values, label=y_train)
        dvalid = lgb.Dataset(x_valid.values, label=y_valid)
        
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds 
        )
        # fit the model
        self.model = lgb.train(
            self.params,
            dtrain,           
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[early_stopping_callback],    
            **kwargs
        )


    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key="infer")
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)


    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
    
            return pd.Series(
                self.model.feature_importance(*args, **kwargs), index=self.model.feature_name()
            ).sort_values(  # pylint: disable=E1101
                ascending=False
            )