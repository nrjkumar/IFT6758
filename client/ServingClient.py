
# import os,sys
# import importlib

# #dir = os.chdir(os.path.dirname(os.path.abspath(__file__)))
# # os.chdir('../')
# # os.chdir('../')
# # os.chdir('../')
# import json
# import requests
# import pandas as pd
# import logging
# #os.chdir(os.path.dirname(os.path.abspath(__file__)))
# import importlib


# currentdir = os.path.dirname(os.path.realpath(__file__))


# #parentdir = os.path.dirname(currentdir)
# sys.path.append(currentdir)
# #print(currentdir)

# from feature_lists import *
# from models.utils import load_stream_data
# from sklearn.preprocessing import StandardScaler
# import pickle
# import numpy as np

# logger = logging.getLogger(__name__)


# class ServingClient:
#     def __init__(self, ip: str = "0.0.0.0", port: int = 8080, features=None):
#         self.base_url = f"http://{ip}:{port}"
#         logger.info(f"Initializing client; base URL: {self.base_url}")

#         # if features is None:
#         #     features = ["distance"]
#         # self.features = feature_list_lgbm
#         # self.scaler = pickle.load(open('parentdir/scaler.pkl','rb'))

#         self.model_registries_to_file_name = {
#             '6-lgbm': ('6-LGBM.pkl', feature_list_lgbm),
#             '5-2-grid-search-model': ('tuned_xgb_model.pkl', feature_list_xgb),
#             '6-2-nn-tuned-model': ('tuned_nn_model.pkl', feature_list_nn),
#             '6-4-stacked-trained-tuned-model': ('tuned_stacked_trained_model.pkl',feature_list_stack_trained),
#             '3-3-angle-dist-logreg-model':('LogReg_dist_angle_model.pkl',feature_list_logreg)
#         }

#     def predict(self, X: pd.DataFrame) -> pd.DataFrame:
#         """
#         Formats the inputs into an appropriate payload for a POST request, and queries the
#         prediction service. Retrieves the response from the server, and processes it back into a
#         dataframe that corresponds index-wise to the input dataframe.
        
#         Args:
#             X (Dataframe): Input dataframe to submit to the prediction service.
#         """
        
#         X = load_stream_data(features = feature_list,data = X, scaler =self.scaler)
        
#         for column in self.features:
#             if column not in X.columns:
#                 X[column] = np.zeros(X.shape[0])
        
#         X = X[self.features] #need to make sure columns are in the same order as training time
#         r = requests.post(f"{self.base_url}/predict", json=X.to_json())

#         result = pd.DataFrame.from_dict(r.json())
#         return result

#     def logs(self) -> dict:
#         """Get server logs"""

#         r = requests.get(f"{self.base_url}/logs")
#         return r.json()

#     def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
#         """
#         Triggers a "model swap" in the service; the workspace, model, and model version are
#         specified and the service looks for this model in the model registry and tries to
#         download it. 
#         See more here:
#             https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
#         Args:
#             workspace (str): The Comet ML workspace
#             model (str): The model in the Comet ML registry to download
#             version (str): The model version to download
#         """

#         assert model in self.model_registries_to_file_name.keys(), f'model name must be in ' \
#                                                                     f'{self.model_registries_to_file_name.keys()} '
#         model_file_name = self.model_registries_to_file_name[model][0]
#         self.features = self.model_registries_to_file_name[model][1]
#         request = {'workspace': workspace, 'registry_name': model, 'model_name': model_file_name, 'version': version}
#         r = requests.post(f"{self.base_url}/download_registry_model", json=request)
#         return r.json()

#!/usr/bin/env python
# coding: utf-8

import json
import requests
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
#from .utils import dataload

logger = logging.getLogger(__name__)

# df = pd.read_csv('../ift6758-project-template-main/notebooks/final_df.csv')

class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 8501, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        logger.info(f"Initializing request to generate predictions")
        try:
            print("I am in Serving Client")
            X = dataload(X)
            X = X.astype(np.float64)
            r = requests.post(
                f"{self.base_url}/predict", 
                json=json.loads(X.to_json())
            )
            logger.info(f"Successfully generated predictions")
            return r.json()
        except Exception as e:
            print(e)
            return None

    def logs(self) -> dict:
        """Get server logs"""
        logger.info(f"Initializing request to server get logs")
        r = requests.get(
            f"{self.base_url}/logs"
        )
        logger.info(f"Server Logs fetched")
        return r.json()

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        logger.info(f"Initializing request to download the model{model}-{version}")
        self.workspace = workspace
        self.model = model
        self.version = version
        self.model_filename = f"{workspace}_{model}_{version}"
        r = requests.post(
            f"{self.base_url}/download_registry_model", 
            json= {'workspace': workspace, 'model': model, 'version': version}
        )
        logger.info(f"Successfully Downloaded Model")



def dataload(test_link):
    """
    This function is for loading the dataset.
    params: train_link
    return train_df
    """
    if not isinstance(test_link, pd.DataFrame):
        test_df = pd.read_csv(test_link)
    else:
        test_df = test_link
    print("I am in dataload")
    test_df.rename(columns={'game date': 'game_date', 'period time': 'period_time',
                             'game id': 'game_id', 'home team': 'home_team',
                             'away team': 'away_team', 'is goal': 'is_goal',
                             'team shot': 'team_shot', 'x shot': 'x_shot',
                             'y shot': 'y_shot', 'shot type': 'shot_type',
                             'empty net': 'empty_net', 'home goal': 'home_goal',
                             'away goal': 'away_goal', 'is rebound': 'is_rebound',
                             'game seconds': 'game_seconds',
                             'last event type': 'last_event_type',
                             'x last event': 'x_last_event',
                             'y last event': 'y_last_event',
                             'time from last event': 'time_from_last_event',
                             'num player home': 'num_player_home',
                             'num player away': 'num_player_away',
                             'time power play': 'time_power_play',
                             'shot distance': 'shot_distance',
                             'shot angle': 'shot_angle',
                             'change in shot angle': 'change_in_shot_angle',
                             'distance from last event': 'distance_from_last_event'
                             }, inplace=True)

    test_df = test_df[['game_seconds', 'shot_distance', 'shot_angle', 'period',
                         'x_shot', 'y_shot', 'shot_type', 'last_event_type',
                         'x_last_event', 'y_last_event', 'time_from_last_event',
                         'distance_from_last_event', 'is_rebound',
                         'change_in_shot_angle', 'speed']]
    test_df['is_rebound'] = np.where(test_df['is_rebound'] == False, 0, 1)
    # One hot-encoding for categorical variables
    transformer = make_column_transformer(
        (OneHotEncoder(), ['shot_type', 'last_event_type']),
        remainder='passthrough')
    transformed = transformer.fit_transform(test_df)
    transformed_X = pd.DataFrame(transformed,
                                 columns=transformer.get_feature_names_out())
    transformed_X.dropna(inplace=True)
    return transformed_X


def load_data(data=None, features=None):
    X_test = data[features]
    return X_test


# current_dir_path = os.getcwd()
# dir_path = os.path.join("\\".join(current_dir_path.split('\\')[:-2]), "data", "processed",
#                         "df_feature_engineering.csv")
# X_test = dataload(dir_path)

        # return r.json()
