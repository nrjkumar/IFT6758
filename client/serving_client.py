
import os,sys
import importlib

#dir = os.chdir(os.path.dirname(os.path.abspath(__file__)))
# os.chdir('../')
# os.chdir('../')
# os.chdir('../')
import json
import requests
import pandas as pd
import logging
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib


currentdir = os.path.dirname(os.path.realpath(__file__))


#parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)
#print(currentdir)

from feature_lists import *
from models.utils import load_stream_data
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 8080, features=feature_list_lgbm):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        # if features is None:
        #     features = ["distance"]
        # self.features = feature_list_lgbm
        # self.scaler = pickle.load(open('parentdir/scaler.pkl','rb'))

        self.model_registries_to_file_name = {
            '6-lgbm': ('6-LGBM.pkl', feature_list_lgbm),
            '5-2-grid-search-model': ('tuned_xgb_model.pkl', feature_list_xgb),
            '6-2-nn-tuned-model': ('tuned_nn_model.pkl', feature_list_nn),
            '6-4-stacked-trained-tuned-model': ('tuned_stacked_trained_model.pkl',feature_list_stack_trained),
            '3-3-angle-dist-logreg-model':('LogReg_dist_angle_model.pkl',feature_list_logreg)
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        
        X = load_stream_data(features = feature_list,data = X, scaler =self.scaler)
        
        for column in self.features:
            if column not in X.columns:
                X[column] = np.zeros(X.shape[0])
        
        X = X[self.features] #need to make sure columns are in the same order as training time
        r = requests.post(f"{self.base_url}/predict", json=X.to_json())

        result = pd.DataFrame.from_dict(r.json())
        return result

    def logs(self) -> dict:
        """Get server logs"""

        r = requests.get(f"{self.base_url}/logs")
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

        assert model in self.model_registries_to_file_name.keys(), f'model name must be in ' \
                                                                    f'{self.model_registries_to_file_name.keys()} '
        model_file_name = self.model_registries_to_file_name[model][0]
        self.features = self.model_registries_to_file_name[model][1]
        request = {'workspace': workspace, 'registry_name': model, 'model_name': model_file_name, 'version': version}
        r = requests.post(f"{self.base_url}/download_registry_model", json=request)
        return r.json()



