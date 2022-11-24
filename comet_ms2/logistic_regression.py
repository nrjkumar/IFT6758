from comet_ml import Experiment
from create_figure import *

import os, sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pickle
from cometconf import experiment
import pandas as pd
import numpy as np

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import utils
import create_figure


 
### setting x_train x_val only once to ensure same dimension and same removal of NaN
feature_list = ['shot_distance', 'shot_angle']
X_train, y_train, X_val, y_val, _, _ = utils.load_data(features=feature_list)

with experiment.context_manager("Distance LogReg"): 
    ### Q3.1 BASE MODEL WITH ONLY DISTANCE
    X_train_dist = pd.DataFrame(X_train['shot_distance'])
    X_valid_dist = pd.DataFrame(X_val['shot_distance'])
    clf_dist = LogisticRegression()
    clf_dist.fit(X_train_dist, y_train)
    
    predictions_distance = clf_dist.predict_proba(X_valid_dist)
    predictions_labl_distance = clf_dist.predict(X_valid_dist)
    accuracy_dist = clf_dist.score(X_valid_dist, y_val)
     
    # save model
    file_name = "LogReg_dist_model.pkl"   
    pickle.dump(clf_dist, open(file_name, "wb"))
     
    #compute metrics
    f1 = f1_score(y_val, predictions_labl_distance)
    accuracy = accuracy_score(y_val, predictions_labl_distance)
    precision = precision_score(y_val, predictions_labl_distance)
    recall = recall_score(y_val, predictions_labl_distance)
    metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
    params = {
         "model_type": 'Logistic Reg Distance',
         "scaler": "standard scaler",
         "param_grid": str(clf_dist.get_params()),
    }
    
    experiment.log_parameters(params)
    experiment.log_metrics(metrics)
    experiment.add_tag('0-1 Dist LogReg')
    experiment.log_model('0_1_dist_logreg__model', file_name)
     
 ### Q 3.2 Figures found in create_figure.py
 
### Q3.3 models
### RANDOM ###
with experiment.context_manager("Random Baseline"): 
    np.random.seed(42)
    prediction_random = np.random.uniform(low=0.0, high=1.0, size=(len(X_val)))
    prediction_labl_random = np.round(prediction_random)
     
    #compute metrics
    f1 = f1_score(y_val, prediction_labl_random)
    accuracy = accuracy_score(y_val, prediction_labl_random)
    precision = precision_score(y_val, prediction_labl_random)
    recall = recall_score(y_val, prediction_labl_random)
    metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
    params = {
         "model_type": 'Random Baseline',
         "scaler": "None",
         "param_grid": 'seed=42',
     }
    
    experiment.log_parameters(params)
    experiment.log_metrics(metrics)
    experiment.add_tag('0-2 Random Baseline')

 
 
 ### ANGLE ###
with experiment.context_manager("shot_angle_logreg"): 
    X_train_angle = pd.DataFrame(X_train['shot_angle'])
    X_val_angle = pd.DataFrame(X_val['shot_angle'])
    clf_ang = LogisticRegression()
    clf_ang.fit(X_train_angle, y_train)
    predictions_angle = clf_ang.predict_proba(X_val_angle)
    predictions_labl_angle = clf_ang.predict(X_val_angle)
    accuracy_angle = clf_ang.score(X_val_angle, y_val)
     
    file_name = "LogReg_angle_model.pkl"   
    pickle.dump(clf_ang, open(file_name, "wb"))
     
     #compute metrics
    f1 = f1_score(y_val, predictions_labl_angle)
    accuracy = accuracy_score(y_val, predictions_labl_angle)
    precision = precision_score(y_val, predictions_labl_angle)
    recall = recall_score(y_val, predictions_labl_angle)
    metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
    params = {
         "model_type": 'Logistic Reg Distance',
         "scaler": "standard scaler",
         "param_grid": str(clf_dist.get_params()),
     }
    
    experiment.log_parameters(params)
    experiment.log_metrics(metrics)
    experiment.add_tag('0-3 Angle LogReg')
    experiment.log_model('0_3_angle_logreg__model', file_name)
     
 
 ### DIST ANGLE ###
with experiment.context_manager("shot_angle_dist_logreg"): 
    clf_dist_ang = LogisticRegression()
    clf_dist_ang.fit(X_train, y_train)
    predictions_dist_angle = clf_dist_ang.predict_proba(X_val)
    predictions_labl_dist_angle = clf_dist_ang.predict(X_val)
    accuracy_dist_angle = clf_dist_ang.score(X_val, y_val)
     
    file_name = "LogReg_dist_angle_model.pkl"   
    pickle.dump(clf_dist_ang, open(file_name, "wb"))
       
     #compute metrics
    f1 = f1_score(y_val, predictions_labl_angle)
    accuracy = accuracy_score(y_val, predictions_labl_angle)
    precision = precision_score(y_val, predictions_labl_angle)
    recall = recall_score(y_val, predictions_labl_angle)
    metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
    params = {
         "model_type": 'Logistic Reg Distance',
         "scaler": "standard scaler",
         "param_grid": str(clf_dist.get_params()),
     }
    
    experiment.log_parameters(params)
    experiment.log_metrics(metrics)
    experiment.add_tag('0-4 Angle Dist LogReg')
    experiment.log_model('0_4_angle_dist_logreg__model', file_name)
 
data_graph = [predictions_distance[:,1,None],predictions_angle[:,1,None], predictions_dist_angle[:,1,None],prediction_random[:,None]]

 
model_names = ['LogReg Distance', 'LogReg Angle', 'LogReg Angle + Distance', 'Random Baseline']
fig_number = '0'
 
#creating figures and saving them locally + on comet
create_figure.fig_roc_auc(y_val, data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.calibration_fig(y_val, data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.fig_cumulative_goal(y_val, data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.fig_goal_rate(y_val, data_graph,fig_number,model_names=model_names,experiment=experiment)