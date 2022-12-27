from comet_ml import API
from comet_ml import Experiment

import os, sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import StackingClassifier

import pickle

import pandas as pd
import numpy as np

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
#print(parentdir)
sys.path.append(parentdir)

import utils
import create_plot
COMET_API_KEY = 'HCGg9ssMTnTjvXmcVwiM5MyfO'

experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="milestone2",
        workspace="ift6758-17",
    )



api = API(api_key='HCGg9ssMTnTjvXmcVwiM5MyfO' )

# api.download_registry_model('axelbogos', '6-lgbm', version='1.0.0')
# api.download_registry_model('axelbogos', '6-2-nn-tuned-model', version='1.0.0')
# api.download_registry_model('axelbogos', '5-2-grid-search-model', version='2.0.0')
# api.download_registry_model('axelbogos', '3-3-angle-dist-logreg-model', version='1.0.0')

lgbm = pickle.load(open(parentdir +'/Stacked/6-LGBM.pkl','rb'))
nn = pickle.load(open(parentdir +'/Stacked/tuned_nn_model.pkl','rb'))
#xgb = pickle.load(open(parentdir +'/Stacked/tuned_xgb_model.pkl','rb'))
logreg = pickle.load(open(parentdir +'/Stacked/LogReg_dist_angle_model.pkl','rb'))

model_registry = ['6-LGBM','tuned_nn_model','tuned_xgb_model','LogReg_dist_angle_model']

feature_list = ['period', 'x_coordinate', 'y_coordinate',
           'game_time(s)', 'prev_event_x', 'prev_event_y',
           'time_since_prev_event', 'is_rebound', 'distance_to_prev_event',
           'speed_since_prev_event', 'shot_distance', 'shot_angle',
           'change_in_angle', 'shot_type', 'prev_event_type','time_since_pp',
           'home_strength','away_strength', 'strength', 'relative_strength']

X_train, y_train, X_val, y_val, _, _ = utils.load_data(features = feature_list,drop_all_na=True, one_hot_encode_categoricals = True )

estimators = [('lgbm',lgbm),('nn',nn),('logreg',logreg)]

stack= StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

stack.fit(X_train, y_train)

predictions = stack.predict_proba(X_val)

model_names=['Stack trained']
fig_number = '6-4'

create_plot.plot_roc_auc(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)
create_plot.calibration_plot_fig(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)
create_plot.plot_cumulative_goal(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)
create_plot.plot_goal_rate(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)

file_name = "tuned_stacked_trained_model.pkl"

# save
pickle.dump(stack, open(file_name, "wb"))

# Compute metrics
y_pred_labels = stack.predict(X_val)
f1 = f1_score(y_val, y_pred_labels)
accuracy = accuracy_score(y_val, y_pred_labels)
precision = precision_score(y_val, y_pred_labels)
recall = recall_score(y_val, y_pred_labels)
metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
params = {
    "model_type": 'stacked classifier',
    "Classifiers": model_registry,
    "scaler": "standard scaler",
    "param_grid": str(stack.get_params()),
}

experiment.log_parameters(params)
experiment.log_metrics(metrics)
experiment.add_tag('6-4_Stacked_trained')
experiment.log_model('6_4_stacked_trained_tuned_model', 'tuned_stacked_trained_model.pkl')