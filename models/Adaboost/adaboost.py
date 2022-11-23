from comet_ml import Experiment

import sys
import os.path
from dotenv import load_dotenv
import optuna
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import utils as utils
import create_plot

load_dotenv()
COMET_API_KEY = os.getenv('COMET_API_KEY')
experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="milestone2",
        workspace="ift6758-17",
    )


DEFAULT_TRAIN_SEASONS = ['201520216', '20162017', '20172018', '20182019' ]
DEFAULT_TEST_SEASONS = ['20192020']

feature_list =['shot_type', 'strength','is_playoff', 'prev_event_type', 'time_since_prev_event', 'is_rebound', 'distance_to_prev_event','speed_since_prev_event', 'is_penalty_shot','shot_distance', 
 'shot_angle', 'change_in_angle', 'time_since_pp','relative_strength']

X_train, y_train, X_val, y_val, _, _ = utils.load_data(features = feature_list )

model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=182, random_state=0)
model.fit(X_train, y_train)
predictions = model.predict_proba(X_val)

# auct=fig_roc_auc(y_val,preditions[:,1,None].T,1,"Neural Network")
# calibration_fig(y_val,preditions[:,1,None].T,1,"Neural Network")
# fig_cumulative_goal(y_val,preditions[:,1,None].T,1,"Neural Network")
# fig_goal_rate(y_val,preditions[:,1,None].T,1,"Neural Network")
model_names=['AdaBoost']
fig_number = '6-3'

create_plot.plot_roc_auc(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)
create_plot.calibration_plot_fig(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)
create_plot.plot_cumulative_goal(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)
create_plot.plot_goal_rate(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)

file_name = "tuned_adaboost_model.pkl"

# save
pickle.dump(model, open(file_name, "wb"))

# Compute metrics
y_pred_labels = model.predict(X_val)
f1 = f1_score(y_val, y_pred_labels)
accuracy = accuracy_score(y_val, y_pred_labels)
precision = precision_score(y_val, y_pred_labels)
recall = recall_score(y_val, y_pred_labels)
metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
params = {
    "model_type": 'Neural Net',
    "scaler": "standard scaler",
    "param_grid": str(model.get_params()),
}

experiment.log_parameters(params)
experiment.log_metrics(metrics)
experiment.add_tag('6-3 AdaBoost')
experiment.log_model('6_3_AdaBoost_tuned_model', 'tuned_adaboost_model.pkl')
