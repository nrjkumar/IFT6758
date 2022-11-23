from comet_ml import Experiment
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

# utils_path = os.path.abspath(os.path.join('..'))
# sys.path.append(utils_path)
# from utils import *
util_path = p = os.path.abspath('../..')
from create_plot import *
from utils import *
load_dotenv()
COMET_API_KEY = os.getenv('COMET_API_KEY')


def train_basic_xgb(X_train, y_train, X_val):
	model = XGBClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict_proba(X_val)[:, 1]
	y_labels = model.predict(X_val)
	return y_pred, y_labels


def main():
	'''
    create 4 figures and save and /figures with each feature
    '''

	X_train, y_train, X_val, y_val, _, _ = load_data(
		features=['shot_distance', 'shot_angle'],
		train_val_seasons=DEFAULT_TRAIN_SEASONS,
		test_season=DEFAULT_TEST_SEASONS,
		train_val_ratio=0.2,
		target='is_goal',
		use_standard_scaler=False,
		drop_all_na=True
	)
	y_pred_vec = []
	y_labels_vec=[]
	FEATURES_LIST = [['shot_distance'], ['shot_angle'], ['shot_distance', 'shot_angle']]

	# All combinations of features
	for features in FEATURES_LIST:
		sub_X_train = X_train[features]
		sub_X_val = X_val[features]
		y_pred,y_Labels = train_basic_xgb(sub_X_train, y_train, sub_X_val)
		y_pred_vec.append(y_pred)
		y_labels_vec.append(y_Labels)

	# Random Baseline
	y_pred_vec.append(np.random.uniform(0, 1, size=y_val.shape[0]))

	experiment = Experiment(
		api_key=os.getenv('COMET_API_KEY'),
		project_name="ift-6758-milestone-2",
		workspace="axelbogos",
	)

	model_names = ['Distance', 'Angle', 'Distance + Angle', 'Random']
	fig_number = '5-1'
	plot_roc_auc(y_val, y_pred_vec, fig_number, model_names)
	plot_cumulative_goal(y_val, y_pred_vec, fig_number, model_names)
	plot_goal_rate(y_val, y_pred_vec, fig_number, model_names)
	calibration_plot_fig(y_val, y_pred_vec, fig_number, model_names)

	# Compute metrics
	f1 = f1_score(y_val, y_labels_vec[2])
	accuracy = accuracy_score(y_val, y_labels_vec[2])
	precision = precision_score(y_val, y_labels_vec[2])
	recall = recall_score(y_val, y_labels_vec[2])
	metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
	params = {
		"model_type": '5-1-XGB',
		"scaler": "standard scaler",
		"param_grid": 'Default XGBClassifier() Distance+Angle',
	}

	experiment.log_parameters(params)
	experiment.log_metrics(metrics)
	experiment.add_tag('5-1 XGB')


if __name__ == "__main__":
	main()
