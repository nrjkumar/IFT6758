from comet_ml import Experiment
import optuna
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,log_loss
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from pprint import pprint
import pickle, os, sys

# import figure plot


# utils_path = os.path.abspath(os.path.join('..'))
# sys.path.append(utils_path)
# from utils import *
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from create_plot import *
from utils import *

from dotenv import load_dotenv

load_dotenv()

feature = ['period', 'x_coordinate', 'y_coordinate',
           'game_time(s)', 'prev_event_x', 'prev_event_y',
           'time_since_prev_event', 'is_rebound', 'distance_to_prev_event',
           'speed_since_prev_event', 'shot_distance', 'shot_angle',
           'change_in_angle', 'shot_type', 'prev_event_type','time_since_pp',
           'home_strength','away_strength']


def objective(trial):
    X, y, _, _ = load_data(
        features=feature,
        train_val_seasons=DEFAULT_TRAIN_SEASONS,
        test_season=DEFAULT_TEST_SEASONS,
        do_split_val=False,
        target='is_goal',
        use_standard_scaler=True,
        drop_all_na=True,
        convert_bool_to_int=True,
        one_hot_encode_categoricals=True
    )
    #X.fillna(0)


    param_grid = {
        "silent": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 25),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000)
    }

    if param_grid["booster"] == "gbtree" or param_grid["booster"] == "dart":
        param_grid["max_depth"] = trial.suggest_int("max_depth", 1, 12)
        param_grid["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param_grid["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param_grid["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param_grid["booster"] == "dart":
        param_grid["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param_grid["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param_grid["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param_grid["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = np.empty(5)
    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        train = xgb.DMatrix(X.iloc[train_idx], label=y[train_idx])
        val = xgb.DMatrix(X.iloc[val_idx], label=y[val_idx])

        # Add a callback for pruning.
        clf = xgb.train(param_grid, train, evals=[(val, "validation")],
                        callbacks=[XGBoostPruningCallback(trial, "validation-auc")])
        y_preds = clf.predict(val)
        score = roc_auc_score(y[val_idx], y_preds)
        cv_scores[idx] = score
    return np.mean(cv_scores)


def main():
    '''
    grid search for the best model
    save figures for the best parameter
    save model, hyperparameter
    '''
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                direction="maximize", study_name="XGB Classifier")
    func = lambda trial: objective(trial)
    study.optimize(func, n_trials=75)

    # Save study plots
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    fig.write_html('5-2_optuna_optim_history.html')
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    fig.write_html('5-2_hyperparam_importance.html')
    print(f"\tBest params:")

    pprint(study.best_params)
    params = study.best_params
    params= {'alpha': 1.8681414898850136e-05,
     'booster': 'dart',
     'eta': 0.2531499873091687,
     'gamma': 1.0644957574704856e-06,
     'grow_policy': 'depthwise',
     'lambda': 1.744968010669035e-05,
     'max_depth': 8,
     'n_estimators': 709,
     'normalize_type': 'forest',
     'rate_drop': 0.047822250564598025,
     'sample_type': 'uniform',
     'scale_pos_weight': 4,
     'skip_drop': 0.00026092912897969413}

    X_train, y_train, X_test, y_test = load_data(
        features=feature,
        train_val_seasons=DEFAULT_TRAIN_SEASONS,
        test_season=DEFAULT_TEST_SEASONS,
        do_split_val=False,
        target='is_goal',
        use_standard_scaler=True,
        drop_all_na=False,
        convert_bool_to_int=True,
        one_hot_encode_categoricals=True
    )
    # Train Model with optimal params
    experiment = Experiment(
        api_key=os.getenv('COMET_API_KEY'),
        project_name="milestone2",
        workspace="ift6758-17",
    )
    model = XGBClassifier(**params)
    X_train = X_train.drop(columns=X_train.columns.difference(X_test.columns))
    X_test = X_test.drop(columns=X_test.columns.difference(X_train.columns))
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]

    y_pred_vec = [y_pred]
    model_names=['Tuned XGB']
    fig_number = '5-2'
    plot_roc_auc(y_test, y_pred_vec, fig_number, model_names,experiment)
    plot_cumulative_goal(y_test, y_pred_vec, fig_number, model_names,experiment)
    plot_goal_rate(y_test, y_pred_vec, fig_number, model_names,experiment)
    calibration_plot_fig(y_test, y_pred_vec, fig_number, model_names,experiment)

    # save xgb_model
    file_name = "tuned_xgb_model.pkl"

    # save
    file_name = "tuned_xgb_model.pkl"

    # save
    pickle.dump(model, open(file_name, "wb"))

    # Compute metrics
    y_pred_labels = model.predict(X_test)
    f1 = f1_score(y_test, y_pred_labels)
    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)
    metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
    params = {
        "model_type": 'XGB',
        "scaler": "standard scaler",
        "param_grid": str(model.get_params()),
    }

    experiment.log_parameters(params)
    experiment.log_metrics(metrics)
    experiment.add_tag('5-2 XGB')
    experiment.log_model('5_2 tuned model', 'tuned_xgb_model.pkl')


if __name__ == "__main__":
    main()
