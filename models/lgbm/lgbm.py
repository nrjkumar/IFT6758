from comet_ml import Experiment
import optuna, sys, os
import lightgbm as lgbm
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import StratifiedKFold
from pprint import pprint
import pickle

# import figure plot
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from create_plot import *
from utils import *
import os
from dotenv import load_dotenv

load_dotenv()

feature_list = ['shot_type', 'strength', 'is_playoff', 'prev_event_type', 'time_since_prev_event', 'is_rebound',
                'distance_to_prev_event', 'speed_since_prev_event', 'is_penalty_shot', 'shot_distance',
                'shot_angle', 'change_in_angle', 'time_since_pp', 'relative_strength']


def objective(trial, X, y):
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 27000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
        "num_leaves": trial.suggest_int("num_leaves", 8, 27000, step=20),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 11, step=2),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 27000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1, step=0.1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1, step=0.1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    cv_scores = np.empty(5)
    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgbm.LGBMClassifier(objective="binary", **param_grid, verbose=-1)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            early_stopping_rounds=25,
            callbacks=[LightGBMPruningCallback(trial, "binary_logloss")],
        )
        preds = model.predict_proba(X_val)
        cv_scores[idx] = log_loss(y_val, preds)
    return np.mean(cv_scores)


def main():
    '''
    optuna search for best lgbm hyper params
    '''

    X, y, X_test, y_test = load_data(
        features=feature_list,
        train_val_seasons=DEFAULT_TRAIN_SEASONS,
        test_season=DEFAULT_TEST_SEASONS,
        do_split_val=False,
        target='is_goal',
        use_standard_scaler=True,
        drop_all_na=False,
        convert_bool_to_int=True,
        one_hot_encode_categoricals=True
    )

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                direction="minimize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=150)
    print(f"\tBest params:")

    pprint(study.best_params)
    params = study.best_params

    # Train Model with optimal params
    experiment = Experiment(
        api_key=os.getenv('COMET_API_KEY'),
        project_name="milestone2",
        workspace="ift6758-17"
    )
    model = lgbm.LGBMClassifier(**params)
    X = X.drop(columns=X.columns.difference(X_test.columns))
    X_test = X_test.drop(columns=X_test.columns.difference(X.columns))
    model.fit(X, y)

    y_pred = model.predict_proba(X_test)[:, 1]

    y_pred_vec = [y_pred]
    model_names = ['Tuned LGBM']
    fig_number = '6-LGBM'
    plot_roc_auc(y_test, y_pred_vec, fig_number, model_names, experiment)
    plot_cumulative_goal(y_test, y_pred_vec, fig_number, model_names, experiment)
    plot_goal_rate(y_test, y_pred_vec, fig_number, model_names, experiment)
    calibration_plot_fig(y_test, y_pred_vec, fig_number, model_names, experiment)

    # save xgb_model
    file_name = "6-LGBM.pkl"

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
        "model_type": 'LGBM',
        "scaler": "standard scaler",
        "param_grid": str(model.get_params()),
    }
    print(metrics)

    experiment.log_parameters(params)
    experiment.log_metrics(metrics)
    experiment.add_tag('6-LGBM')
    experiment.log_model('6-LGBM', '6-LGBM.pkl')


if __name__ == "__main__":
    main()
