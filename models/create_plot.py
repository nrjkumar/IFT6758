import os

from comet_ml import Experiment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_curve, auc

FIGURE_PATH = os.path.join(os.path.dirname(__file__), '..', 'figures', 'milestone2')


def plot_roc_auc(y_val, y_pred_vec, fig_number,model_names,experiment=None) -> None:
    fig = plt.figure(figsize=(10, 10))
    for idx, y_pred in enumerate(y_pred_vec):
        # Get FPR, TPR and AUC
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC ({model_names[idx]}): area = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{fig_number} ROC Curves")
    plt.legend(loc="lower right")
    fig_name = f'{fig_number}_roc_auc.png'
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    if experiment != None:
        experiment.log_figure(figure=fig)
    plt.show()
    fig.savefig(os.path.join(FIGURE_PATH, fig_name))
    plt.close()
    return None

def cumulative_goal_data(y_test, y_pred):
    '''
    y_test: testing label
    y_pred: probability of estimate x_test
    use ecdfplot in seaborn for estimate y_pred probability
    '''
    y_pred_percentile = 100 * stats.rankdata(y_pred, "min") / len(y_pred)
    test_est = np.array([np.round(y_pred_percentile), y_test]).T
    df_test_est = pd.DataFrame(test_est, columns=['model_per', 'is_goal'])

    df_fil = df_test_est[df_test_est['is_goal'] == 1]
    return df_fil

def plot_cumulative_goal(y_val, y_pred_vec, fig_number,model_names,experiment=None) -> None:
    '''
    y_val: testing label
    y_pred: probability of estimate x_test
    use ecdfplot in seaborn for estimate y_pred probability
    '''
    fig = plt.figure(figsize=(10, 10))
    for idx, y_pred in enumerate(y_pred_vec):
        cumulative_data = cumulative_goal_data(y_val, y_pred)
        ax = sns.ecdfplot(data=cumulative_data, x=100 - cumulative_data.model_per, label=model_names[idx])

    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.xticks(np.arange(0, 100 * 1.01, 10))
    xvals = ax.get_xticks()
    ax.set_xticklabels(100 - xvals.astype(np.int32))
    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in yvals])
    ax.set(xlabel='Shot probability model percentile')
    ax.set_title(f"{fig_number} Cumulative % of Goals")
    plt.legend(loc='lower right')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    if experiment != None:
        experiment.log_figure(figure=fig)
    plt.show()
    fig_name = f'{fig_number}_cumulative_goals.png'
    fig.savefig(os.path.join(FIGURE_PATH, fig_name))
    # experiment.log_figure(figure_name='cumulative_goal', figure=plt)
    plt.close()

def plot_goal_rate(y_val, y_pred_vec, fig_number,model_names,experiment=None) -> None:
    '''
    create goal rate figure
    y_val: testing label
    y_pred: probability of estimate x_test
    count number of goal, goal+shot
    change xlable, ylabel of the figure
    '''

    fig = plt.figure(figsize=(10, 10))
    for idx, y_pred in enumerate(y_pred_vec):
        # plot GOAL RATE

        if y_val.shape != y_pred.shape:
            y_pred = y_pred.reshape(-1)
        test_est = np.array([np.array(np.round(y_pred * 100)), y_val]).T

        df_test_est = pd.DataFrame(test_est)

        g = df_test_est.groupby(0)

        # count goals.
        feature_mat = np.array(g.sum())

        # count total of shots + goals
        group_count = np.array(g[[0]].count())

        goal_percentate = feature_mat / group_count  # goal / (goal + shot)
        model_percentage = list(g.groups.keys())

        # convert model_percentage to percentile
        model_percentile = 100 * stats.rankdata(model_percentage, "min") / len(model_percentage)

        goal_rate = np.array([goal_percentate[:, 0], model_percentile])

        df_test_est = pd.DataFrame(goal_rate[:, ::-1].T, columns=['goal_per', 'model_per'])

        xval = 100 - df_test_est.model_per

        ax = sns.lineplot(x=xval, y=df_test_est.goal_per, label=model_names[idx])

    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.xticks(np.arange(0, 100 * 1.01, 10))

    ax.set(xlabel='Shot probability model percentile', ylabel="Goals/(Shots + Goals)")

    ax.set(ylim=(0.05, 1.05))
    ax.set(xlim=(0, 110))

    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in yvals])
    xvals = ax.get_xticks()

    ax.set_xticklabels(100 - xvals.astype(np.int32))
    ax.set_title(f"{fig_number} Goal Rate")
    plt.legend(loc="upper right")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    fig_name = f'{fig_number}_goal_rate.png'
    fig.savefig(os.path.join(FIGURE_PATH, fig_name))
    if experiment != None:
        experiment.log_figure(figure=fig)
    plt.show()
    plt.close()
    return None

def calibration_plot_fig(y_val, y_pred_vec, fig_number,model_names,experiment=None) -> None:
    '''
    create calibration curve figures from y_val, y_pred database
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    for idx, y_pred in enumerate(y_pred_vec):
        disp = CalibrationDisplay.from_predictions(y_val, y_pred, n_bins=10, label=model_names[idx], ax=ax)

    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    fig_name = f'{fig_number}_calibration.png'
    plt.savefig(os.path.join(FIGURE_PATH, fig_name))
    plt.legend(loc="center right")
    if experiment != None:
        experiment.log_figure(figure=fig)
    plt.show()
    plt.close()
    return None