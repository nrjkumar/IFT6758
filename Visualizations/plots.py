import os.path, sys
import seaborn  as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TIDY_DATA_DIR = os.path.expanduser('~')+'/nhlapidata/csv/tidy_data.csv'
#SAVE_PLOT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

SAVE_PLOT_DIR = parentdir + '/figures/'
#print(TIDY_DATA_DIR)
print(SAVE_PLOT_DIR)

class IceHockey_plot:
    def __init__(self, fig_size=(15, 10)):
        self.fig_size = fig_size

    def milestone2_goal_rate_vs_distance_and_angle(self, df: pd.DataFrame, save_fig=True) -> plt.Figure:
        """
        Plots a comparative graph across seasons (2017 - 2020) of the relationship between
        shot distance and goals (as described in Part 5 Q2)
        :param df: tidy pandas.DataFrame
        :param save_fig: boolean to save the plot to SAVE_PLOT_DIR
        :return: a plt.Figure object instance
        """

        fig = plt.figure(figsize=(25, 20))

        plt.subplot(221)
        filtered_df = df[df['shot_distance'].notnull()]
        filtered_df['shot_distance'] = filtered_df['shot_distance'].round(0)
        filtered_df = filtered_df.groupby(["shot_distance"])["is_goal"].mean().to_frame().reset_index()
        ax = sns.lineplot(x='shot_distance', y='is_goal', data=filtered_df)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Goal Probability')
        plt.title('Goal Probability vs Distance')

        plt.subplot(222)
        ax = sns.histplot(data=df, x='shot_distance', hue='is_goal', multiple='stack')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Count')
        plt.title('Histogram of shot distance')

        plt.subplot(223)
        filtered_df = df[df['shot_angle'].notnull()]
        filtered_df['shot_angle'] = filtered_df['shot_angle'].round(0)
        filtered_df = filtered_df.groupby(["shot_angle"])["is_goal"].mean().to_frame().reset_index()
        ax = sns.lineplot(x='shot_angle', y='is_goal', data=filtered_df)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.xlabel('Shot Angle (deg rel. to center-line)')
        plt.ylabel('Goal Probability')
        plt.title('Goal Probability vs Angle')

        plt.subplot(224)
        ax = sns.histplot(data=df, x='shot_angle', hue='is_goal', multiple='stack')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 190, 10), rotation=20)
        plt.xlabel('Shot Angle (deg rel. to center-line)')
        plt.ylabel('Goal Probability')
        plt.title('Histogram of shot angle')

        plt.show()
        if save_fig:
            fig.savefig(os.path.join(SAVE_PLOT_DIR, 'milestone2', "Q2-2-goal_rate.png"))
        return fig

    def milestone2_goal_dist_by_empty_net(self, df: pd.DataFrame, save_fig=True) -> plt.Figure:

        fig = plt.figure(figsize=(20, 25))
        filtered_df = df[df['is_goal'] == True]
        non_empty_net = filtered_df[filtered_df['is_empty_net'] == False]
        non_empty_net_zoomed = non_empty_net[non_empty_net['shot_distance'] >= 100]
        empty_net = filtered_df[filtered_df['is_empty_net'] == True]

        plt.subplot(311)
        ax = sns.histplot(data=non_empty_net, x='shot_distance')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Count')
        plt.title('Histogram of shot distance for goals (non-empty net)')
        left, bottom, width, height = (100, -50, 100, 500)
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False,
                                  color="red",
                                  linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(130, 500, 'Zone of interest shown below', fontsize=12)

        plt.subplot(312)
        ax = sns.histplot(data=non_empty_net_zoomed, x='shot_distance')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(100, 210, 10), rotation=20)
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Count')
        plt.title(
            'Histogram of shot distance for goals (non-empty net) \n Zoom on the 100ft to 200ft shot distance zone')
        plt.text(130, 16.5, f'Total number of instance for all seasons: {non_empty_net_zoomed.shape[0]}', fontsize=14)

        plt.subplot(313)
        ax = sns.histplot(data=empty_net, x='shot_distance')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Count')
        plt.title('Histogram of shot distance for goals (empty net)')

        plt.show()
        if save_fig:
            fig.savefig(os.path.join(SAVE_PLOT_DIR, 'milestone2', "Q2-3-empty_net_goal_dist.png"))
        return fig

    
    def milestone2_shot_dist_by_is_goal(self, df: pd.DataFrame, save_fig=True):
        fig = plt.figure(figsize=(10, 10))

        ax = sns.histplot(data=df, x="shot_distance", hue="is_goal", multiple ="stack", bins=20)
        #ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xlabel('Shot distance (ft)')
        plt.title('Histogram of shot distance for shots (goals and no-goals separated)')
        ax.legend(['Goal','No Goal'], title = "Shot Result")
        
        plt.show()
        if save_fig:
            fig.savefig(os.path.join(SAVE_PLOT_DIR, 'milestone2', "Q2-1-shot_dist_by_is_goal.png"))
        
        return fig
    
    def milestone2_shot_angle_by_is_goal(self, df: pd.DataFrame, save_fig=True):
        fig = plt.figure(figsize=(10, 10))

        ax = sns.histplot(data=df, x="shot_distance", hue="is_goal", multiple ="stack", bins=20)
        #ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xlabel('Shot distance (ft)')
        plt.title('Histogram of shot distance for shots (goals and no-goals separated)')
        ax.legend(['Goal','No Goal'], title = "Shot Result")
        
        plt.show()
        if save_fig:
            fig.savefig(os.path.join(SAVE_PLOT_DIR, 'milestone2', "Q2-1-shot_angle_by_is_goal.png"))        
        return fig
    
    def milestone2_2Dhist_shot_angle_distance(self, df: pd.DataFrame, save_fig=True):
        
        fig = plt.figure(figsize=(20, 20))

        ax = sns.jointplot(data=df, x="shot_distance", y="shot_angle", kind="hist", ratio=4)
        ax.fig.suptitle('2D Histogram of shots by angle and distance from net')
        ax.fig.tight_layout()
        ax.ax_joint.set_xlabel("Shot Distance (feet)")
        ax.ax_joint.set_ylabel("Shot Angle (Degree)")
               
        plt.show()
        if save_fig:
            ax.savefig(os.path.join(SAVE_PLOT_DIR, 'milestone2', "Q2-1-2Dhist_shot_angle_distance.png"))  
        return fig    

def main():
    df = pd.read_csv(TIDY_DATA_DIR)

    hockey_plotter = IceHockey_plot()

    # ---- Milestone 1 Plots ----
    # # Plot Q5.1
    # hockey_plotter.milestone1_shot_type_histogram(df)
    # # Plot Q5.2
    # hockey_plotter.milestone1_distance_vs_goal_chance(df)
    # # Plot Q5.3
    # hockey_plotter.milestone1_distance_and_type_vs_goal(df)

    # ---- Milestone 2 Plots ----

    # Plot Q2.1
    hockey_plotter.milestone2_shot_dist_by_is_goal(df)
    hockey_plotter.milestone2_shot_angle_by_is_goal(df)
    hockey_plotter.milestone2_2Dhist_shot_angle_distance(df)
    
    # Plot Q2.2
    hockey_plotter.milestone2_goal_rate_vs_distance_and_angle(df)

    # Plot Q2.3
    hockey_plotter.milestone2_goal_dist_by_empty_net(df)

if __name__ == "__main__":
    main()
