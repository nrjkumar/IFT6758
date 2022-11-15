import os.path
import seaborn  as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TIDY_DATA_DIR = os.path.expanduser('~')+'/nhlapidata/csv/tidy_data.csv'
SAVE_PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')

class IceHockey_plot:
    def __init__(self, fig_size=(15, 10)):
        self.fig_size = fig_size

    def Goalrate_distance_and_angle(self, df:pd.DataFrame, save_figure= True):

        """Plots a comparison graph across seasons for relationship between shot distance and goals"""
        
        fig = plt.figure(figsize=(25, 20))

        plt.subplot(221)
        filtered_df = df[df['shot_distance'].notnull()]
        filtered_df['shot_distance'] = filtered_df['shot_distance'].round(0)
