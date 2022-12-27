import numpy as np
import pandas as pd

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt



def normalize_df(df):
    df_normalized = df.copy()
    df_normalized
    df['xCoordNorm'] = np.where(df['rinkSide'] == 'left', df['xCoord'], -1 * df['xCoord'])
    df_normalized['yCoordNorm'] = np.where(df['rinkSide'] == 'left', df['yCoord'], -1 * df['yCoord'])

    return df_normalized

# Normalize Shot Direction
df_normalized = normalize_df(df)


# Grouped list of game IDs by season and game type
games = df.groupby(['season', 'gameType'])['gameID'].unique()



def plot_one_event(event, normalize=False, title='', marker='o', size=100, color='#ea6969', alpha=.7):
    fig, ax = plt.subplots(figsize=(13,8.5))
    plt.axis('on')

    img = plt.imread('../figures/nhl_rink.png')
    # rink dimensions
    ext = [-100, 100, -42.5, 42.5]
    plt.imshow(img, zorder=0, extent=ext, aspect=1)

    # plot the points on top of the image
    xLabel, yLabel = ('xCoordNorm', 'yCoordNorm') if normalize else ('xCoord', 'yCoord')
    plt.scatter(event[xLabel], event[yLabel], s=size, c=color, alpha=alpha, marker=marker, label='shot')

    # plot the opposing team's net
    net_x, net_y = (89, 0) if (event['rinkSide'] == 'left') or normalize else (-89, 0)
    plt.scatter(net_x, net_y, s=200, c='blue', alpha=alpha, marker='x', label='net')
    plt.xlabel('feet')
    plt.ylabel('feet')
    plt.title(title)
    plt.legend()


@interact(
    season_year=widgets.IntSlider(min=2016, max=2020, step=1),
    season_type=['R', 'P']
)
def show_data(season_year, season_type):
    season_games = games[(season_year, season_type)]

    @interact(game_id=widgets.SelectionSlider(options=season_games))
    def show_game(game_id):
        selected_game = df_normalized[df_normalized['gameID'] == game_id]

        @interact(event_id=widgets.IntSlider(min=0, max=selected_game.shape[0]-1, step=1, value=0))
        def show_event(event_id, normalize_shots=True):
            event = selected_game.iloc[event_id]
            title = f"{event['eventType']} \t {event['shotType']} \t {event['shooter']} -> {event['goalie']}"
            plot_one_event(event, normalize=normalize_shots, title=title, marker='o', size=300, color='red', alpha=1)
            print(event)

