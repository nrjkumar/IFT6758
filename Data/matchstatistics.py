import pandas as pd
import numpy as np

"""
This file contains functions related to statistics of gameplays and teams 
"""


def add_home_offensive_side_feature(df: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Function to create the column determining on which side of the rink the home team is currently attacking.
    -1 if the net the home team scores into is in the negative x-coordinates, +1 if they score in the net
    in the positive x-coordinates.
    :param df: A complete tidy data-frame
    :param inplace: Boolean determining whether the feature is added in place
    :return: A dataframe with the aforementioned column
    """
    if not inplace:
        df = df.copy()
    if 'home_offensive_side' in df.columns:
        return df
    coordinates = df[df['team'] == df['home_team']]
    coordinates = coordinates.groupby([ 'game_id', 'home_team', 'period'])['x_coordinate'].mean().reset_index()
    coordinates['home_offensive_side'] = np.sign(coordinates['x_coordinate'])
    coordinates = coordinates.drop(['x_coordinate'], axis=1)
    return pd.merge(df, coordinates, on=['game_id', 'home_team', 'period'], how='left')


def add_shot_distance_feature(df: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Computes the distance between a shooter and the adversary goal net.
    :param df: A complete tidy data-frame
    :param inplace: Boolean determining whether the feature is added in place
    :return: A dataframe with the aforementioned column
    """

    def compute_net_distance(x, y, team, home_team, home_offensive_side):
        """
        Helper function. Computes and returns the distance between the xy shooter coordinates
        and the net they are scoring into based on "home_offensive_side".
        :param x: shooter x-coordinate
        :param y: shooter y-coordinate
        :param team: shooter's team
        :param home_team: Game home team
        :param home_offensive_side: side of the rink the home team is offensive toward in that period.
        :return: distance between the shooter and the net he shoots towards.
        """
        goal_coord = np.array([89, 0])
        if x is None or y is None:
            return None
        if team == home_team:
            goal_coord = home_offensive_side * goal_coord
        else:
            goal_coord = -1 * home_offensive_side * goal_coord
        return np.linalg.norm(np.array([x, y]) - goal_coord)

    if not inplace:
        df = df.copy()
    df['shot_distance'] = df.apply(lambda row: compute_net_distance(row['x_coordinate'],row['y_coordinate'],
        row['team'],row['home_team'],
        row['home_offensive_side']), axis=1)
    return df
