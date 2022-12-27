import pandas as pd
import numpy as np

"""
This file contains all feature engineering functions (i.e. features that are not directly extracted from
the JSON files.
"""


def add_offensive_side_feature(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
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
    coordinates = df
    coordinates = coordinates.groupby(['game_id', 'team', 'period'])['x_coordinate'].mean().reset_index()
    coordinates['offensive_side'] = np.sign(coordinates['x_coordinate'])
    coordinates = coordinates.drop(['x_coordinate'], axis=1)
    result = pd.merge(df, coordinates, on=['game_id', 'team', 'period'], how='left')
    
    return result


def shot_distance_compute_feature(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Computes the distance between a shooter and the adversary goal net.
    :param df: A complete tidy data-frame
    :param inplace: Boolean determining whether the feature is added in place
    :return: A dataframe with the aforementioned column
    """

    def compute_net_distance(x, y, offensive_side):
        """
        Helper function. Computes and returns the distance between the xy shooter coordinates
        and the net they are scoring into based on "home_offensive_side".
        :param x: shooter x-coordinate
        :param y: shooter y-coordinate
        :param offensive_side: side of the rink the team is offensive toward in that period.
        :return: distance between the shooter and the net he shoots towards.
        """
        goal_coord = np.array([89, 0])
        if x is None or y is None:
            return None
        goal_coord = offensive_side * goal_coord
        return np.linalg.norm(np.array([x, y]) - goal_coord)

    if not inplace:
        df = df.copy()
    df['shot_distance'] = df.apply(lambda row: compute_net_distance(
        row['x_coordinate'],
        row['y_coordinate'],
        row['offensive_side']), axis=1)
    return df


def add_shot_angle(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Computes the angle relative to the middle frontal line of the goal..
    :param df: A complete tidy data-frame
    :param inplace: Boolean determining whether the feature is added in place
    :return: A dataframe with the aforementioned column
    """

    def compute_shot_angle(x: float, y: float, offensive_side: int) -> float:
        """
        Helper function. Computes and returns the angle between the xy shooter coordinates
        and the net they are scoring into based on "home_offensive_side" relative to the center line of the net.
        :param x: shooter x-coordinate
        :param y: shooter y-coordinate
        :param offensive_side: side of the rink the team is offensive toward in that period.
        :return: angle between the front of the goal and the shot
        """
        goal_coord = np.array([89, 0])
        if x is None or y is None:
            return 0
        goal_coord = offensive_side * goal_coord

        relative_x = x - goal_coord[0]  # bring x-coordinate relative to the goal
        angle = 0  # Defaults to 0 if x = [-89 or 89]. That's actually common.
        y += 1e-5  # avoid division by zero
        if np.sign(goal_coord[0]) == -1:  # left goal
            if (np.sign(relative_x)) == 1:  # front of the goal
                angle = np.arctan(np.abs(y) / relative_x)
            elif (np.sign(relative_x)) == -1:  # behind the goal
                angle = np.arctan(np.abs(relative_x) / y) + np.pi/2  # +90 degrees to account its from behind
        elif np.sign(goal_coord[0]) == 1:  # right goal
            if (np.sign(relative_x)) == -1:  # front of the goal
                angle = np.arctan(np.abs(y) / np.abs(relative_x))
            elif (np.sign(relative_x)) == 1:  # behind the goal
                angle = np.arctan(relative_x / y) + np.pi/2  # +90 degrees to account its from behind
        return np.rad2deg(angle)

    if not inplace:
        df = df.copy()
    df['shot_angle'] = df.apply(lambda row: compute_shot_angle(
        row['x_coordinate'],
        row['y_coordinate'],
        row['offensive_side']), axis=1)
    return df


def change_in_shot_angle(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Function to create the column determining the absolute value of the change in angle relative to the center line
    between two consecutive shots.
    :param df: A complete tidy data-frame
    :param inplace: Boolean determining whether the feature is added in place
    :return: A dataframe with the aforementioned column
    """
    if not inplace:
        df = df.copy()
    df['change_in_angle'] = 0
    for index, row in df.loc[(df['is_rebound']) & (df['prev_event_type'] == 'Shot')].iterrows():
        prev_angle = df.iloc[index - 1]['shot_angle']
        df.loc[index, 'change_in_angle'] = np.abs(row['shot_angle'] - prev_angle)
    return df


def feature_engineer(df_game_tidied):
    """
    The main function of the task feature engineering
    :param df_game_tidied: The tidied version of nhl data, stored in a dataframe
    :return: A new dataframe added all neccesary features.
    """
    df_game = df_game_tidied.copy()
    
    print("Start feature engineering for the tidied dataframe...")
    
    print("Start correcting incorrect coordinates and adding shot distance...")
    #This function must always be called at first of all, because it help correcting wrong coordinates
    df_game = add_shot_distance_and_correct_coordinates(df_game)

    print("Start adding shot angle...")
    df_game = add_shot_angle(df_game)

    print("Start adding change in shot angle...")
    df_game = add_change_in_shot_angle(df_game)

    print("Start adding distance from last event and the shot speed...")
    df_game = add_distance_from_last_event_and_speed(df_game)

    print("Finish feature engineering!")
    return df_game
