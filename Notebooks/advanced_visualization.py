import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import os

DATA_PATH = os.path.expanduser("~") + '/nhlapidata/csv'

def compute_avgshots_league(df: pd.DataFrame, year: int) -> np.array:
    """
    Computes the average shots per hour across the league for a season( year)

    :param df:  Tidy or cleaned dataframe with coordinates projected for Half Rink
    :param year: season for which average shots per hour are computed
    :return: array with shape 100X85
    """

    season = int(str(year) + str(year + 1))

    df_copy = df[df["season"] == season].copy()
    df_copy["coord_tuple"] = df_copy[["x_coordinate_adj", "y_coordinate_adj"]].apply(tuple, axis=1)

    league_data = np.zeros((100, 85))

    for i, j in df_copy["coord_tuple"]:
        if np.isnan(i) or np.isnan(j):
            pass
        else:
            league_data[int(i), int(j)] += 1

    # total hours in the season
    season_matches_drop = df_copy.drop_duplicates(subset=["game_id"], keep="last")
    season_hours = 0
    for i, txt in enumerate(season_matches_drop["game_time"]):
        time = txt.split(":")
        hour_match = int(time[0]) / 60.0 + int(time[1]) / 3600.0
        season_hours += max(hour_match, 1.0)

    # need to count each game twice since two team, need to replace with actual calculation of total game time
    league_data = league_data / (season_hours * 2)

    return league_data


def compute_avgshots_team(df, year: int, team: str) -> np.array:
    """
    Computes the average number of shots per hour for 1 team for 1 season.
    :param df: tidy frame with xy coordinates projected on a half-rink
    :param year: int representation of a year. ex: 2016
    :return: np.array with shape 100 X 85 (i.e. the XY half rink plane)
    """
    season = int(str(year) + str(year + 1))

    # use date to keep the same match with different date
    df_copy = df[df["season"] == season].copy()
    df_copy2 = df_copy[df_copy["team"] == team].copy()
    df_copy2["coord_tuple"] = df_copy2[["x_coordinate_adj", "y_coordinate_adj"]].apply(tuple, axis=1)

    data_team = np.zeros((100, 85))

    for i, j in df_copy2["coord_tuple"]:
        if np.isnan(i) or np.isnan(j):
            pass
        else:
            data_team[int(i), int(j)] += 1

    # count team hours
    # count match as home & away in the season, drop duplicate for detail match
    team_matches_df = df_copy.loc[
        (df_copy["home_team"] == team) | (df_copy["away_team"] == team)
        ]
    team_matches_drop = team_matches_df.drop_duplicates(subset=["game_id"], keep="last")
    team_hours = 0
    for i, txt in enumerate(team_matches_drop["game_time"]):
        time = txt.split(":")
        hour_match = int(time[0]) / 60.0 + int(time[1]) / 3600.0
        team_hours += max(hour_match, 1.0)

    data_team = data_team / team_hours

    return data_team


def all_season_all_team_avg(df: pd.DataFrame, start_year: int = 2016, end_year: int = 2020, sigma: int = 4,
                        threshold: float = 0.001) -> dict:
    """
    Computes the average number of shots per hour across the league for all season in-between start_year & end_year.
    :param df: tidy frame with xy coordinates projected on a half-rink
    :param start_year: int representation of the year of the first season of interest. ex: 2016
    :param end_year: int representation of the year of the last season of interest. ex: 2020
    :param sigma: Gaussian kernel hyper-parameter. Recommended range: [2,4]
    :param threshold: All gaussian differences within float threshold of 0 are ignored and replaced by None
    :return: dict of years of dict of teams
    """
    teams_per_season = {}
    shot_freq_team_season = {}

    # Build dict of all uniques teams per season
    for year in range(start_year, end_year + 1):
        # Index season as 20162017 for example
        season = int(str(year) + str(year + 1))
        # Get all unique teams for a season
        teams_per_season[str(year)] = np.array(df[df['season'] == season]['home_team'].unique())

    for year in range(start_year, end_year + 1):
        # Create a dict object for each year. Each such dict object is a dict of teams containing their individual xy
        # shot frequencies
        shot_freq_team_season[str(year)] = {}
        league_avg = compute_avgshots_league(df, year)
        for team in teams_per_season[str(year)]:
            team_avg = compute_avgshots_team(df, year, team)
            avg_difference = team_avg - league_avg
            # smoothing results
            test_total_fine = gaussian_filter(avg_difference, sigma=sigma)

            # Filter out values that are very close to zero for plotting purposes
            test_total_fine[np.abs(test_total_fine - 0) <= threshold] = None

            # Store result
            shot_freq_team_season[str(year)][team] = test_total_fine

    return shot_freq_team_season


def get_half_rink_projection_df() -> pd.DataFrame:
    """
    Loads the "normal" tidy dataframe and returns a projection of the XY shot coordinates on a half-rink.
    Shortly said, takes the absolute value of XY coordinates.
    :return: half-rink projected dataframe. All columns identical to tidy_data.csv except x/y-coordinates.
    """
    half_rink_df = pd.read_csv(os.path.join(DATA_PATH, "tidy_data.csv"))
    # need to remove NaN coordinates. Low number of such instances (~16 in total).
    half_rink_df = half_rink_df.dropna(subset=["x_coordinate"])

    half_rink_df["x_coordinate_adj"] = np.where(
        half_rink_df["x_coordinate"] < 0,
        -half_rink_df["x_coordinate"],
        half_rink_df["x_coordinate"],
    )
    half_rink_df["y_coordinate_adj"] = np.where(
        half_rink_df["x_coordinate"] < 0,
        -half_rink_df["y_coordinate"] + 42,
        half_rink_df["y_coordinate"] + 42,
    )
    return half_rink_df