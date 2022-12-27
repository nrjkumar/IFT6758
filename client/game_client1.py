import json
import requests
import time
import logging
import os,sys

#os.chdir(os.path.dirname(os.path.abspath(__file__)))
currentdir = os.path.dirname(os.path.realpath(__file__))


parentdir = os.path.dirname(currentdir)

sys.path.append(parentdir)
from features.utility import *
from features.tidy_data_features import  tidy_dataframe
from features.feature_eng  import shot_distance_compute_feature, add_offensive_side_feature, add_shot_angle,\
    change_in_shot_angle

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
LIVE_RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'live_raw')
GAME_ID_FLAG = 'GAME_ID_FLAG'


class Game_Client:
    def __init__(self, game_id: int,
                 api_url: str = f"https://statsapi.web.nhl.com/api/v1/game/{GAME_ID_FLAG}/feed/live/") -> None:
        self.save_path = LIVE_RAW_DATA_PATH
        self.api_url = api_url.replace(GAME_ID_FLAG, str(game_id))
        self.game_id = game_id
        self.last_event_processed = 0
        self.file_path = os.path.join(self.save_path, f'{int(game_id)}.json')
        self.period = 1
        self.time_left = "20:00"
        self.home_strength =  5
        self.away_strength = 5
        self.homepenaltylist = []
        self.awaypenaltylist = []
        self.homemajorlist = []
        self.awaymajorlist = [] 
        self.homepenaltystart= None
        self.awaypenaltystart= None

    def call_api(self):
        """
        Pings the API and returns all the new events since previous ping as a list.
        Save current files under data/live_raw
        Returns: New events as a list
        """
        # Request API
        response = requests.get(self.api_url)

        data_exist = os.path.isfile(self.file_path)

        if not data_exist:
            # Write to file
            with open(self.file_path, 'w') as f:
                f.write(response.text)

            with open(self.file_path, 'r+') as f:
                game = json.load(f)
                #DFB = DataFrameBuilder()
                DFB = tidy_dataframe()
                game_data, self.period, self.time_left = DFB.parse_live_game_data(game)
                game_data = pd.DataFrame(game_data, columns=DFB.features)
                # Append engineered features
                game_data = add_offensive_side_feature(game_data)
                game_data = shot_distance_compute_feature(game_data)
                game_data = add_shot_angle(game_data)
                game_data = change_in_shot_angle(game_data)
                game_data_penalty, self.home_strength, self.away_strength, self.homepenaltylist, self.awaypenaltylist, self.homemajorlist, self.awaymajorlist, self.homepenaltystart, self.awaypenaltystart = DFB.parse_penalty_data_ongoing(game)
                
                game_data_penalty = pd.DataFrame(game_data_penalty, columns=DFB.penalty_features)
                
                #imputing so number of player on ice is a minimum of 3
                game_data_penalty["home_strength"] = game_data_penalty.home_strength.apply(lambda x: np.where(x<3,3,x))
                game_data_penalty["away_strength"] = game_data_penalty.away_strength.apply(lambda x: np.where(x<3,3,x))
                game_data_penalty =game_data_penalty.drop_duplicates(subset=["game_id", "game_time"], keep="last")
                
                game_data["game_time_second"] = game_data['game_time'].str.split(':', expand = True)[0].astype(int) *60 + game_data['game_time'].str.split(':', expand = True)[1].astype(int)
    
                game_data2 = game_data.sort_values(by =["game_time_second","game_id"], ascending = (True, True))
                
                game_data_penalty["game_time_second"] = game_data_penalty['game_time'].str.split(':', expand = True)[0].astype(int) *60 + game_data_penalty['game_time'].str.split(':', expand = True)[1].astype(int)
                game_data_penalty2 = game_data_penalty.sort_values(by=["game_time_second","game_id"], ascending = (True, True))
                
                game_data = pd.merge_asof(game_data2, game_data_penalty2, on= 'game_time_second', by="game_id")
                
                game_data = game_data.sort_values(by = ["game_id", "game_time_second"], ascending = (True, True))
                game_data["game_time"] = game_data["game_time_x"]
                game_data = game_data.drop(["game_time_y", "game_time_x"], axis=1)
                game_data['time_since_home_pp'] = (game_data['game_time'].str.split(':', expand = True)[0].astype('float') *60 + game_data['game_time'].str.split(':', expand = True)[1].astype('float')) - \
                    (game_data['homepenaltystart'].str.split(':', expand = True)[0].astype('float') *60 + game_data['homepenaltystart'].str.split(':', expand = True)[1].astype('float'))
                game_data['time_since_away_pp'] = (game_data['game_time'].str.split(':', expand = True)[0].astype('float') *60 + game_data['game_time'].str.split(':', expand = True)[1].astype('float')) - \
                    (game_data['awaypenaltystart'].str.split(':', expand = True)[0].astype('float') *60 + game_data['awaypenaltystart'].str.split(':', expand = True)[1].astype('float'))
                
                game_data['time_since_home_pp'] = game_data['time_since_home_pp'].fillna(0)
                game_data['time_since_away_pp'] = game_data['time_since_away_pp'].fillna(0)
                temp1 = [game_data["away_team"] == game_data['team']]* game_data['time_since_home_pp'].to_numpy()
                temp2 = [game_data["home_team"] == game_data['team']]* game_data['time_since_away_pp'].to_numpy()
                game_data['time_since_pp'] = temp1.T + temp2.T
                game_data = game_data.drop(["homepenaltystart","awaypenaltystart","time_since_away_pp","time_since_home_pp"], axis = 1)
                
                
                #adding information relative to strength for each team based on penalty info
                game_data["strength"] = (game_data.home_strength.to_numpy() * [game_data.home_team==game_data.team] + game_data.away_strength.to_numpy() * [game_data.away_team == game_data.team]).T
                game_data["relative_strength"] = ([game_data.team == game_data.home_team] * (game_data.strength - game_data.away_strength).to_numpy() + [game_data.team == game_data.away_team] * (game_data.strength - game_data.home_strength).to_numpy()).T
                
                #correctly fill empty_net column, empty = False
                game_data['is_empty_net'] = game_data['is_empty_net'].fillna(False)
                
                #Removing data that can't be used for predictions due to missing values
                game_data = game_data.dropna()
                
                #filtering out shots that have a high probability of being due to faulty equipment (further than 100 feet away goals that are not empty net)
                game_data= game_data.drop(game_data[(game_data.shot_distance > 100.) & (game_data.is_goal == True) &(game_data.is_empty_net != True) ].index)
                
                self.last_event_processed = len(game['liveData']['plays']['allPlays'])-1
                
                return game_data, self.period, self.time_left
                
                
        else:
            with open(self.file_path, 'r+') as f:
                game = json.load(f)
                DFB = tidy_dataframe()
                game_data, self.period, self.time_left = DFB.parse_live_game_data(game, start_event=self.last_event_processed)
                game_data = pd.DataFrame(game_data, columns=DFB.features)
                game_data = game_data.iloc[1:] #drop last_event which was processed last run
                # Append engineered features
                game_data = add_offensive_side_feature(game_data)
                game_data = shot_distance_compute_feature(game_data)
                game_data = add_shot_angle(game_data)
                game_data = change_in_shot_angle(game_data)
                game_data_penalty, self.home_strength, self.away_strength, self.homepenaltylist, self.awaypenaltylist, self.homemajorlist, self.awaymajorlist, self.homepenaltystart, self.awaypenaltystart = DFB.parse_penalty_data_ongoing(game,self.home_strength, self.away_strength, self.homepenaltylist, self.awaypenaltylist, self.homemajorlist, self.awaymajorlist, self.homepenaltystart, self.awaypenaltystart, start_event = self.last_event_processed, first_run = False)
                game_data_penalty = pd.DataFrame(game_data_penalty, columns=DFB.penalty_features)
                
                #imputing so number of player on ice is a minimum of 3
                game_data_penalty["home_strength"] = game_data_penalty.home_strength.apply(lambda x: np.where(x<3,3,x))
                game_data_penalty["away_strength"] = game_data_penalty.away_strength.apply(lambda x: np.where(x<3,3,x))
                game_data_penalty =game_data_penalty.drop_duplicates(subset=["game_id", "game_time"], keep="last")
                
                game_data["game_time_second"] = game_data['game_time'].str.split(':', expand = True)[0].astype(int) *60 + game_data['game_time'].str.split(':', expand = True)[1].astype(int)
    
                game_data2 = game_data.sort_values(by =["game_time_second","game_id"], ascending = (True, True))
                
                game_data_penalty["game_time_second"] = game_data_penalty['game_time'].str.split(':', expand = True)[0].astype(int) *60 + game_data_penalty['game_time'].str.split(':', expand = True)[1].astype(int)
                game_data_penalty2 = game_data_penalty.sort_values(by=["game_time_second","game_id"], ascending = (True, True))
                
                game_data = pd.merge_asof(game_data2, game_data_penalty2, on= 'game_time_second', by="game_id")
                
                game_data = game_data.sort_values(by = ["game_id", "game_time_second"], ascending = (True, True))
                game_data["game_time"] = game_data["game_time_x"]
                game_data = game_data.drop(["game_time_y", "game_time_x"], axis=1)
                game_data['time_since_home_pp'] = (game_data['game_time'].str.split(':', expand = True)[0].astype('float') *60 + game_data['game_time'].str.split(':', expand = True)[1].astype('float')) - \
                    (game_data['homepenaltystart'].str.split(':', expand = True)[0].astype('float') *60 + game_data['homepenaltystart'].str.split(':', expand = True)[1].astype('float'))
                game_data['time_since_away_pp'] = (game_data['game_time'].str.split(':', expand = True)[0].astype('float') *60 + game_data['game_time'].str.split(':', expand = True)[1].astype('float')) - \
                    (game_data['awaypenaltystart'].str.split(':', expand = True)[0].astype('float') *60 + game_data['awaypenaltystart'].str.split(':', expand = True)[1].astype('float'))
                
                game_data['time_since_home_pp'] = game_data['time_since_home_pp'].fillna(0)
                game_data['time_since_away_pp'] = game_data['time_since_away_pp'].fillna(0)
                temp1 = [game_data["away_team"] == game_data['team']]* game_data['time_since_home_pp'].to_numpy()
                temp2 = [game_data["home_team"] == game_data['team']]* game_data['time_since_away_pp'].to_numpy()
                game_data['time_since_pp'] = temp1.T + temp2.T
                game_data = game_data.drop(["homepenaltystart","awaypenaltystart","time_since_away_pp","time_since_home_pp"], axis = 1)
                
                
                #adding information relative to strength for each team based on penalty info
                game_data["strength"] = (game_data.home_strength.to_numpy() * [game_data.home_team==game_data.team] + game_data.away_strength.to_numpy() * [game_data.away_team == game_data.team]).T
                game_data["relative_strength"] = ([game_data.team == game_data.home_team] * (game_data.strength - game_data.away_strength).to_numpy() + [game_data.team == game_data.away_team] * (game_data.strength - game_data.home_strength).to_numpy()).T
                
                #correctly fill empty_net column, empty = False
                game_data['is_empty_net'] = game_data['is_empty_net'].fillna(False)
                
                #Removing data that can't be used for predictions due to missing values
                game_data = game_data.dropna()
                
                #filtering out shots that have a high probability of being due to faulty equipment (further than 100 feet away goals that are not empty net)
                game_data= game_data.drop(game_data[(game_data.shot_distance > 100.) & (game_data.is_goal == True) &(game_data.is_empty_net != True) ].index)
                
                self.last_event_processed = len(game['liveData']['plays']['allPlays'])-1
                
                return game_data, self.period, self.time_left

    def timed_call_api(self, time_to_run: int = 120.0, time_interval = 60.0)-> list:
        """
        Pings the API every `time_to_interval` seconds.
        Args:
            time_to_run: Total thing to idly ping the API in seconds. Defaults to 120 seconds
            time_interval: Interval between each API ping. Defaults to 60 seconds
        Returns: A list of all new events collected during `time_to_run`
        """
        new_events = []
        starttime = time.time()
        while (time.time() - starttime) < time_to_run:
            new_events.extend(self.call_api())
            time.sleep(time_interval - ((time.time() - starttime) % time_interval))
