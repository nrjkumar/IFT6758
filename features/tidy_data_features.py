import glob
import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from features.feature_eng import shot_distance_compute_feature, add_offensive_side_feature, add_shot_angle,\
    change_in_shot_angle
from datetime import datetime

RAW_DATA_PATH = os.path.expanduser('~') + '/nhlapidata/raw'
DATA_DIR = os.path.expanduser('~')+ '/nhlapidata/csv'

#RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'raw')
#DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')

class tidy_dataframe:
    """ Class used to build tidy dataframe using Raw data"""
    #print(RAW_DATA_PATH)
    def __init__(self, base_file_path=RAW_DATA_PATH):
        self.base_file_path = base_file_path
        
        self.features = ['game_id', 'season', 'date', 'home_team', 'away_team', 'game_time', 'period', 'period_time',
                         'team', 'shooter', 'goalie', 'is_goal', 'shot_type', 'x_coordinate', 'y_coordinate',
                         'is_empty_net', 'strength', 'is_playoff', 'home_goal', 'away_goal', 'game_time(s)',
                         'prev_event_type', 'prev_event_x', 'prev_event_y', 'time_since_prev_event',
                         'is_rebound', 'distance_to_prev_event', 'speed_since_prev_event', 'is_penalty_shot']
        self.penalty_features = ['game_id', "game_time","home_strength","away_strength","homepenaltystart",
                                 "awaypenaltystart"]

    def read_json_file(self, file_path) -> dict:
        """
        Decode the content of a single json file and returns the content as a dict.
        :param file_path: Path to json file
        :return: Parsed json file as dict
        """
        with open(file_path) as f:
            return json.load(f)

    def read_all_json(self) -> list:
        """
        Decodes all json file in dir self.base_file_path. Returns content as a list of dicts.
        :return: list of dict
        """
        json_files = glob.glob(os.path.join(self.base_file_path, '*.json'))
        print("reading json")
        return [self.read_json_file(file) for file in
                tqdm(json_files, total=len(json_files), desc="Reading JSON files")]

    def parse_penalty_data(self, json_data) -> list:
       
        game_data = []  # List of event dict
        event_dict = {}  # dictionary containing all features of a shot/goal
        
        
        # Verify we have all the necessary basic json keys
        if 'liveData' not in json_data or \
                'plays' not in json_data['liveData'] or \
                'allPlays' not in json_data['liveData']['plays']:
            return [None] * len(self.features)
        
        home_team = json_data['gameData']['teams']['home']['name']
        away_team = json_data['gameData']['teams']['away']['name']

        home_strength = 5
        away_strength = 5
        home_penalty_list = []
        home_major_list = []
        away_penalty_list = []
        away_major_list = []
        home_penalty_start = None
        away_penalty_start = None
        
        event_dict["period"] = 1
        event_dict["period_time"] = "00:00"
        event_dict["game_time"] = "00:00"
        event_dict["game_id"] = json_data['gamePk']
        event_dict["homepenaltystart"] = None
        event_dict["awaypenaltystart"] = None
        event_dict["home_strength"] = 5
        event_dict["away_strength"] = 5
        
        game_data.append(event_dict.copy())
        event_dict.clear()

        for event_id, event in enumerate(json_data['liveData']['plays']['allPlays']):
           
                    
            if home_major_list != []:
                gametime = f"{(int(event['about']['period'] - 1) * 20) + int(event['about']['periodTime'].split(':')[0])}:" \
                                   f"{ event['about']['periodTime'].split(':')[1]}"
                
                if gametime >= home_major_list[0] or len(gametime) >= len(home_major_list[0]):
                    home_strength += 1
                    
                    event_dict['period'] = event['about']['period']
                    event_dict['period_time'] = event['about']['periodTime']
                    
                    event_dict["game_time"] = home_major_list[0]
                    
                    if home_strength == 5:
                        home_penalty_start = None
                    
                    home_major_list.pop(0)
                    
                    event_dict["home_strength"] = home_strength
                    event_dict["away_strength"] = away_strength
                    event_dict["homepenaltystart"] = home_penalty_start
                    event_dict["awaypenaltystart"] = away_penalty_start
                    event_dict['game_id'] = json_data['gamePk']
                    game_data.append(event_dict.copy())
                    event_dict.clear()
                    
            if home_penalty_list != []:  
                   
                if event['result']['event'] == ('Goal') and event["team"]["name"] == away_team and home_strength != away_strength:
                   
                    
                    event_dict['period'] = event['about']['period']
                    event_dict['period_time'] = event['about']['periodTime']
                    
                    event_dict["game_time"] = f"{(int(event_dict['period']) - 1) * 20 + int(event_dict['period_time'].split(':')[0])}:" \
                               f"{event_dict['period_time'].split(':')[1]}"
                    
                    home_penalty_list[0] = f"{int(event_dict['game_time'].split(':')[0]) -2}:" \
                        f"{event_dict['game_time'].split(':')[1]}"

                    
                    if home_penalty_list[0] < event_dict["game_time"] or len(home_penalty_list[0]) < len(event_dict["game_time"]): #edge case going from sub 10 min to over 10 min, comparison between 9:59 and 11:59 don't work
                        home_penalty_list.pop(0)
                        home_strength += 1
                    
                    if home_strength == 5:
                        homepenaltystart = None
                        
                    event_dict["home_strength"] = home_strength
                    event_dict["away_strength"] = away_strength
                    event_dict["homepenaltystart"] = home_penalty_start
                    event_dict["awaypenaltystart"] = away_penalty_start
                    event_dict['game_id'] = json_data['gamePk']
                    game_data.append(event_dict.copy())
                    event_dict.clear()
                    continue
                
                gametime = f"{(int(event['about']['period'] - 1) * 20) + int(event['about']['periodTime'].split(':')[0])}:" \
                               f"{ event['about']['periodTime'].split(':')[1]}"      


        
                if gametime >= home_penalty_list[0] or len(gametime) > len(home_penalty_list[0]):
                    home_strength += 1
                    #print(homepenaltylist)
                    
                    event_dict['period'] = event['about']['period']
                    event_dict['period_time'] = event['about']['periodTime']
                    
                    event_dict["game_time"] = home_penalty_list[0]
                    
                    if home_strength == 5:
                        homepenaltystart = None
                    
                    home_penalty_list.pop(0)    
                    
                    event_dict["home_strength"] = home_strength
                    event_dict["away_strength"] = away_strength
                    event_dict["homepenaltystart"] = home_penalty_start
                    event_dict["awaypenaltystart"] = away_penalty_start
                    event_dict['game_id'] = json_data['gamePk']
                    game_data.append(event_dict.copy())
                    event_dict.clear()
                    
         
            if away_major_list != []:
                gametime = f"{(int(event['about']['period'] - 1) * 20) + int(event['about']['periodTime'].split(':')[0])}:" \
                                   f"{ event['about']['periodTime'].split(':')[1]}"
                
                if gametime >= away_major_list[0] or len(gametime) > len(away_major_list[0]):
                    away_strength += 1
                    
                    event_dict['period'] = event['about']['period']
                    event_dict['period_time'] = event['about']['periodTime']
                    
                    event_dict["game_time"] = away_major_list[0]
                    
                    if away_strength == 5:
                        awaypenaltystart = None
                    
                    away_major_list.pop(0)
                    
                    event_dict["home_strength"] = home_strength
                    event_dict["away_strength"] = away_strength
                    event_dict["homepenaltystart"] = home_penalty_start
                    event_dict["awaypenaltystart"] = away_penalty_start
                    event_dict['game_id'] = json_data['gamePk']
                    game_data.append(event_dict.copy())
                    event_dict.clear()
                    
                    
            if away_penalty_list != []:
                
                if event['result']['event'] == ('Goal') and event["team"]["name"] == home_team and home_strength != away_strength:
                    
                    event_dict['period'] = event['about']['period']
                    event_dict['period_time'] = event['about']['periodTime']
                    
                    event_dict["game_time"] = f"{(int(event_dict['period']) - 1) * 20 + int(event_dict['period_time'].split(':')[0])}:" \
                               f"{event_dict['period_time'].split(':')[1]}"
                    
                    away_penalty_list[0] = f"{int(event_dict['game_time'].split(':')[0]) -2}:"\
                        f"{event_dict['game_time'].split(':')[1]}"
                    
                    if away_penalty_list[0] < event_dict["game_time"] or len(away_penalty_list[0]) < len(event_dict["game_time"]):
                        away_penalty_list.pop(0)
                        away_strength +=1
                    
                    if away_strength == 5:
                        awaypenaltystart = None
                        
                    event_dict["home_strength"] = home_strength
                    event_dict["away_strength"] = away_strength
                    event_dict["homepenaltystart"] = home_penalty_start
                    event_dict["awaypenaltystart"] = away_penalty_start
                    event_dict['game_id'] = json_data['gamePk']
                    game_data.append(event_dict.copy())
                    event_dict.clear()
                    continue
                
                gametime = f"{(int(event['about']['period'] - 1) * 20) + int(event['about']['periodTime'].split(':')[0])}:" \
                               f"{event['about']['periodTime'].split(':')[1]}"
                
                if gametime >= away_penalty_list[0] or len(gametime) > len(away_penalty_list[0]):
                    away_strength += 1
                    
                    event_dict['period'] = event['about']['period']
                    event_dict['period_time'] = event['about']['periodTime']
                    
                    event_dict["game_time"] = away_penalty_list[0]
                    
                    if away_strength == 5:
                        awaypenaltystart = None
                    
                    away_penalty_list.pop(0)    
                    
                    event_dict["home_strength"] = home_strength
                    event_dict["away_strength"] = away_strength
                    event_dict["homepenaltystart"] = home_penalty_start
                    event_dict["awaypenaltystart"] = away_penalty_start
                    event_dict['game_id'] = json_data['gamePk']
                    game_data.append(event_dict.copy())
                    event_dict.clear()
                     
            
            
            if event['result']['event'] == 'Penalty' and (event["result"]["penaltySeverity"] in ['Minor', 'Bench Minor','Major'] )and (event['result']['secondaryType'] != 'Fighting'):
                    
                    
                    event_dict['period'] = event['about']['period']
                    event_dict['period_time'] = event['about']['periodTime']
                    
                    event_dict["game_time"] = f"{(int(event_dict['period']) - 1) * 20 + int(event_dict['period_time'].split(':')[0])}:" \
                               f"{event_dict['period_time'].split(':')[1]}"
                    if event["team"]["name"] == home_team:                        
                        home_strength -= 1
                        if event["result"]["penaltySeverity"]  in(["Minor", "Bench Minor"]):
                            home_penalty_list.append(f"{int(event_dict['game_time'].split(':')[0]) + event['result']['penaltyMinutes']}:"\
                                                 f"{event_dict['game_time'].split(':')[1]}")
                        elif event["result"]["penaltySeverity"] == "Major":
                            home_major_list.append(f"{int(event_dict['game_time'].split(':')[0])+5}:"\
                                                 f"{event_dict['game_time'].split(':')[1]}")
                        if home_penalty_start == None:
                            home_penalty_start = event_dict["game_time"]
                        home_penalty_list.sort()
                        
                    elif event["team"]["name"] == away_team:
                        away_strength -= 1
                        if event["result"]["penaltySeverity"]  in(["Minor", "Bench Minor"]):
                            away_penalty_list.append(f"{int(event_dict['game_time'].split(':')[0])+ + event['result']['penaltyMinutes']}:"\
                                                   f"{event_dict['game_time'].split(':')[1]}")
                        elif event["result"]["penaltySeverity"] == "Major":
                            away_major_list.append(f"{int(event_dict['game_time'].split(':')[0])+5}:"\
                                                f"{event_dict['game_time'].split(':')[1]}")
                        if away_penalty_start == None:
                            away_penalty_start = event_dict["game_time"]
                            away_penalty_list.sort()    

                    event_dict["home_strength"] = home_strength
                    event_dict["away_strength"] = away_strength
                    event_dict["homepenaltystart"] = home_penalty_start
                    event_dict["awaypenaltystart"] = away_penalty_start
                    event_dict['game_id'] = json_data['gamePk']
                    game_data.append(event_dict.copy())
                    event_dict.clear()           
        return game_data

    def parse_game_data(self, json_data) -> list:
        """
        Parses the required data from 1 json file (i.e. 1 game).
        :param json_data: json data to be parsed
        :return: returns a list of list (game shots/goals)
        """
        game_data = []  # List of event dict
        event_dict = {}  # dictionary containing all features of a shot/goal

        # Verify we have all the necessary basic json keys
        if 'liveData' not in json_data or \
                'plays' not in json_data['liveData'] or \
                'allPlays' not in json_data['liveData']['plays']:
            return [None] * len(self.features)

        for event_id, event in enumerate(json_data['liveData']['plays']['allPlays']):
            # Only interested in goals and shots
            if event['result']['event'] not in ('Goal', 'Shot'):
                continue

            event_dict['game_id'] = json_data['gamePk']
            event_dict['season'] = json_data['gameData']['game']['season']
            # Add datetime
            event_dict['date'] = event['about']['dateTime'][0:10]
            event_dict['period'] = event['about']['period']
            event_dict['period_time'] = event['about']['periodTime']
            # Total game time formatted mm:ss, no zero padding
            event_dict[
                'game_time'] = f"{(int(event_dict['period']) - 1) * 20 + int(event_dict['period_time'].split(':')[0])}:" \
                               f"{event_dict['period_time'].split(':')[1]}"
            event_dict['team'] = event['team']['name']
            event_dict['shooter'] = event['players'][0]['player']['fullName']
            event_dict['goalie'] = None
            for player in event['players']:
                if player['playerType'] == "Goalie":
                    event_dict['goalie'] = player['player']['fullName']

            event_dict['is_goal'] = True if event['result']['event'] == 'Goal' else False
            event_dict['shot_type'] = event['result']['secondaryType'] if 'secondaryType' in event['result'] else None
            event_dict['x_coordinate'] = event['coordinates']['x'] if 'x' in event['coordinates'] else None
            event_dict['y_coordinate'] = event['coordinates']['y'] if 'y' in event['coordinates'] else None
            event_dict['is_empty_net'] = event['result']['emptyNet'] if 'emptyNet' in event['result'] else None
            event_dict['strength'] = event['result']['strength']['name'] if 'strength' in event['result'] else None
            event_dict['is_playoff'] = json_data['gameData']['game']['type'] == "P"
            event_dict['home_team'] = json_data['gameData']['teams']['home']['name']
            event_dict['away_team'] = json_data['gameData']['teams']['away']['name']
            event_dict['home_goal'] = event['about']['goals']['home']
            event_dict['away_goal'] = event['about']['goals']['away']

            # Milestone 2 features below
            prev_event = json_data['liveData']['plays']['allPlays'][event_id - 1]

            event_dict['game_time(s)'] = int(event_dict['game_time'].split(':')[0]) * 60 + \
                                         int(event_dict['game_time'].split(':')[1])

            event_dict['prev_event_type'] = prev_event['result']['event']
            event_dict['prev_event_x'] = prev_event['coordinates']['x'] if 'x' in prev_event['coordinates'] else None
            event_dict['prev_event_y'] = prev_event['coordinates']['y'] if 'y' in prev_event['coordinates'] else None
            prev_event_period = int(prev_event['about']['period'])
            prev_event_period_time = prev_event['about']['periodTime'].split(':')
            game_time_prev_event = (prev_event_period - 1) * 20 * 60 + int(prev_event_period_time[0]) * 60 + \
                                   int(prev_event_period_time[1])
            event_dict['time_since_prev_event'] = event_dict['game_time(s)'] - game_time_prev_event
            event_dict['is_rebound'] = True if event_dict['prev_event_type'] == 'Shot' and \
                                               prev_event['team']['name'] == event_dict['team'] else False

            # distance_to_prev_event feature
            if event_dict['x_coordinate'] is not None and \
            event_dict['y_coordinate'] is not None and \
            event_dict['prev_event_x'] is not None and \
            event_dict['prev_event_y'] is not None:
                event_dict['distance_to_prev_event'] = np.linalg.norm(np.array([event_dict['x_coordinate'],
                                                                                event_dict['y_coordinate']]) -
                                                                      np.array([event_dict['prev_event_x'],
                                                                                event_dict['prev_event_y']]))
            else:
                event_dict['distance_to_prev_event'] = None

            # speed since prev event feature
            if event_dict['distance_to_prev_event'] and event_dict['time_since_prev_event']:
                event_dict['speed_since_prev_event'] = (event_dict['distance_to_prev_event'] / event_dict['time_since_prev_event'])
            else:
                event_dict['speed_since_prev_event'] = None

            if prev_event['result']['event'] == 'Penalty' and prev_event['result']['penaltySeverity'] == 'Penalty Shot':
                event_dict['is_penalty_shot'] = True
            elif not event_dict['is_playoff'] and event_dict['period'] == 5:
                event_dict['is_penalty_shot'] = True
            else:
                event_dict['is_penalty_shot'] = False
            assert (len(event_dict) == len(self.features))
            game_data.append(event_dict.copy())
            event_dict.clear()
        return game_data
        
    def parse_live_game_data(self, json_data, start_event = 0) -> list:
        """
        Parses the required data from 1 json file (i.e. 1 game).
        :param json_data: json data to be parsed
        :return: returns a list of list (game shots/goals)
        """
        game_data = []  # List of event dict
        event_dict = {}  # dictionary containing all features of a shot/goal
        period = None
        time_left = None
        
        # Verify we have all the necessary basic json keys
        if 'liveData' not in json_data or \
                'plays' not in json_data['liveData'] or \
                'allPlays' not in json_data['liveData']['plays']:
            return [None] * len(self.features)

        for event_id, event in enumerate(json_data['liveData']['plays']['allPlays']):
            if event['result']['event'] == 'Game_End':
                period = 3
                time_left = "0:00"
            else:
                period = int(event['about']['period'])
                period_time = event['about']['periodTime']
                time_left = f"{20-int(period_time.split(':')[0])}:{(60-int(period_time.split(':')[1])) % 60}"
                if len(period_time.split(':')[1]) == 1:
                    time_left = f"{20-int(period_time.split(':')[0])}:00"
                
            # Only interested in goals and shots
            if event['result']['event'] not in ('Goal', 'Shot'):
                continue
            if event_id < start_event:
                continue

            event_dict['game_id'] = json_data['gamePk']
            event_dict['season'] = json_data['gameData']['game']['season']
            # Add datetime
            event_dict['date'] = event['about']['dateTime'][0:10]
            event_dict['period'] = event['about']['period']
            event_dict['period_time'] = event['about']['periodTime']
            # Total game time formatted mm:ss, no zero padding
            event_dict[
                'game_time'] = f"{(int(event_dict['period']) - 1) * 20 + int(event_dict['period_time'].split(':')[0])}:" \
                               f"{event_dict['period_time'].split(':')[1]}"
            event_dict['team'] = event['team']['name']
            event_dict['shooter'] = event['players'][0]['player']['fullName']
            event_dict['goalie'] = None
            for player in event['players']:
                if player['playerType'] == "Goalie":
                    event_dict['goalie'] = player['player']['fullName']

            event_dict['is_goal'] = True if event['result']['event'] == 'Goal' else False
            event_dict['shot_type'] = event['result']['secondaryType'] if 'secondaryType' in event['result'] else None
            event_dict['x_coordinate'] = event['coordinates']['x'] if 'x' in event['coordinates'] else None
            event_dict['y_coordinate'] = event['coordinates']['y'] if 'y' in event['coordinates'] else None
            event_dict['is_empty_net'] = event['result']['emptyNet'] if 'emptyNet' in event['result'] else None
            event_dict['strength'] = event['result']['strength']['name'] if 'strength' in event['result'] else None
            event_dict['is_playoff'] = json_data['gameData']['game']['type'] == "P"
            event_dict['home_team'] = json_data['gameData']['teams']['home']['name']
            event_dict['away_team'] = json_data['gameData']['teams']['away']['name']
            event_dict['home_goal'] = event['about']['goals']['home']
            event_dict['away_goal'] = event['about']['goals']['away']

            # Milestone 2 features below
            prev_event = json_data['liveData']['plays']['allPlays'][event_id - 1]

            event_dict['game_time(s)'] = int(event_dict['game_time'].split(':')[0]) * 60 + \
                                         int(event_dict['game_time'].split(':')[1])

            event_dict['prev_event_type'] = prev_event['result']['event']
            event_dict['prev_event_x'] = prev_event['coordinates']['x'] if 'x' in prev_event['coordinates'] else None
            event_dict['prev_event_y'] = prev_event['coordinates']['y'] if 'y' in prev_event['coordinates'] else None
            prev_event_period = int(prev_event['about']['period'])
            prev_event_period_time = prev_event['about']['periodTime'].split(':')
            game_time_prev_event = (prev_event_period - 1) * 20 * 60 + int(prev_event_period_time[0]) * 60 + \
                                   int(prev_event_period_time[1])
            event_dict['time_since_prev_event'] = event_dict['game_time(s)'] - game_time_prev_event
            event_dict['is_rebound'] = True if event_dict['prev_event_type'] == 'Shot' and \
                                               prev_event['team']['name'] == event_dict['team'] else False

            # distance_to_prev_event feature
            if event_dict['x_coordinate'] is not None and \
            event_dict['y_coordinate'] is not None and \
            event_dict['prev_event_x'] is not None and \
            event_dict['prev_event_y'] is not None:
                event_dict['distance_to_prev_event'] = np.linalg.norm(np.array([event_dict['x_coordinate'],
                                                                                event_dict['y_coordinate']]) -
                                                                      np.array([event_dict['prev_event_x'],
                                                                                event_dict['prev_event_y']]))
            else:
                event_dict['distance_to_prev_event'] = None

            # speed since prev event feature
            if event_dict['distance_to_prev_event'] and event_dict['time_since_prev_event']:
                event_dict['speed_since_prev_event'] = (event_dict['distance_to_prev_event'] / event_dict['time_since_prev_event'])
            else:
                event_dict['speed_since_prev_event'] = None

            if prev_event['result']['event'] == 'Penalty' and prev_event['result']['penaltySeverity'] == 'Penalty Shot':
                event_dict['is_penalty_shot'] = True
            elif not event_dict['is_playoff'] and event_dict['period'] == 5:
                event_dict['is_penalty_shot'] = True
            else:
                event_dict['is_penalty_shot'] = False
            assert (len(event_dict) == len(self.features))
            game_data.append(event_dict.copy())
            event_dict.clear()
            
        return game_data, period, time_left









    def build_dataframe(self) -> pd.DataFrame:
        """
        This function builds the complete data frame by reading all jsons and storing them in a list,
        then parsing the data into a list of list and finally returns the dataframe
        :return: Resulting dataframe
        """
        json_data = self.read_all_json()
        result = []
        for game in tqdm(json_data, total=len(json_data), desc="Building Dataframe"):
            game_data = self.parse_game_data(game)
            if game_data == [None] * len(self.features):  # empty row
                continue
            result.extend([i for i in game_data])  # quicker than just extend

        # Make dataframe
        result = pd.DataFrame(result, columns=self.features)
        # Append engineered features
        print('Append home offensive side feature... ')
        result = add_offensive_side_feature(result)
        print('Append shot distance feature...')
        result = shot_distance_compute_feature(result)
        print('Append shot angle feature...')
        result = add_shot_angle(result)
        print('Append change in shot angle distance feature...')
        result = change_in_shot_angle(result)
        return result


    def build_penalty_dataframe(self) -> pd.DataFrame:
        """
        This function builds the complete data frame by reading all jsons and storing them in a list,
        then parsing the data into a list of list and finally returns the dataframe
        :return: Resulting dataframe
        """
        json_data = self.read_all_json()
        result = []
        for game in tqdm(json_data, total=len(json_data), desc="Building penalty Dataframe"):
            game_data = self.parse_penalty_data(game)
            if game_data == [None] * len(self.features):  # empty row
                continue
            result.extend([i for i in game_data])  # quicker than just extend
        
        # Make dataframe
        result = pd.DataFrame(result, columns=self.penalty_features)
        
        #imputing so number of player on ice is a minimum of 3
        result["home_strength"] = result.home_strength.apply(lambda x: np.where(x<3,3,x))
        result["away_strength"] = result.away_strength.apply(lambda x: np.where(x<3,3,x))
        resultdf =result.drop_duplicates(subset=["game_id", "game_time"], keep="last")
        print('Save CSV...')
        #result.to_csv(os.path.join(DATA_DIR, 'tidy_data_pen.csv'), index=False)
    
        return resultdf

def main():
    tidy_df = tidy_dataframe()
    df = tidy_df.build_dataframe()
    df_pen = tidy_df.build_penalty_dataframe()
    
    
    df["game_time_second"] = df['game_time'].str.split(':', expand = True)[0].astype(int) *60 + df['game_time'].str.split(':', expand = True)[1].astype(int)
    
    df2 = df.sort_values(by =["game_time_second","game_id"], ascending = (True, True))
    df_pen["game_time_second"] = df_pen['game_time'].str.split(':', expand = True)[0].astype(int) *60 + df_pen['game_time'].str.split(':', expand = True)[1].astype(int)
    df_pen2 = df_pen.sort_values(by=["game_time_second","game_id"], ascending = (True, True))
    df_merged = pd.merge_asof(df2, df_pen2, on= 'game_time_second', by="game_id")
    
    df_merged = df_merged.sort_values(by = ["game_id", "game_time_second"], ascending = (True, True))
    df_merged["game_time"] = df_merged["game_time_x"]
    df_merged = df_merged.drop(["game_time_y", "game_time_x"], axis=1)
    df_merged['time_since_home_pp'] = (df_merged['game_time'].str.split(':', expand = True)[0].astype('float') *60 + df_merged['game_time'].str.split(':', expand = True)[1].astype('float')) - \
        (df_merged['homepenaltystart'].str.split(':', expand = True)[0].astype('float') *60 + df_merged['homepenaltystart'].str.split(':', expand = True)[1].astype('float'))
    df_merged['time_since_away_pp'] = (df_merged['game_time'].str.split(':', expand = True)[0].astype('float') *60 + df_merged['game_time'].str.split(':', expand = True)[1].astype('float')) - \
        (df_merged['awaypenaltystart'].str.split(':', expand = True)[0].astype('float') *60 + df_merged['awaypenaltystart'].str.split(':', expand = True)[1].astype('float'))
    
    df_merged['time_since_home_pp'] = df_merged['time_since_home_pp'].fillna(0)
    df_merged['time_since_away_pp'] = df_merged['time_since_away_pp'].fillna(0)
    temp1 = [df_merged["away_team"] == df_merged['team']]* df_merged['time_since_home_pp'].to_numpy()
    temp2 = [df_merged["home_team"] == df_merged['team']]* df_merged['time_since_away_pp'].to_numpy()
    df_merged['time_since_pp'] = temp1.T + temp2.T
    df_merged = df_merged.drop(["homepenaltystart","awaypenaltystart","time_since_away_pp","time_since_home_pp"], axis = 1)
    
    
    #adding information relative to strength for each team based on penalty info
    df_merged["strength"] = (df_merged.home_strength.to_numpy() * [df_merged.home_team==df_merged.team] + df_merged.away_strength.to_numpy() * [df_merged.away_team == df_merged.team]).T
    df_merged["relative_strength"] = ([df_merged.team == df_merged.home_team] * (df_merged.strength - df_merged.away_strength).to_numpy() + [df_merged.team == df_merged.away_team] * (df_merged.strength - df_merged.home_strength).to_numpy()).T
    
    #correctly fill empty_net column, empty = False
    df_merged['is_empty_net'] = df_merged['is_empty_net'].fillna(False)
    
    print('Save CSV...')
    df_merged.to_csv(os.path.join(DATA_DIR, 'tidy_data.csv'), index=False)
    
if __name__ == "__main__":
    main()
