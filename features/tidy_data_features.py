import glob
import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from feature_engineering import shot_distance_compute_feature, add_offensive_side_feature, add_shot_angle,\
    change_in_shot_angle
from datetime import datetime

RAW_DATA_PATH = os.path.expanduser('~') + '/nhlapidata/raw'
DATA_DIR = os.path.expanduser('~')+ '/nhlapidata/csv'

class tidy_dataframe:
    """ Class used to build tidy dataframe using Raw data"""

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
                    event_dict["homepenaltystart"] = homepenaltystart
                    event_dict["awaypenaltystart"] = awaypenaltystart
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