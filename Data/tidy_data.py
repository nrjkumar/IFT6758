import glob
import json
import os
import pandas as pd
from tqdm import tqdm
from matchstatistics import add_shot_distance_feature, add_home_offensive_side_feature

#RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
RAW_DATA_PATH = os.path.expanduser('~')+ '/nhlapidata/raw'
DATA_DIR = os.path.expanduser('~')+ '/nhlapidata/csv'


class DataFrameBuilder:
    """
    Class used to build tidy dataframe from raw data.
    !Careful! Execution of make_data_frame() is slow. main bottleneck is I/O of jsons: 91% of exec time while
    profiling on an SSD, so not much can be done to speed it up except faster drives.
    """

    def __init__(self, base_file_path=RAW_DATA_PATH):
        self.base_file_path = base_file_path
        self.features = ['game_id', 'season', 'date', 'home_team', 'away_team', 'game_time', 'period', 'period_time',
                         'team', 'shooter', 'goalie','is_goal', 'shot_type', 'x_coordinate', 'y_coordinate',
                         'is_empty_net','strength', 'is_playoff',  'home_goal', 'away_goal']

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
        return [self.read_json_file(file) for file in tqdm(json_files, total=len(json_files), desc="Reading JSON files")]

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

        for event in json_data['liveData']['plays']['allPlays']:
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
            assert (len(event_dict) == len(self.features))
            game_data.append(event_dict.copy())
            event_dict.clear()
        return game_data

    def make_dataframe(self) -> pd.DataFrame:
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
        print('Append Engineered Features and save csv...')
        # Append engineered features
        result = add_home_offensive_side_feature(result)
        print(result)
        result = add_shot_distance_feature(result)
        return result


def main():
    df_builder = DataFrameBuilder()
    result = df_builder.make_dataframe()
    result.to_csv(os.path.join(DATA_DIR, 'tidy_data.csv'), index=False)


if __name__ == "__main__":
    main()
