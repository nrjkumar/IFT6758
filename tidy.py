import pandas as pd
import pickle
import json
import csv
import os
from os.path import exists
from tqdm import tqdm


from data.NHLPlayByPlay import NHLPlayByPlay


def load_season_from_pkl(season = '2016', gameType="02") :
  localdir = os.getcwd() + '/nhlapidata'
  csvdir = os.getcwd() + '/data'
  FileName = localdir + '/' + season + gameType + '.pkl'
  output_csv = csvdir + '/' + season + gameType + '.csv'
  play_types = ['SHOT', 'GOAL']   # Fixed locally... with possibility to evolve and be received as a parameter


  if (not os.path.isfile(FileName)):
    NHLdata = NHLPlayByPlay(season,gameType, localdir=localdir)

  file = open(FileName, 'rb')
  pkData = pickle.load(file)

  if exists(output_csv):
    os.remove(output_csv)

  pcolumns = ['gamePk', 'eventIdx', 'eventTypeId', 'eventId', 'period', 'periodType', 'periodTime', 'dateTime', 'teamId', 'teamName', 'x', 'y', 'shotType', 'emptyNet', 'strength', 'shooter', 'goalie']

  with open(output_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(pcolumns)
    games = pkData.keys()

    for gameId in tqdm(games) :
      events = pkData[gameId]['liveData']['plays']['allPlays']
      # Filtering SHOTs and GOALs
      shots_data = filter(  lambda x: x['result']['eventTypeId'] in play_types, events)
      for x in shots_data :
        # Retrieving nested json data...
        x_shooter = ""
        x_goalie = ""
        # x_shotType = ""
        x_emptyNet = ""
        x_strength = ""
        x_secondaryType = ""
        try:
          x_secondaryType = x['result']['secondaryType']
        except:
          print(f"No secondary type data for eventIdx {x['about']['eventIdx']}")

        if (x['result']['eventTypeId'] == 'GOAL') :
          try:
            x_strength = x['result']['strength']['name']
          except:
            print(f"No strength data for GamePk {gameId} - EventIdx {x['about']['eventIdx']}")

          try:
            x_emptyNet = x['result']['emptyNet']
          except:
            print(f"No emptyNet data for GamePk {gameId} - EventIdx {x['about']['eventIdx']}")

        for player in x['players'] :
          if player['playerType'] == "Shooter" :
            try:
              x_shooter = player['player']['fullName']
            except:
              print(f"No Shooter-fulName data for GamePk {gameId} - EventIdx {x['about']['eventIdx']}")

          if player['playerType'] == "Goalie" :
            try:
              x_goalie = player['player']['fullName']
            except:
              print(f"No Goalie-fulName data for GamePk {gameId} - EventIdx {x['about']['eventIdx']}")


        row_x = [gameId,      #x['gamePk'],
              x['about']['eventIdx'], 
              x['result']['eventTypeId'],           
              x['about']['eventId'],
              x['about']['period'],
              x['about']['periodType'], 
              x['about']['periodTime'], 
              x['about']['dateTime'],
              x['team']['id'],
              x['team']['name'],
              x['coordinates']['x'],
              x['coordinates']['y'],
              x_secondaryType,
              x_emptyNet,
              x_strength,
              x_shooter,
              x_goalie  ] 
        writer.writerow(row_x)
        
  return


def load_game_from_json(fileName):  #, playTypes = ['SHOT', 'GOAL']) :
  # The types of events we're interested in  (acording to 'eventTypeId' from NHL live data):
  play_types = ['SHOT', 'GOAL']   # Fixed locally... with possibility to evolve and be received as a parameter

  # Open the input data-file, with the json retrieved from NHL api
  with open(file_name, 'r') as f :
    data = json.load(f)

  events = data['liveData']['plays']['allPlays']

  # Filtering the events (SHOTs and GOALs)

  # Initialising Plays DataFrame
# plays = pd.DataFrame(columns=['gamePk', 'eventIdx', 'eventId', 'period', 'periodType', 'periodTime', 'dateTime', 
#                       'team', 'eventTypeId', 'x', 'y', ])

  events = data['liveData']['plays']['allPlays']
  shots_data = filter(  lambda x: x['result']['eventTypeId'] in play_types, events)
  pcolumns = ['eventIdx', 'eventTypeId', 'eventId', 'period', 'periodType', 'periodTime', 'dateTime', 'teamId', 'teamName', 'x', 'y', 'shotType', 'emptyNet', 'strength', 'shooter', 'goalie']

  if exists(output_csv):
    os.remove(output_csv)

  with open(output_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(pcolumns)
    for x in shots_data :
      # Retrieving nested json data...
      x_shooter = ""
      x_goalie = ""
      # x_shotType = ""
      x_emptyNet = ""
      x_strength = ""

      x_secondaryType = ""

      if (x['result']['eventTypeId'] == 'GOAL') :
        x_strength = x['result']['strength']['name']
        x_emptyNet = x['result']['emptyNet']

      for player in x['players'] :
        if player['playerType'] == "Shooter" :
          x_shooter = player['player']['fullName']
        if player['playerType'] == "Goalie" :
          x_goalie = player['player']['fullName']

      try:
        x_secondaryType = x['result']['secondaryType']
      except:
        print(f"No secondary type data for eventIdx {x['about']['eventIdx']}")

      row_x = [x['about']['eventIdx'], 
            x['result']['eventTypeId'],           
            x['about']['eventId'],
            x['about']['period'],
            x['about']['periodType'], 
            x['about']['periodTime'], 
            x['about']['dateTime'],
            x['team']['id'],
            x['team']['name'],
            x['coordinates']['x'],
            x['coordinates']['y'],
            x_secondaryType,
            x_emptyNet,
            x_strength,
            x_shooter,
            x_goalie  ]  #])
      writer.writerow(row_x)


def main():
  seasons = ['2016', '2017', '2018', '2019', '2020'] 
  gameTypes = ['01', '02', '03']

  for season in seasons:
    for gameType in gameTypes:
      try:
        load_season_from_pkl(season, gameType)
      except:
        print(f"Error loading data from season {season} - type {gameType}")

if __name__ == "__main__":
    main()