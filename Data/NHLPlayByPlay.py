import pandas as pd
import pickle
import os
import copy, json
import requests
import warnings

class NHLPlayByPlay:
    def __init__(self, season = '2016', GameType="02", localdir = os.getcwd() + '/nhlapidata' ,
        Base_Url = "https://statsapi.web.nhl.com/api/v1/"):

        FileName = localdir + '/' + season + GameType + '.pkl'
        #print(FileName)
        GameData = {}
        self.seasons = {season : {GameType : {"GameData" : GameData,
                                              "nbGames" : 0,
                                              "FileName" : FileName}}}
        self.localdir =localdir

        if os.path.isfile(FileName):
            #file is stored in local system
            #Load file and print contents
            with open(FileName,'rb') as file:
                self.seasons[season][GameType]["GameData"] = pickle.load(file)
            self.seasons[season][GameType]["nbGames"] = len(self.seasons[season][GameType]["GameData"])
            print("Loaded " + str(self.seasons[season][GameType]["nbGames"]) + " Games stored in local drive" + " " + FileName)
        else:
            #Not Available locally, so downloading 
            # The game id nomenclature is different for game type "03 vs game types ["01","02"]
            if GameType in ["01","02"]:
                
                #Buildup the game id string with the season, gametype and Game Number (1-based)
                #First game starts at 0001
                GameCount = 1
                GameId = season + GameType + ("%04d" % GameCount)
                
                #Query the API endpoint for the first game
                response = requests.get(Base_Url +"/game/" + GameId + "/feed/live/")
                
                #Loop over all games, until we receive a 404 response indicating we got all games for the season
                while response.status_code != 404:
                
                    #Status code should be 200 when the API returns data for our request
                    if response.status_code == 200:
                        GameData[GameId] = response.json()
                    else:
                        #Other status code, raise error.
                        raise ValueError('Got status code ' + response.status_code + " for GameId = " + GameId)
                    
                    #Next game Id
                    GameCount +=1
                    GameId = season + GameType + ("%04d" % GameCount)
                    #Query API for the next game
                    response = requests.get(Base_Url +"/game/" + GameId + "/feed/live/")
            elif GameType == '03':
                #playoff game
                
                #r is the playoff round (1 to 4).  Included round 0 because the 20192020 playoff had an unsual qualyfing round 0 because of the COVID-19 pandemic
                for rounds in range(0,5):
                    # m is the matchup.  There are 8 matchups in round 1, 4 matchups in round 2, 2 matchups in round 3 and 1 matchup in the final round
                    for matchup in range(1,int(16/int(2**rounds))+1):
                        #There game be up to 7 games in a matchup (first team to win 4 out of 7)
                        for game in range(1,8):
                            #Buildup the gameid from round, matchup and game
                            GameId = season + GameType + "0" + str(rounds) + str(matchup) + str(game)
                            
                            #Query the NHL API
                            response = requests.get(Base_Url +"/game/" + GameId + "/feed/live/")
                            
                            #If game not found, this means that we got all games from this matchup (there may be less than 7)
                            if response.status_code == 404:
                                break  #break out of inner loop
                            #Status code should be 200 if API returns data for the request
                            elif response.status_code == 200:
                                GameData[GameId] = response.json()
                            else:
                                #Other status code, raise error.
                                raise ValueError('Got status code ' + response.status_code + " for gameId = " + GameId)
            
            os.makedirs(localdir, exist_ok=True)
            
            #Store the file in our local cache using the pickle format
            with open(FileName,'wb') as file:
                pickle.dump(GameData,file)
            
            self.seasons[season][GameType]["nbGames"] = len(GameData)

            
            #Print a summary to the user of what was downloaded.
            print("Downloaded " + str(self.seasons[season][GameType]["nbGames"]) + " games and saved them to local cache in " + FileName)

    def dataset_info(self):
        """
        Print a summary of the dataset
        """
        if self.GameId is None:  
            for s in self.seasons:
                print("Season: " + s )
                for t in self.seasons[s]:
                    print("  Game Type: " + t +"("+ self.SUPPORTED_GAME_TYPES[t] +") : "+ str(self.seasons[s][t]["nbGames"]) +  " games")    
        else:
            print(f"Live Game Id : {self.gameId} ; LastEventProcessed : {self.lastEventProcessed}")

    def GameData(self,season,GameType):
        if season in self.seasons:
            if GameType in self.seasons[season]:
                return self.seasons[season][GameType]['gameData']