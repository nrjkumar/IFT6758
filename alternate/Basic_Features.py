from comet_ml import Experiment
from comet_ml import Artifact
import pandas as pd
import numpy as np
import math,json,os

# from  IFT6758-PROJECT.data.data-acquisition import Season
# import pandas as pd
#from features.utility import Load_Clean_Raw_Data
import IFT6758

def ComputeAngle(x,y):
    
    if x < 89:
        angle_degree = math.atan(y / (89-x)) *(180/math.pi)
    elif x == 89 and y >= 0:
        angle_degree = 90  #straight 90 degree angle from the net
    elif x == 89 and y < 0:
        angle_degree = -90 #other side
    elif x > 89 and y >= 0:
        angle_degree = 90 + (90 + math.atan(y / (89-x)) *(180/math.pi))  #behind the net positive side
    else:
        angle_degree = -90 - (90 - math.atan(y / (89-x)) *(180/math.pi))  #behind the net negative side
    return angle_degree


def Basic_Features(TidyData,gametype="Regular Season"):
    
    #Only keep the requested GameType
    TidyData = TidyData[TidyData["GameType"] == gametype]

    #Keep only Shot Events
    TidyData = TidyData[(TidyData["EventType"] == "SHOT") | (TidyData["EventType"] == "GOAL")]

    #IsGoal
    TidyData["IsGoal"] = [0 if value == "SHOT" else 1 for index, value in TidyData["EventType"].items()]

    #IsEmptyNet
    TidyData["IsEmptyNet"] = [0 if value == False else 1 for index, value in TidyData["EmptyNet"].items()]

    #Note that 
    #DistanceFromNet XCoordsNormNoFix/YCoordsNormNoFix are used instead of
    TidyData["DistanceFromNetWithError"] =  np.sqrt(((TidyData['XCoordsNormNoFix'] - 89)**2) + (TidyData['YCoordsNormNoFix']**2))
    TidyData["DistanceFromNet"] =  np.sqrt(((TidyData['XCoordsNorm'] - 89)**2) + (TidyData['YCoordsNorm']**2))

    #AngleFromNet
    TidyData["AngleFromNetWithError"] = [ComputeAngle(TidyData.loc[ind,'XCoordsNormNoFix'], TidyData.loc[ind,'YCoordsNormNoFix']) for ind in TidyData.index]
    TidyData["AngleFromNet"] = [ComputeAngle(TidyData.loc[ind,'XCoordsNorm'], TidyData.loc[ind,'YCoordsNorm']) for ind in TidyData.index]

    #Make an Shot Identifer  from the GameId + Period + PeriodTimeInSeconds
    TidyData["ShotID"]  = [str(ind) + "_" + TidyData.loc[ind,'GameId'] + "_" + str(TidyData.loc[ind,'Period']) + "_" + str(TidyData.loc[ind,'PeriodTimeSeconds']) for ind in TidyData.index]
    
    return TidyData



if __name__ == '__main__':

    #location
    #localdir = os.environ.get('IFT6758-NHLDATA')
    localdir = os.path.expanduser('~')+'/nhlapidata/'
    #Comet.ml related
    project_name="ift6758-17"
    workspace="nrjscorprion"
    api_key = os.environ.get('COMET_API_KEY')

    #Ensure variables are set
    if localdir is None or localdir == "":
        raise ValueError('The environment variable IFT6758-NHLDATA is not set.  Please set it to the location of a local folder to store the data')

    if api_key is None or api_key == "":
        raise ValueError('The environment variable COMET_API_KEY is not set.  Please set it to the value of your comet.ml API key')

    #Ensure localCache exists
    if not os.path.isdir(localdir):
        raise ValueError('The provided local cache folder does not exists.  Please review the value of the IFT6758-NHLDATA environment variable')

    #Create a new experiment
    experiment = Experiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspace,
    )
    #Retrieve the raw training data artifact
    logged_artifact = experiment.get_artifact("raw-data-training",workspace="wcallag3")

    #Compute the basic features on this
    gameType = "Regular Season"
    TidyData = Load_Clean_Raw_Data(logged_artifact, localdir)
    data = Basic_Features(TidyData,gameType)

    #Keep Only new columns
    data = data[['ShotID','IsGoal', 'IsEmptyNet', 'DistanceFromNetWithError', 'DistanceFromNet', 'AngleFromNetWithError', 'AngleFromNet']]

    #Replace the Index with the ShotID created
    data = data.set_index('ShotID')

    #Log as dataset info the input artifact name/version and Game Type selected for this experiment
    experiment.log_dataset_info(name=logged_artifact.name, version=logged_artifact.version)
    experiment.log_dataset_info(name="Regular Season")

    # Save output artifact
    localTempFile = localdir+"/M2_BasicFeatures_Training.csv"
    data.to_csv(localTempFile)
    artifactTrainingBasicFeatures = Artifact("basic-features-training", "dataset")
    artifactTrainingBasicFeatures.add(localTempFile)

    experiment.log_artifact(artifactTrainingBasicFeatures)
    experiment.end()

