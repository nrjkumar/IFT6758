import pandas as pd
import numpy as np
import os
import pickle
#from IFT6758-PROJECT.
""" Alternate Approach don't use this file as of now """
localdir = os.path.expanduser('~') + '/nhlapidata/'

def Load_Clean_Raw_Data(LogRawArtifact, localdir):
    print("Downloading Artifact.....Patience!!!")
    local_artifact = LogRawArtifact.download(localdir)

    TidyData = None
    for asset in local_artifact.assets:
        localTempFile = local_artifact.download_local_path + "/" + asset.logical_path
        print("Parsing " + localTempFile)
        with open(localTempFile,'rb') as file:
            d = pickle.load(file)
            if TidyData is None:
                TidyData = d.tidyData()
            else:
                TidyData = pd.concat([TidyData,d.tidyData()],ignore_index=True)
    return TidyData
