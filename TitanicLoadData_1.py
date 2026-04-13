import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

#-------------------------------------------------------------
# Function Name: DisplayInfo
# Description :  Print Border and title
# Parameters :   title(str)
# Return :       None
# Date :         14/03/2026
# Author :       Haritalika Parasram Munde
#-------------------------------------------------------------

def DisplayInfo(title):
    print("\n"+"="*70)
    print(title)
    print("="*70)

#-------------------------------------------------------------
# Function Name: ShowData
# Description :  It shows basic information about dataset
# Parameters :   df
#                df ->      Pandas dataframe object
#                message
#                message -> Heading text to display
# Return :       None
# Date :         14/03/2026
# Author :       Haritalika Parasram Munde
#-------------------------------------------------------------

def ShowData(df, message):
    DisplayInfo(message)

    print("First five rows of dataset")
    print(df.head())

    print("\nshape of dataset")
    print(df.shape)

    print("\nCloumn names :")
    print(df.columns.tolist())

    print("\nMissing values in each column")
    print(df.isnull().sum())

#-------------------+------------------------------------------
# Function Name: MarvellousTitanicLogistic
# Description :  This is main pipeline controller
#                it loads the dataset, shows the raw data
#                it preprocess the dataset & train the model
# Parameters :   Data path of dataset file
# Return :       None
# Date :         14/03/2026
# Author :       Haritalika Parasram Munde
#-------------------------------------------------------------

def MarvellousTitanicLogistic(DataPath):
    DisplayInfo("Step 1: Loading the Dataset ")

    df=pd.read_csv(DataPath)

    ShowData(df,"Initial Dataset")

#-------------------------------------------------------------
# Function Name: main
# Description :  Strating point of the application
# Parameters :   None
# Return :       None
# Date :         14/03/2026
# Author :       Haritalika Parasram Munde
#-------------------------------------------------------------
def main():
    MarvellousTitanicLogistic("MarvellousTitanicDataset.csv")


if __name__=="__main__":
    main()