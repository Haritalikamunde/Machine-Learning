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

#-------------------------------------------------------------
# Function Name: CleanTitanicData
# Description :  It does preprocessing
#                It removes unnecessary columns
#                It converts text data to numeric format
#                It does encoding to categorical columns
# Parameters :   df -> Pandas Dataframe     
# Return :       df -> clean pandas dataframe
# Date :         14/03/2026
# Author :       Haritalika Parasram Munde
#-------------------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original data")
    print(df.head())

    # remove unnecessary columns
    drop_columns=["Passengerid","zero","Name","Cabin"]
    existing_columns=[col for col in drop_columns if col in df.columns]

    print("\nColumns to be droped : ")
    print(existing_columns)

    # drop the unwanted columns
    df=df.drop(columns=existing_columns)
    DisplayInfo("Step 3 : Data after colums removal")
    print(df.head())

    # Handle age column
    if "Age" in df.columns:
        print("Age column before filling missing values")
        print(df["Age"].head(10))

        # coerce -> invalid value gets converted as NaN
        df["Age"]=pd.to_numeric(df["Age"],errors="coerce")

        age_meadian=df["Age"].median()

        #Replace missing values with median
        df["Age"]=df["Age"].fillna(age_meadian)

        print("\nAge column after preprocessing")
        print(df["Age"].head(10))

    # Handle fare column
    if "Fare" in df.columns:
        print("\nFare column before preprocessing")
        print(df["Fare"].head(10))

        df["Fare"]=pd.to_numeric(df["Fare"],errors="coerce")
        fare_meadian=df["Fare"].median()

        #Replace missing values with median
        df["Fare"]=df["Fare"].fillna(fare_meadian)

        print("\n Median of fare column : ",fare_meadian)

        print("\nFare column after preprocessing")
        print(df["Fare"].head(10))

    # Handle Embarked column

    if "Embarked" in df.columns:
        print("\n column before preprocessing")
        print(df["Embarked"].head(10))

        # convert the data into string
        df["Embarked"]=df["Embarked"].astype(str).str.strip()

        # Remove missing values
        df["Embarked"]=df["Embarked"].replace(['nan','None',''],np.nan)

        # get most frquent value
        embarked_mode=df["Embarked"].mode()[0]
        print("Mode of embarked column : ", embarked_mode)

        df["Embarked"]=df["Embarked"].fillna(embarked_mode)

        print("\nEmbarked column after preprocessing")
        print(df["Embarked"].head(10))

    # handle sex column
    if "Sex" in df.columns:
        print("\nFare column before preprocessing")
        print(df["Sex"].head(10))

        df["Sex"]=pd.to_numeric(df["Sex"],errors="coerce")

        print("\nEmbarked column after preprocessing")
        print(df["Sex"].head(10))

    DisplayInfo("Data after preprocessing")
    print(df.head())

    print("\nMissing values after preprocessing")
    print(df.isnull().sum())

    #Encode embark column
    df=pd.get_dummies(df,columns=["Embarked"],drop_first=True)
    print("\nData after encoding")

    print(df.head())
    print("Shape of dataset : ",df.shape)

    # convert boolean columns into integer
    for col in df.columns:
        if df[col].dtype==bool:
            df[col]=df[col].astype(int)

    print("\nData after encoding")

    print(df.head())
    

    return df

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

    df=CleanTitanicData(df)

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