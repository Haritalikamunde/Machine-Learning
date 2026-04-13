import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import ( 
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Border="-"*40
####################################################################################
# Step 1 : Load the dataset
####################################################################################
print(Border)
print("Step 1: Load the dataset")
print(Border)

DatasetPath="iris.csv"

df = pd.read_csv(DatasetPath)

print("Dataset gets loaded successfully....")
print("Initial entries from dataset :")
print(df.head())

####################################################################################
# Step 2 : Data Analysis (EDA)
####################################################################################
print(Border)
print("Step 2: Data Analysis")
print(Border)

print("shape of dataset : ", df.shape)
print("Column names : ", list(df.columns))

print("Missing values (Per column)")
print(df.isnull().sum())

print("class Distribution ( species count)")
print(df["species"].value_counts())

print("Statistical report of dataset")
print(df.describe())

####################################################################################
# Step 3 : Decide Independent and dependent variables
####################################################################################
print(Border)
print("Step 3: Decide Independent and dependent variables")
print(Border)

# X : Independent variables/ features
# Y : Deependent variables/ labels

feature_cols=[
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

X=df[feature_cols]
Y=df["species"]

print("X shape :", X.shape)
print("Y shape :",Y.shape)

####################################################################################
# Step 4 : Visualization of data set
####################################################################################
print(Border)
print("Step 2: Visualization of dataset")
print(Border)

#Scatter plot
plt.figure(figsize=(7,5))

for sp in df["species"].unique():
    temp= df[df["species"]==sp]
    plt.scatter(temp["petal length (cm)"],temp["petal width (cm)"], label=sp)

plt.title("Iris: Petal length vs Petal width")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.legend()
plt.grid(True)
plt.show()


####################################################################################
# Step 5 : split the dataset for the training and testing
####################################################################################
print(Border)
print("Step 5: split the dataset for the training and testing")
print(Border)

# Test size= 20%
# Train size =80%

X_train, X_test, Y_train, Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("X -Independent:", X.shape)              #(150,4)
print("Y -Dependent:",Y.shape)                 #(150)
print("data spliting activity done :")
print("X_train: ", X_train.shape)               #(120,4)
print("X_test: ", X_test.shape)                 #(30,4)
print("Y_train: ", Y_train.shape)               #(120)
print("Y_test: ", Y_test.shape)                 #(30,)
