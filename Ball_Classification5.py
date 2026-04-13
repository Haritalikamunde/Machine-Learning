from sklearn import tree

# Rough = 1
# Smooth = 0

# Tennis = 1
# Cricket = 2


def main():
    print("Ball classification case study")

    #original Encoded dataset
    # Independent variables
    X =[[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]

    # Dependent variables
    Y =[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    # Independent variables for training
    Xtrain=[[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0]]

    # Independent variables for testing
    Xtest=[[35,1],[95,0]]

    # dependent variables for training
    Ytrain=[1,1,2,1,2,1,2,1,1,1,2,1,2]

    # dependent variables for testing
    Ytest=[1,2]
    modelobj=tree.DecisionTreeClassifier()

    trainedmodel=modelobj.fit(Xtrain,Ytrain)

    Result= trainedmodel.predict(Xtest)           #[ 1, 2 ]

    print("Model predicts the object as :",Result)

if __name__=="__main__":
    main()
