from sklearn.datasets import load_iris

def main():
    print("Iris classification case study")

    Dataset= load_iris()

    #metadeta of dataset
    print("Independent variables are :")
    print(Dataset.feature_names)

    print("length of independent dataset is :",len(Dataset.feature_names))

    print("dependent variables are :")
    print(Dataset.target_names)

    print("length of dependent dataset is :",len(Dataset.target_names))

    

if __name__=="__main__":
    main()
