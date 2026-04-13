from sklearn.datasets import load_iris

def main():
    print("Iris classification case study")

    Dataset= load_iris()

    border="-"*40

    print(border)

    for i in range(len(Dataset.target)):
        print("ID %d, Features %s, Label %s" %(i, Dataset.data[i],Dataset.target[i]))

if __name__=="__main__":
    main()
