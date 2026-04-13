import pandas as pd


def main():
    Data={
        "Name" : ["Sagar","Amit","Pooja"],
        "Age" : [23,26,25],
        "City": ["pune","Mumbai","Satara"]
    }

    dobj=pd.DataFrame(Data)

    # Fetch specific row 
    print(dobj.loc[0])

if __name__=="__main__":
    main()