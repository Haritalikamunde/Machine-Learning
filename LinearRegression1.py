import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def MarvellousPredictor():
    #Load the data
    X=[1,2,3,4,5]
    Y=[3,4,2,4,5]

    print("Values of independent variables : X -",X)
    print("Values of dependent variables :Y- ", Y)

    mean_x=np.mean(X)
    mean_Y=np.mean(Y)

    print("X_mean is :", mean_x)
    print("Y_mean is :", mean_Y)

def main():
    MarvellousPredictor()


if __name__=="__main__":
    main()