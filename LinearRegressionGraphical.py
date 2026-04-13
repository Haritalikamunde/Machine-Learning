import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def MarvellousPredictor():
    #Load the data
    X=[1,2,3,4,5]
    Y=[3,4,2,4,5]

    print("Values of independent variables : X -",X)
    print("Values of dependent variables :Y- ", Y)

    mean_x=np.mean(X)                     #3.0
    mean_Y=np.mean(Y)                     #3.6

    print("X_mean is :", mean_x)
    print("Y_mean is :", mean_Y)

    n=len(X)       #5

    # Y= mX+C

    # m= (sum(x-x_bar)*(y-y_bar))/(Summ(X-X_bar)**2)

    numerator = 0
    denominator = 0

    for i in range(n):
        numerator=numerator +((X[i]-mean_x) *(Y[i]-mean_Y))
        denominator=denominator +((X[i]-mean_x)**2)

    m= numerator/ denominator

    print("Slope of line i.e. m:", m)

    c=mean_Y-(m*mean_x)

    print("Y intercept of line ie. c :",c)
    
    x=np.linspace(1,6,n)
    y=c+m*x

    plt.plot(x,y,color='g',label="Regression Line")
    plt.scatter(X,Y,color='r', label="Scatter plot")

    plt.xlabel("X : Indepenedent variables")
    plt.ylabel("Y : dependent variable")

    plt.legend()
    plt.show()

def main():
    MarvellousPredictor()


if __name__=="__main__":
    main()


# logic for Yp
# logic for R^2
