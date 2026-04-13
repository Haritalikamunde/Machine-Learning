from sklearn.metrics import confusion_matrix

def main():

    # 1: positive
    # 2: negative
    Actual=    [1,0,1,1,1,0,1,0,0,1]
    Predicted= [1,0,0,1,1,1,1,1,0,1]

    print("Actual Data :", Actual)
    print("Predicted data: ", Predicted)

    con_mat=confusion_matrix(Actual,Predicted)

    print(con_mat)
    
if __name__=="__main__":
    main()