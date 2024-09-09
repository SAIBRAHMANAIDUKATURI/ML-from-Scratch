import pandas as pd
import numpy as np

class LinearRegression:
    def __init__(self):
        self.W = None
    def fit(self,X,y):
        X_b = np.c_[np.ones((X.shape[0],1)),X]
        self.W = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    def predict(self,X):
        X_b = np.c_[np.ones((X.shape[0],1)),X]
        return X_b.dot(self.W)
def main():
    data = {
        "YearsofExp":[1,2,3,4,5,6,7],
        "salary":[100,200,300,400,500,600,700]
    }
    df = pd.DataFrame(data)
    X_train = df.iloc[:5,:-1].values
    y_train = df.iloc[:5,-1].values
    X_test = df.iloc[5:,:-1].values
    y_test = df.iloc[5:,-1].values
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    for i,j in zip(y_pred,y_test):
        print(f"predicted value is {i} and  actual values is {j}")

if __name__ =="__main__":
    main()