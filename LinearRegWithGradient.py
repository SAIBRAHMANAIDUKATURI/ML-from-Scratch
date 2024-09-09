import pandas as pd
import numpy as np

class LinearRegression:
    def __init__(self,iterations,learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self,X,y):
        self.m ,self.n = X.shape
        self.W = np.zeros(self.n)
        self.b=0
        self.X = X
        self.y =y
        for i in range(self.iterations):
            self.updateweights()
        return self
    def updateweights(self):
        y_pred = self.predict(self.X)
        dW = -(2*(self.X.T).dot(self.y-y_pred)) /self.m
        db = -2 *(np.sum(self.y -y_pred) /self.m)
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db    
    def predict(self,X):
        return X.dot( self.W ) + self.b 
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
    reg = LinearRegression(iterations =1000, learning_rate=0.01)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    for i,j in zip(y_pred,y_test):
        print(f"predicted value is {i} and  actual values is {j}")

if __name__ =="__main__":
    main()