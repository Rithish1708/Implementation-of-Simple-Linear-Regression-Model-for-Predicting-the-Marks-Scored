# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import necessary libraries 

2.Load dataset 

3.Define input (independent variable) and output (dependent variable)

4.Split dataset into training and testing sets 

5.Build Linear Regression model

6.Predict test set results 

7.Visualize Training set results 

8.Visualize Testing set results 

9.Evaluate model performance

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rithish R
RegisterNumber: 212224040278
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head()
df.tail()
print(df.head())
print(df.tail())
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="brown")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae )
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:

TO Read Head and Tail Files

<img width="302" height="249" alt="image" src="https://github.com/user-attachments/assets/bda7cadd-e3eb-411e-891c-d508e0947d38" />




Compare Dataset

<img width="790" height="568" alt="image" src="https://github.com/user-attachments/assets/3e2f0a79-c92b-4da2-b90e-8940cac680bf" />




Predicted Values

<img width="799" height="75" alt="image" src="https://github.com/user-attachments/assets/ff97249e-f6f9-40ee-9451-431da13955c0" />



Graph for Training Set

<img width="1000" height="574" alt="image" src="https://github.com/user-attachments/assets/e6624d5c-02ee-45e0-a45c-ee2d171780b3" />



Graph for Testing Set

<img width="873" height="562" alt="image" src="https://github.com/user-attachments/assets/d1a8f703-fbb4-4333-adfb-bc69e8796253" />

<img width="271" height="82" alt="image" src="https://github.com/user-attachments/assets/751641b8-f5e5-4a42-8d3e-68cbe7dadd3d" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
