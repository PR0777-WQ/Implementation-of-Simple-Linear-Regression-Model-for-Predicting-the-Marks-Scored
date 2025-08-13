# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
# G.Nitin Karthikeyan
#roll number:212224040227
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SAIPRASATH.P
RegisterNumber: 212224230238
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```

## Output:
<img width="558" height="619" alt="image" src="https://github.com/user-attachments/assets/385d10d4-836f-4987-8adb-6cca5cf80c4f" />

<img width="598" height="501" alt="image" src="https://github.com/user-attachments/assets/52ff031c-01ba-48a4-b896-d2933807f290" />

<img width="256" height="402" alt="image" src="https://github.com/user-attachments/assets/a496425e-3302-4594-8178-6a0362894515" />

<img width="283" height="142" alt="image" src="https://github.com/user-attachments/assets/69267f8a-a7a4-4847-a2a8-cdca5d709116" />

<img width="212" height="211" alt="image" src="https://github.com/user-attachments/assets/ece1e5e8-8fa5-4eff-b924-1d76f58be6a6" />

<img width="1227" height="88" alt="image" src="https://github.com/user-attachments/assets/67ca4c60-5b6e-489f-8243-26c12b7a31fb" />

<img width="788" height="649" alt="image" src="https://github.com/user-attachments/assets/ea3e7da7-9c54-4ff2-8c8d-0464af059cc1" />

<img width="288" height="223" alt="image" src="https://github.com/user-attachments/assets/1ba8bdd1-ea8a-4101-98b1-fead7b5c8f45" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
