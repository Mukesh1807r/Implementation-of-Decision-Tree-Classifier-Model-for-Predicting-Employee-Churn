# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: Mukesh R

RegisterNumber: 212224240098
```python
import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:
![image](https://github.com/user-attachments/assets/c4643aaa-2ea5-4c14-a1c9-81f60daa4fa3)

![image](https://github.com/user-attachments/assets/cfcd9fd9-bdf4-4609-b02c-688c8c3f6120)

![image](https://github.com/user-attachments/assets/a6644839-fb4c-469d-bd40-7afbb2a6b6d2)

![image](https://github.com/user-attachments/assets/f7ee8c3d-580b-4de5-891d-0ea743b7daa6)

![image](https://github.com/user-attachments/assets/ec966879-9f98-41f4-b827-c9f74c87388e)

![image](https://github.com/user-attachments/assets/100aa6cd-31d3-4754-a651-7cdcea9d955f)

![image](https://github.com/user-attachments/assets/ef7fe646-5d9c-4a62-a957-f4e09070438e)

![image](https://github.com/user-attachments/assets/ad57f519-a03f-4870-b591-34c47cdd7da0)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
