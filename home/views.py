from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from django.http import HttpResponse

# Create your views here.

def home(request):
    return render(request,"index.html")

def result(request):

    db = pd.read_csv(r'E:\Computer_Science\Mechine_Learning\projects\Diabetes_predection\Data\diabetes.csv')
    db= db.drop('Age', axis=1)
    db=db.drop("DiabetesPedigreeFunction", axis=1)
    X = db.iloc[:, :-1]
    y=db.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])

    pred = reg.predict([[val1,val2,val3,val4,val5,val6]])

    result2 = ''

    if pred == [1]:
        result2 = "Positive"
    else :
        result2 = "Negative"

    return render(request,"index.html",{"result2":result2})
