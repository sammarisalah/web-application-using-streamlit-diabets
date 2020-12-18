# IMPORT LIBRARY
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# Create a title and sub-title

st.write("""
# Diabet detection""")

#open image

image = Image.open("C:/Users/ASUS/Desktop/big project/project 1 detectation des maladie/diabet/d.jpg")
st.image(image,caption='ML',use_column_width=True)

#get data
df = pd.read_csv("C:/Users/ASUS/Desktop/big project/project 1 detectation des maladie/diabet/diabetes.csv")

#set a subbheader

st.subheader('data information')

# show data as a table

st.dataframe(df)

#show static on the data

st.write(df.describe())

# show data as chart

chart = st.bar_chart(df)


#split data

X = df.iloc[:, 0:8].values
y = df.iloc[:,-1].values

#split data into 75% training and 25% test
x_train , x_test , y_train , y_test = train_test_split(X , y ,test_size=0.25,random_state=0)

# get the feature imput from the users
def get_user_input():
    pregnacies = st.sidebar.slider('pregnacies',0,17,3) # first = min / second = max / third = the cursor in page web
    Glucose = st.sidebar.slider('Glucose',0,199,117)
    BloodPressure = st.sidebar.slider('BloodPressure',0,122,72)
    SkinThickness = st.sidebar.slider('SkinThickness',0,99,23)
    Insulin = st.sidebar.slider('Insulin',0.0,846.0,30.0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.3725)
    Age = st.sidebar.slider('Age',21,81,29)

    # store a dictionarry into a variable
    user_data = {'pregnacies':pregnacies,
             'Glucose':Glucose,
             'BloodPressure':BloodPressure,
             'SkinThickness':SkinThickness,
             'Insulin':Insulin,
             'BMI':BMI,
             'DPF':DPF,
             'Age':Age
             }
#transform data in to data dataframe
    features = pd.DataFrame(user_data,index=[0])
    return features

#store the user input into a data frame
user_input = get_user_input()

 #set a subheader and display the user_input
st.subheader('user_input:')
st.write(user_input)

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)

#show the model metrics
st.subheader('accuracy_score:')
st.write( str(accuracy_score(y_test,RandomForestClassifier.predict(x_test))*100)+'%')

#store the models prediction in a variable

prediction = RandomForestClassifier.predict(user_input)

# set a subheader and display the classification
st.subheader('resultat')
st.write(prediction)
