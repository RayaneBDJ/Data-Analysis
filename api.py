import streamlit as st
from PIL import Image

# requirements 

from mapping import *


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import seaborn as sns 
import plotly_express as px

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelBinarizer
from sklearn.compose import make_column_transformer

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


from IPython.display import display, HTML 

css = """
.output {
    flex-direction: row;
}
"""

HTML('<style>{}</style>'.format(css))



data_url = ("data_to_API.csv")

st.markdown("# We will explore the  dataset \"drugs  consumers\" to predict if a young consumer is taking drugs currently or will use it eventually" )


st.markdown("Our dataset features this drugs which are estimated to be the most dangerous :")


st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Alcohol", unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Amphétamine", unsafe_allow_html=True)      
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Benzodiazepine",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Cannabis ",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Cocaine   ",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Crack ",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Ectasy ",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Heroin",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Ketamine ",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; LSD ",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Methadone ",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Magic Mushrooms ",unsafe_allow_html=True)
st.markdown("&nbsp;&nbsp;&nbsp;&#x2022; Nicotine ",unsafe_allow_html=True)

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)


    

    
 
 
    
#code X_train, Y train etc 

X_train = pd.read_csv('Data-analysis/X_train.csv')
Y_train = pd.read_csv('Data-Analysis/Y_train.csv')
X_test = pd.read_csv('Data-analysis/X_test.csv')
Y_test = pd.read_csv('Data-analysis/Y_test.csv')

X_train.set_index('ID',inplace = True)
Y_train.set_index('ID',inplace = True)
X_test.set_index('ID',inplace = True)
Y_test.set_index('ID',inplace = True)


countries = ['Country_Australia', 'Country_Canada', 'Country_Other', 'Country_UK',
       'Country_USA']
ethnicities = ['Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Other',
       'Ethnicity_White', 'Ethnicity_White-Asian', 'Ethnicity_White-Black']


# we can add a new row 

#predict your frequence of consumption  of alcohol

st.markdown('#### You can now enter the informations of a young consumer you know or your own informations')

dict_new_row = {}

list_values = []

age = st.number_input('Insert your age ',min_value = 0,step = 1)
list_values.append(age)

gender = st.selectbox('What gender are you',('Male','Female'))
if(gender == "male "):
        list_values.append(0)
else :
        list_values.append(1)

Stress_score = st.number_input('Rate your stress (%)',min_value = 0,max_value = 100)
list_values.append(Stress_score)

csore = st.number_input("Rate how rigorous you are (%)",min_value = 0,max_value = 100)
list_values.append(csore)

communicative_score = st.number_input("Rate how communicative you are (%)",min_value = 0,max_value = 100)
list_values.append(communicative_score)

impulsive_score = st.number_input("Rate how impulsive you are (%)",min_value = 0,max_value = 100)
list_values.append(impulsive_score)


list_values.extend([0]*(len(countries) + len(ethnicities)))


country = st.selectbox('Select you country',set(countries))
ethnicity = st.selectbox('Select you ethnicity',set(ethnicities))



dict_new_row = dict(zip(list(X_test.columns),list_values))
dict_new_row[country] = 1
dict_new_row[ethnicity] = 1

st.write('Your specifications : ',dict_new_row)









st.markdown('#### You will now have to choose the model you want to use to predict the frequence of consumptions of a consumer you know')

# accuracy of models
# SVM which is supposed to be the most accurate

scores_SVM = []
svc_fit = []

for col in Y_train.columns :
    svc=SVC() 
    svc_fit.append(svc.fit(X_train, Y_train.loc[:,col]))
    y_pred=svc.predict(X_test)
    scores_SVM.append(accuracy_score(Y_test.loc[:,col],y_pred))
  
# Logic Regression 

scores_LR = []
lr_fit = []

for col in Y_train.columns :
    lr=LogisticRegression() 
    lr_fit.append(lr.fit(X_train, Y_train.loc[:,col]))
    scores_LR.append(lr.score(X_test,Y_test.loc[:,col]))

# KNN  classification
# Knn with sklearn
from sklearn.neighbors import KNeighborsClassifier

scores_KNN = []
knn_fit = []

for col in Y_train.columns :
    knn=KNeighborsClassifier(n_neighbors = 10) 
    knn_fit.append(knn.fit(X_train, Y_train.loc[:,col]))
    scores_KNN.append(knn.score(X_test,Y_test.loc[:,col]))
  

from sklearn.naive_bayes import GaussianNB

scores_NB = []
nb_fit = []

for col in Y_train.columns :
    nb=GaussianNB()
    nb_fit.append(nb.fit(X_train, Y_train.loc[:,col]))
    scores_NB.append(nb.score(X_test,Y_test.loc[:,col]))
   

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

scores_DT = []
dt_fit = []

for col in Y_train.columns :
    dt=DecisionTreeClassifier()
    dt_fit.append(dt.fit(X_train, Y_train.loc[:,col]))
    scores_DT.append(dt.score(X_test,Y_test.loc[:,col]))
   
# Random Forest Classification

scores_RF = []
RF_fit = []

for col in Y_train.columns :
    rf=RandomForestClassifier(n_estimators=100,random_state=1)
    RF_fit.append(rf.fit(X_train, Y_train.loc[:,col]))
    scores_RF.append(rf.score(X_test,Y_test.loc[:,col]))


#plot 
st.write('You can see here the accuracy of each model')


fig = plt.figure()

plt.plot(list(Y_train.columns), scores_LR, label = "logistic Regression", c = "red")
plt.plot(list(Y_train.columns), scores_SVM, label = "SVM", c = "blue")
plt.plot(list(Y_train.columns), scores_KNN, label = "KNN with 10 neighbours", c = "orange")
plt.plot(list(Y_train.columns), scores_DT, label = "Decision Tree ", c = "green")
plt.plot(list(Y_train.columns), scores_RF, label = "Random Forest with 100 forests", c = "purple")
plt.plot(list(Y_train.columns), scores_NB, label = "Naive Baye", c = "brown")

plt.xticks(list(Y_train.columns),rotation=25)

st.plotly_chart(fig,use_container_width = True)



models = ['Support Vector Machine','Logistic Regression','Naive Baye','Random Forest']

model = st.selectbox('Select the model you want to use to guess and predict',set(models))


# prediction 


new_row = pd.pivot_table(pd.DataFrame(pd.Series(dict_new_row)),columns = X_test.columns )
new_row = new_row[X_test.columns]
st.write(new_row)

if (model == 'Random Forest') :

    for col in Y_train.columns :
        rf=RandomForestClassifier(n_estimators=100,random_state=1)
        rf.fit(X_train, Y_train.loc[:,col])
        if(rf.predict(new_row)[0] == 0) :
            st.write(col, ' : Never Used it in your life ',rf.predict(new_row)[0])
        elif(rf.predict(new_row)[0]  == 1) :
            st.write(col, ' : You used it before this year ',rf.predict(new_row)[0])
        elif(rf.predict(new_row)[0]  == 2) :
            st.write(col, ' : You used it this year and you may use it again',rf.predict(new_row)[0])

elif (model == 'Logistic Regression') :
    
    for col in Y_train.columns :
        lg=LogisticRegression()
        lg.fit(X_train, Y_train.loc[:,col])
        if(lg.predict(new_row)[0] == 0) :
            st.write(col, ' : Never Used it in your life ',lg.predict(new_row)[0])
        elif(lg.predict(new_row)[0]  == 1) :
            st.write(col, ' : You used it before this year ',lg.predict(new_row)[0])
        elif(lg.predict(new_row)[0]  == 2) :
            st.write(col, ' : You used it this year and you may use it again',lg.predict(new_row)[0])

elif (model == 'Support Vector Machine') :
    
    for col in Y_train.columns :
        svc=SVC()
        svc.fit(X_train, Y_train.loc[:,col])
        if(svc.predict(new_row)[0] == 0) :
            st.write(col, ' : Never Used it in your life ',svc.predict(new_row)[0])
        elif(svc.predict(new_row)[0]  == 1) :
            st.write(col, ' : You used it before this year ',svc.predict(new_row)[0])
        elif(svc.predict(new_row)[0]  == 2) :
            st.write(col, ' : You used it this year and you may use it again',svc.predict(new_row)[0])

elif (model == 'Naive Baye') :
    
    for col in Y_train.columns :
        nb=GaussianNB()
        nb.fit(X_train, Y_train.loc[:,col])
        if(nb.predict(new_row)[0] == 0) :
            st.write(col, ' : Never Used it in your life ',nb.predict(new_row)[0])
        elif(nb.predict(new_row)[0]  == 1) :
            st.write(col, ' : You used it before this year ',nb.predict(new_row)[0])
        elif(nb.predict(new_row)[0]  == 2) :
            st.write(col, ' : You used it this year and you may use it again',nb.predict(new_row)[0])



