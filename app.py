import pandas as pd
import numpy as np
import seaborn 
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv(r'https://raw.githubusercontent.com/rishank-shah/Diabetes-Prediction/main/diabetes.csv')

st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')

dataset["Glucose"].fillna(dataset["Glucose"].mean(), inplace = True)
dataset["BloodPressure"].fillna(dataset["BloodPressure"].mean(), inplace = True)
dataset["SkinThickness"].fillna(dataset["SkinThickness"].mean(), inplace = True)
dataset["Insulin"].fillna(dataset["Insulin"].mean(), inplace = True)
dataset["BMI"].fillna(dataset["BMI"].mean(), inplace = True)

continuous_variables = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

for attr in continuous_variables:
    Q1 = dataset[attr].quantile(0.25)
    Q3 = dataset[attr].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR
    dataset = dataset[(dataset[attr]>lower_limit)&(dataset[attr]<upper_limit)]

df_majority = dataset[(dataset["Outcome"]==0)]
df_minority = dataset[(dataset["Outcome"]==1)]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=500,random_state=42)
dataset = pd.concat([df_minority_upsampled,df_majority])

scaler = MinMaxScaler()
scaler.fit_transform(dataset)


X = dataset.drop(['Outcome'], axis = 1)
Y = dataset["Outcome"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)


def user_report():
  Pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  Glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  BloodPressure = st.sidebar.slider('BloodPressure', 0,122, 70 )
  SkinThickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  Insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  BMI = st.sidebar.slider('BMI', 0,67, 20 )
  DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  Age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'Pregnancies':Pregnancies,
      'Glucose':Glucose,
      'BloodPressure':BloodPressure,
      'SkinThickness':SkinThickness,
      'Insulin':Insulin,
      'BMI':BMI,
      'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
      'Age':Age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

model = RandomForestClassifier(n_estimators= 100, random_state=0)
model.fit(X_train,Y_train)
Y_pred_random_forest = model.predict(X_test)


user_data = user_report()

user_result = model.predict(user_data)

st.header('Your Report: ')
output = ''
color = ''
if user_result[0]==0:
  output = 'You are not Diabetic'
  color = 'blue'
else:
  output = 'You are Diabetic'
  color = 'red'

st.subheader(output)
st.subheader('Accuracy: ')
st.write(str(model.score(X_test,Y_test) * 100)+'%')


st.title('Visualized Report:')

def plot_graph(attr, palette, xticks, yticks):
    st.header(f'{attr} Graph')
    fig_preg = plt.figure()
    seaborn.scatterplot(x = 'Age', y = attr, data = dataset, hue = 'Outcome', palette = palette)
    seaborn.scatterplot(x = user_data['Age'], y = user_data[attr], s = 150, color = color)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.title('0 - Not Diabetic & 1 - Diabetic')
    st.pyplot(fig_preg)


plot_graph(
   'Pregnancies',
   'Greens',
   xticks = np.arange(10,100,5),
   yticks = np.arange(0,20,2)
)

plot_graph(
   'Glucose',
   'magma',
   xticks = np.arange(10,100,5),
   yticks = np.arange(0,220,10)
)

plot_graph(
   'BloodPressure',
   'Reds',
   xticks = np.arange(10,100,5),
   yticks = np.arange(0,130,10)
)

plot_graph(
   'SkinThickness',
   'Blues',
   xticks = np.arange(10,100,5),
   yticks = np.arange(0,110,10)
)

plot_graph(
   'Insulin',
   'rocket',
   xticks = np.arange(10,100,5),
   yticks = np.arange(0,900,50)
)

plot_graph(
   'BMI',
   'rainbow',
   xticks = np.arange(10,100,5),
   yticks = np.arange(0,70,5)
)

plot_graph(
   'DiabetesPedigreeFunction',
   'YlOrBr',
   xticks = np.arange(10,100,5),
   yticks = np.arange(0,3,0.2)
)
