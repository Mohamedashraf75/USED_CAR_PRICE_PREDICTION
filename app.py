import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OrdinalEncoder , PolynomialFeatures ,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression , Ridge , Lasso , ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVR

st.set_page_config(layout="wide" , page_title="Used Car APP")

st.title('USED CAR PRICE PREDICTION')

column_1 , column_2 , column_3 = st.columns([70,5,70])
with column_1:
    Engine=st.radio('Engine? ',["1600","1500","1300"])
    Brand=st.radio('Brand? ',['Hyundai', 'Chevrolet', 'Fiat'])
    Body=st.radio('Body? ',['Sedan', 'Hatchback', 'SUV'])
    Fuel=st.radio('Fuel? ',['Benzine', 'Natural Gas'])  
    Transmission=st.radio('Transmission? ',['Automatic', 'Manual'])





    
with column_3:
    Kilometers =st.slider('Kilometers? ',0,400000,200000)
    Age_of_Car=st.slider('Age_of_Car? ',0,50,25)
    Model=st.selectbox('Model? ',['Accent', 'Avante', 'I10', 'Elantra', 'Excel', 'Matrix', 'Tucson', 'Verna', 'Cruze', 'Aveo', 'Lanos', 'Optra', '128', 'Punto', 'Shahin', 'Tipo', 'Uno'] )
    Color=st.selectbox('Color? ',['Black', 'Silver', 'Gray', 'Blue- Navy Blue', 'Green', 'Red', 'Gold', 'Other Color', 'Burgundy', 'White', 'Yellow', 'Brown', 'Orange', 'Beige'] )
    Gov=st.selectbox('Gov? ',['Giza', 'Qena', 'Cairo', 'Minya', 'Alexandria', 'Dakahlia', 'Suez', 'Sharqia', 'Kafr al-Sheikh', 'Beheira', 'Ismailia', 'Sohag', 'Monufia', 'Qalyubia', 'Beni Suef', 'Asyut', 'Fayoum', 'Gharbia', 'Matruh', 'Damietta', 'Red Sea', 'Port Said', 'Luxor', 'South Sinai', 'New Valley', 'Aswan'])


    
New_Date = pd.DataFrame({'Engine':[Engine],
                         'Brand':[Brand],
                         'Body':[Body],
                         'Fuel':[Fuel],
                         'Transmission':[Transmission],
                         'Kilometers':[Kilometers],
                         'Age_of_Car':[Age_of_Car],
                         'Model':[Model],
                         'Color':[Color],
                         'Gov':[Gov]},index=[0])


transformer=joblib.load('column_Transformer.h5')
model=joblib.load('XGBRegressor.h5')

Preprocess = transformer.transform(New_Date)
Predict = model.predict(Preprocess)

st.dataframe(New_Date,width=1200,height=10,use_container_width=True)

if st.button('Predict'):
    st.subheader(round(Predict[0],2))
