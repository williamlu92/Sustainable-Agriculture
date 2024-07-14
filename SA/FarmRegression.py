import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV


metadata = pd.read_table('Sustainable Agriculture/sample_metadata.tsv')
metadata.index = ['farm_%i' % i for i in range(len(metadata))]
metadata['crop_yield'] += 1

bacteria_counts_lognorm = pd.read_csv('Sustainable Agriculture/bacteria_counts_lognorm.csv', index_col=0)
bacteria_counts = pd.read_table('Sustainable Agriculture/bacteria_counts.tsv')

bacteria_counts = bacteria_counts.drop(['Unnamed: 0'], axis=1)
#@title ###Setup notebook.
#@title ###Setup notebook.

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error



# We helped you define your X and y data here.
X = bacteria_counts_lognorm
y = metadata['crop_yield']

# Split your data into testing and training.
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Random Forest Model
Rmodel = RandomForestRegressor(n_estimators=10, ccp_alpha=0.00005)
Rmodel.fit(X_train, y_train)
Rpreds = Rmodel.predict(X_test)


#MLP Regressor Model

st.title('AI Model for Sustainable Agriculture: Locate Fertile Soil to Secure Food Supply') 
st.write("By William Lu, Material Provided by Inspirit AI ")
st.image('SA/IndianFarm.png')
st.write("As global temperatures rise and weather patterns shift, the amount of fertile land is diminishing. Meanwhile, the world is projected to have 9 billion people by 2050. How can we make sure we are able to nourish 9 billion people while also nourishing our planet? Managing and developing agricultural lands can help use reach greater crop yields while using less land, energy, and harmful chemicals.")
st.write("Scientists have discovered that the soil microbiome, the collection of bacteria that live in a region of soil play an important role in the health of plants! Therefore, maybe we can predict how well plants will grow in a region based on the bacterial composition of the soil.")


st.subheader("DataSet")
st.dataframe(metadata.head(5))
st.write("Farm Crop-Yield Dataset from Australia")
st.dataframe(bacteria_counts_lognorm.columns[20::20].head())
st.write("Table: Bacteria Counts Log Transformed")
st.write("Through data collected on farms and their bacterial composition, we can train machine learning models to accurately predict crop yields. This can help farmers and institutions discover fertile soil, better managing resources and make more informed decisions.")


st.header("Machine Learning Models")
col1, col2 = st.columns(2)
with col1: 
    st.subheader("Random Forest Model")
    st.image('SA/RandomForest.png')
    st.write("A Random forest regression model combines multiple decision trees to create a single model. Each tree in the forest builds from a different subset of the data and makes its own independent prediction. The final prediction for input is based on the average or weighted average of all the individual trees' predictions.")
    
with col2: 
    st.subheader("Multilayer Perceptron Neural Network")
    st.image('SA/MLP.png')
    st.write("An MLP (Multilayer Perceptron) regressor is a type of artificial neural network used for predicting continuous values. It consists of layers of interconnected nodes, or neurons, where each layer processes the input data and passes it to the next layer. The network learns to make accurate predictions by adjusting the connections (weights) between neurons based on the error of its predictions during training. By using multiple layers and nonlinear activation functions, an MLP can model complex relationships in the data, making it useful for tasks like predicting housing prices or crop yields.")
st.image("SA/Performance.png")

col3, col4, col5= st.columns(2)

with col3: 
    st.image("SA/Graph")
with col4:
    st.image("SA/Graph2")
with col5:
    st.image("SA/Histogram")

st.header("Model Demo")
st.subheader("Select a Farm and its sampled bacteria profile below")
b = st.slider('Slide me', min_value=0, max_value=10)
X_sample = bacteria_counts_lognorm.iloc[b]

# If you want to convert it into a DataFrame, you can use .to_frame() or .transpose() methods
X_sample = X_sample.to_frame().transpose()
st.dataframe(X_sample)

y_sample = Rmodel.predict(X_sample)


st.write(f"This field can produce a yield of {y_sample[0] * 1000} kilograms/hectare")
st.write(f"农场预计产出大麦 {y_sample[0] * 2000} 斤/公顷")



