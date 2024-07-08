import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import plot_tree
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV


metadata = pd.read_table('Sustainable Agriculture/sample_metadata.tsv')
metadata.index = ['farm_%i' % i for i in range(len(metadata))]

bacteria_counts_lognorm = pd.read_csv('Sustainable Agriculture/bacteria_counts_lognorm.csv', index_col=0)
bacteria_counts = pd.read_csv('Sustainable Agriculture/bacteria_counts.tsv')

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


model = RandomForestRegressor(n_estimators=10, ccp_alpha=0.00005)

# Now, initialize your model (just use the default settings for now!)
model.fit(X_train, y_train)

# Make predictions on your test data. (Don't try to compute accuracy just yet...)
preds = model.predict(X_test)
preds_train = model.predict(X_train)


st.title('AI Model for Sustainable Agriculture') 

# Later: add more Streamlit code here
st.header("Locate Fertile Soil and Predicting Crop Yield ")
st.subheader("William Lu 2024, Material Provided by Inspirit AI ")
st.image('SA/Plant.png')
st.write("One way we can improve both people and planet health is through better farming practices. More efficient and effective farming practices can have greater crop yields while using less land, energy, and harmful chemicals. The world is projected to have 9 billion people by 2050... How can we make sure we are able to nourish 9 billion people while also nourishing our planet?")
st.write("Scientists have discovered that the soil microbiome, the collection of bacteria that live in a region of soil play an important role in the health of plants! Therefore, maybe we can predict how well plants will grow in a region based on the bacterial composition of the soil.")



st.subheader("DataSet")
st.dataframe(metadata)
st.dataframe(bacteria_counts)
st.write("Through data collected on farms and their bacterial composition, we can train machine learning models to accurately predict crop yields. This can help farmers and institutions discover fertile soil, better managing resources and make more informed decisions.")


st.header("Machine Learning Models")
st.subheader("Random Forest Model")
st.image('SA/RandomForest.png')
st.write("A Random forest regression model combines multiple decision trees to create a single model. Each tree in the forest builds from a different subset of the data and makes its own independent prediction. The final prediction for input is based on the average or weighted average of all the individual trees' predictions.")
st.image('SA/RandomForestTrained.png')



st.header("Demonstration")
st.subheader("Select a Farm and its sampled bacteria profile below")

b = st.slider('Slide me', min_value=0, max_value=50)

X_sample = bacteria_counts.iloc[b]

# If you want to convert it into a DataFrame, you can use .to_frame() or .transpose() methods
X_sample = X_sample.to_frame().transpose()
st.dataframe(X_sample)

y_sample = model.predict(X_sample)
st.write(f"This farm is expected to produce a yield of {y_sample[0]} tons of Barley/hectare")
st.write(f"农场预计产出大麦 {y_sample[0]} 吨/公顷")
