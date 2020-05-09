import streamlit as st

from utils.paths import DATA_PATH
from utils.helpers import read_data

from modules.introduction import introduction


st.sidebar.title('Interpretable Machine Learning')
st.sidebar.subheader('Choose which type of interpretability')

choice = st.sidebar.radio('',
                          ('Introduction', 'GLM Coefficients',
                           'Feature importances', 'SHAP',
                           'Surrogate Decision Tree'))

df = read_data(DATA_PATH / 'churn_data.csv')

if choice == 'Introduction':
    introduction(df)

