import streamlit as st
import pandas as pd
import os


# ompirt profiling ability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


# ML
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://img.freepik.com/free-vector/ai-powered-marketing-tools-abstract-concept-illustration_335657-3796.jpg?w=826&t=st=1692554166~exp=1692554766~hmac=31ae5e233fd9d6460c739ff819bcd8814e12670623e706fa30bd158d0fce7446")
    st.title("Auto StreamML")
    choice = st.radio("Navigation",
                      ["Upload Dataset", "Profiling", "Module Learning", "Download Model"])
    st.info("This application allows you to build automated ML pipeline using Streamlit, Pandas, Profiling and Pycarrot")


if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)


if choice == "Upload Dataset":
    st.title("Upload your data for modelling")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)


if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "Module Learning":
    st.title("Machine Learning Section")
    target = st.selectbox("Select Your target", df.columns)
    if st.button("Train model"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("ML experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')


if choice == "Download Model":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")
