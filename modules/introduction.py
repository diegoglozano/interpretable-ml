import pandas as pd
import ppscore as pps
import streamlit as st
import plotly.express as px

from utils.helpers import (plot_binary_feature,
                           plot_feature_distribution,
                           define_lr_pipeline,
                           get_cross_val_score)

from utils.constants import (TARGET_COL,
                             N_JOBS,
                             N_SPLITS,
                             RANDOM_STATE)


def introduction(df: pd.DataFrame):

    st.write(f"""
        For this problem we are going to use churn data, downloaded from Kaggle:
    """)
    st.dataframe(df.head())

    st.write(f"""
        It consists on {df.shape[0]} rows and {df.shape[1]} columns
    """)

    st.write(f"""
        # Analysis
        Let's analyze each column:
    """)

    st.write(f"""
        ### _customerID_

        This column seems to be an identifier. How many unique values does it have?  
        `{df['customerID'].nunique()}`  
        Same as number of rows! It's an identifier column. We should drop it for modelling  
    """)

    binary_cols = [col for col in df.columns
                   if (df[col].nunique() == 2)
                   and (col != 'Churn')]

    other_cols = list(set(df.columns).difference(set(binary_cols)))
    other_cols = [col for col in other_cols if (col != 'Churn') and (col != 'customerID')]

    [plot_binary_feature(df, col) for col in binary_cols]

    [plot_feature_distribution(df, col) for col in other_cols]

    st.write(f"""
        ### Null values
    """)
    st.write(df.isnull().any().any())

    st.write(f"""
        Predictive Power Score
    """)

    pps_churn = pd.concat([
        pd.Series(pps.score(df, col, 'Churn').get('ppscore'), name=col)
        for col in df.columns if col != 'Churn'
    ], axis=1).T.reset_index().rename(columns={'index': 'Feature', 0: 'Score'}).sort_values(by='Score')

    st.dataframe(pps_churn)
    st.plotly_chart(
        px.bar(pps_churn, x='Feature', y='Score')
    )

    pps_matrix = pps.matrix(df)
    st.plotly_chart(
        px.imshow(pps_matrix,
                  x=pps_matrix.columns.tolist(),
                  y=pps_matrix.columns.tolist())
    )

    st.write(f"""
        # Model
    """)

    st.write(f"""
        ## Logistic Regression
    """)

    metrics = ['accuracy', 'f1']
    df.drop(columns='customerID', inplace=True)
    df['Churn'] = df['Churn'].replace({'No': 0,
                                       'Yes': 1})

    lr_pipeline = define_lr_pipeline(df, TARGET_COL, N_JOBS, RANDOM_STATE)
    results_lr_pipeline = get_cross_val_score(lr_pipeline,
                                              df, TARGET_COL,
                                              N_SPLITS, metrics,
                                              N_JOBS, RANDOM_STATE)
    st.write(f"""
        `{results_lr_pipeline["test_accuracy"].mean().round(2)}`
         ± `{results_lr_pipeline["test_accuracy"].std().round(2)}`
    """)
    st.write(f"""
        `{results_lr_pipeline["test_f1"].mean().round(2)}`
         ± `{results_lr_pipeline["test_f1"].std().round(2)}`
    """)

    st.write(f"""
        ## Decision tree
    """)

    st.write(f"""
        ## Boosting trees
    """)
