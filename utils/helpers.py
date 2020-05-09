import pandas as pd
import streamlit as st
import plotly.express as px
import category_encoders as ce

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from os import PathLike
from typing import List
from utils.constants import TARGET_COL


def read_data(path: PathLike,
              nrows=None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def plot_binary_feature(df: pd.DataFrame, col: str):
    st.write(f"""
        ### _{col}_

        We have two possible classes here:
        - {df[col].unique()[0]}
        - {df[col].unique()[1]}
    """)

    st.plotly_chart(
        px.bar((df[col]
                .value_counts()
                .to_frame()
                .reset_index(drop=False)
                .rename(columns={col: f'Count of {col}',
                                 'index': col})),
               x=col,
               y=f'Count of {col}')
    )

    st.plotly_chart(
        px.bar((df.groupby('Churn')[col]
                .value_counts().to_frame()
                .rename(columns={col: 'count'})
                .reset_index().sort_values(by='count')),
               x=col,
               y='count',
               color='Churn')
    )


def plot_feature_distribution(df: pd.DataFrame,
                              col: str):
    st.write(f"""
        ### _{col}_
    """)

    st.plotly_chart(
        px.histogram(df, x=col)
    )


def define_lr_pipeline(df: pd.DataFrame,
                       target_col: str,
                       n_jobs: int,
                       random_state: int) -> Pipeline:
    woe = ce.WOEEncoder()
    sc = StandardScaler()

    lr = LogisticRegression(
        n_jobs=n_jobs,
        random_state=random_state
    )

    # from sklearn.tree import DecisionTreeClassifier
    # dt = DecisionTreeClassifier(max_depth=5, random_state=random_state)

    cat_features = (df.drop(columns=target_col)
                    .select_dtypes('object').columns)
    num_features = (df.drop(columns=target_col)
                    .select_dtypes('number').columns)

    transformer = ColumnTransformer([
        ('woe', woe, cat_features),
        ('sc', sc, num_features)
    ])

    pipeline = Pipeline([
        ('transformer', transformer),
        ('clf', lr)
    ])

    return pipeline


def train_logistic_regression(df: pd.DataFrame,
                              target_col: str,
                              n_jobs: int,
                              random_state: int) -> Pipeline:

    pipeline = define_lr_pipeline(df,
                                  target_col,
                                  n_jobs,
                                  random_state)
    pipeline.fit(
        df.drop(columns=target_col),
        df[target_col]
    )

    return pipeline


def get_cross_val_score(pipeline: Pipeline,
                        df: pd.DataFrame,
                        target_col: str,
                        n_splits: int,
                        metrics: List,
                        n_jobs: int,
                        random_state: int):
    skf = StratifiedKFold(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=True
    )

    X = df.drop(columns=target_col)
    y = df[target_col]

    results = cross_validate(pipeline,
                             X, y,
                             cv=skf,
                             scoring=metrics,
                             n_jobs=n_jobs)

    key_metrics_results = [result for result in results if result.startswith('test_')]
    metrics_results = {metric: result
                       for metric, result in results.items()
                       if metric in key_metrics_results}

    return metrics_results
