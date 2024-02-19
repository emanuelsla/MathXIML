"""
streamlit UI
$ python -m streamlit run modules/ui/Mockup_StreamlitCaipiUI_GermanCreditRisk.py
"""


import plotly.express as px
import plotly.graph_objects as go
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from modules.preprocessors.preprocess_german_credit_data import preprocess_german_credit_data
from modules.counterfactual_explainers.apply_dice import apply_dice


def radar_chart(df: pd.DataFrame):
    fig = px.line_polar(df, r='r', theta='theta', color='instance', line_close=True,
                        color_discrete_sequence=['gray', 'orange'])
    fig.update_traces(fill='toself')
    fig.update_layout(margin=dict(l=25, r=25, t=25, b=25))
    st.write(fig)


def gauge_chart(val=1):
    fig = go.Figure(go.Indicator(
        mode="gauge",
        gauge={'shape': "bullet",
               'axis': {'range': [0, 1]},
               'steps': [
                   {'range': [0, 0.5], 'color': "lightpink"},
                   {'range': [0.5, 1], 'color': "lightblue"}],
               'bar': {'color': "black"}},
        value=val))
    fig.update_layout(
        autosize=False,
        width=700,
        height=20,
        margin=dict(l=0, r=0, t=0, b=0))
    st.write(fig)


if __name__ == '__main__':

    data_path = 'binary-tabular-caipi/data/german_credit_data.csv'
    df_train, df_test, _, metric_transformers = preprocess_german_credit_data(data_path, test_ratio=0.3, seed=42)

    df_plot = df_train.copy()

    model_path = 'binary-tabular-caipi/models/20231009085100_svc.joblib'
    svc = joblib.load(model_path)

    for col, transformer in metric_transformers.items():
        df_plot[col] = transformer.inverse_transform(np.asarray(df_plot[col]).reshape((-1, 1))).flatten()
        df_plot[col] = np.round(df_plot[col], -1) / 10

    df_test = df_test.reset_index(drop=True)
    instance = df_test[0:1].drop(['risk'], axis=1)

    with st.container():

        st.subheader('Modify Counterfactual')

        radius = list(instance.iloc[0])
        theta = list(instance.columns)
        df_orig = pd.DataFrame.from_dict({'r': radius, 'theta': theta})
        df_orig['instance'] = 'original'

        pred_orig = svc.predict_proba(np.asarray(instance))

        age_slider = st.slider("Select 'age'", 0, 10, 6)
        sex_slider = st.slider("Select 'sex'", 0, 1, 1)
        job_slider = st.slider("Select 'job'", 0, 3, 1)
        housing_slider = st.slider("Select 'housing'", 0, 2, 1)
        savingaccounts_slider = st.slider("Select 'savingaccounts'", 0, 3, 1)
        checkingaccount_slider = st.slider("Select 'checkingaccount'", 0, 3, 0)
        duration_slider = st.slider("Select 'duration'", 0, 10, 2)
        purpose_slider = st.slider("Select 'purpose'", 0, 7, 5)

        radius = [age_slider, sex_slider, job_slider, housing_slider,
                  savingaccounts_slider, checkingaccount_slider,
                  duration_slider, purpose_slider]
        df_cf = pd.DataFrame.from_dict({'r': radius, 'theta': theta})
        df_cf['instance'] = 'counterfactual'

        cf_instance_updated = np.asarray(radius).reshape(1, -1)
        cf_instance_updated = np.asarray(radius).reshape(1, -1)
        pred_cf = svc.predict_proba(np.asarray(cf_instance_updated))

        df = pd.concat([df_orig, df_cf])

    with st.container():
        radar_chart(df)
        # st.write('Classification Score Original Instance - Credit Risk (bad: 0, good: 1)')
        # gauge_chart(pred_orig[0][1])
        # st.write('Classification Score Counterfactual Instance - Credit Risk (bad: 0, good: 1)')

        st.write("Counterfactual Classification Outcome (red: 'high risk', blue: 'low risk')")
        gauge_chart(pred_cf[0][1])

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button('RRR', type='secondary')
        with col2:
            st.button('RWR', type='secondary')
        with col3:
            st.button('W', type='secondary')

    with st.container():
        st.subheader('Variable Information')

        st.write("savingaccounts: {'little': 0, 'moderate': 1, 'quiterich': 2, 'rich': 3}")
        st.write("checkingaccount: {'little': 0, 'moderate': 1, 'rich': 2}")
        st.write("job: {'highlyskilled': 0, 'skilled': 1, 'unskilled_no_res': 2, 'unskilled_res': 3}")
        st.write("sex: {'female': 0, 'male': 1}")
        st.write("housing: {'free': 0, 'own': 1, 'rent': 2}")
        st.write("purpose: {'business': 0, 'car': 1, 'domesticappliances': 2, "
                 "'education': 3, 'furniture/equipment': 4, 'radio/TV': 5, 'repairs': 6, 'vacation/others': 7}")
        st.write("'age' and 'duration': x ** 10 (rounded)")
