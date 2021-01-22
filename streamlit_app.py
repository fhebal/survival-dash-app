import streamlit as st
import pandas as pd
from pandas import CategoricalDtype
from lifelines.datasets import load_rossi
from lifelines import WeibullAFTFitter
from utils import plotter, read_config


st.set_page_config(layout="wide")
rossi = load_rossi()

# setup
df = load_rossi()
DURATION = 'week'
EVENT = 'arrest'
cfg = read_config('data_dictionary.yaml')

# custom features
df['Age Group'] = pd.cut(df['age'], bins=4, precision=0)
df['Priors'] = pd.cut(df['prio'], bins=4, precision=0)
num_cols = ['age', 'prio']
option = 'priors_group'

features = pd.DataFrame({'features': [x for x in df.columns if x not in [DURATION, EVENT] + num_cols]})



# STREAMLIT CODE
st.title('KKBox Survival Analysis')
st.write("Data source: " + 'https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data')
st.write('''In this challenge, you are asked to predict whether a user will churn after his/her subscription expires. Specifically, we want to forecast if a user make a new service subscription transaction within 30 days after the current membership expiration date.

KKBOX offers subscription based music streaming service. When users signs up for our service, users can choose to either manual renew or auto-renew the service. Users can actively cancel their membership at any time. In this dataset, KKBox has included more users behaviors than the ones in train and test datasets, in order to enable participants to explore different user behaviors outside of the train and test sets. For example, a user could actively cancel the subscription, but renew within 30 days.
''')

col1, col2 = st.beta_columns(2)

with col1:
    # KAPLAN MEIER CURVES
    st.title('Kaplan-Meier Curves')
    option = st.selectbox(
            '',
         features['features'])

    #'You selected: ', option
    #st.write("### Description: " + "\n" + str(cfg[option]))

    #option = 'prio'
    plt = plotter(df, option, DURATION, EVENT, num_cols, CategoricalDtype)
    KM_plot = st.pyplot(plt)


with col2:
    # COX PROPORTIONAL HAZARDS SUMMARY
    from lifelines import CoxPHFitter
    rossi= load_rossi()

    st.title('Regression Model Summary')
    cph = CoxPHFitter()
    cph.fit(rossi, duration_col='week', event_col='arrest')

    st.write("## Coefficients")
    cols = ['coef','exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
    cph.summary[cols]

    st.write("## Summary")
    col_2 = [
        cph._n_examples,
        sum(cph.event_observed),
        cph.baseline_estimation_method,
        cph.event_col,
        cph.duration_col,
        cph._class_name,
        cph.log_likelihood_,
        cph.concordance_index_,
        cph.AIC_partial_]

    col_1 = [
        'observations',
        'events_oberved',
        'baseline estimation',
        'event column',
        'duration column',
        'model',
        'log likelihood',
        'concordance',
        'partial AIC'    
    ]
    results = pd.DataFrame([col_1,col_2]).T
    results.columns = ['', ' ']
    results.set_index('')
    results







# INDIVIDAL PREDICTIONS 
st.sidebar.title("Individual Prediction")
week = st.sidebar.slider(
    'Weeks on subscription',
    0, 52
)
fin = st.sidebar.slider(
    'Discount',
    0, 1
)
age = st.sidebar.slider(
    'Age',
    17, 75
)

mar = st.sidebar.slider(
    'Marital Status',
    0, 1
)

paro = st.sidebar.slider(
    'Referral',
    0, 1
)





wf = WeibullAFTFitter().fit(rossi, "week", "arrest")
predict_input = pd.DataFrame([week, 0, fin, age, 1, 1, mar, paro, 1]).T
predict_input.columns = ['week', 'arrest', 'fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
prediction_output = wf.predict_median(predict_input, conditional_after=predict_input[DURATION])

st.sidebar.write("## Weeks until churn:", round(prediction_output[0]))

