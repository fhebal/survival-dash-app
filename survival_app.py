import streamlit as st
import pandas as pd
from pandas import CategoricalDtype
from lifelines.datasets import load_rossi
from utils import plotter, read_config


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
st.title('Lifelines Survival Analysis')
st.write("Data source: " + 'https://rdrr.io/cran/RcmdrPlugin.survival/man/Rossi.html')
st.write('This data set is originally from Rossi et al. (1980), and is used as an example in Allison (1995). The data pertain to 432 convicts who were released from Maryland state prisons in the 1970s and who were followed up for one year after release. Half the released convicts were assigned at random to an experimental treatment in which they were given financial aid; half did not receive aid.')



st.title('Kaplan-Meier Curves')
option = st.selectbox(
        '',
     features['features'])

#'You selected: ', option
#st.write("### Description: " + "\n" + str(cfg[option]))

#option = 'prio'
plt = plotter(df, option, DURATION, EVENT, num_cols, CategoricalDtype)
KM_plot = st.pyplot(plt)


from lifelines import CoxPHFitter
rossi= load_rossi()

st.title('Cox Proportional Hazards Regression')
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

st.title("Individual predictions")
