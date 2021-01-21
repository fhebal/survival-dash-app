import streamlit as st
import numpy as np
import pandas as pd
from lifelines.datasets import load_rossi

df = load_rossi()



import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

ax = plt.subplot(111)

def plotter(data, option):
    dem = (data[option] == 1)

    T = data["week"]
    E = data["arrest"]

    kmf.fit(T, event_observed=E)

    kmf.fit(T[dem], event_observed=E[dem], label="Democratic Regimes")
    kmf.plot_survival_function(ax=ax)

    kmf.fit(T[~dem], event_observed=E[~dem], label="Non-democratic Regimes")
    kmf.plot_survival_function(ax=ax)

    plt.title("Lifespans of different global regimes");
    return plt

# STREAMLIT CODE
st.title('Lifelines Survival Analysis')

st.write("""
The dataset required for survival regression must be in the format of a Pandas DataFrame. Each row of the DataFrame represents an observation. There should be a column denoting the durations of the observations. There may (or may not) be a column denoting the event status of each observation (1 if event occurred, 0 if censored). There are also the additional covariates you wish to regress against. Optionally, there could be columns in the DataFrame that are used for stratification, weights, and clusters which will be discussed later in this tutorial.""")
#st.write(df)



chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

#st.line_chart(chart_data)





st.title('CHECKBOXES')
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)


DURATION = 'week'
EVENT = 'arrest'
features = pd.DataFrame({'features': [x for x in df.columns if x not in [DURATION, EVENT]]})


st.title('SELECTBOX')
option = st.selectbox(
        'Select an feature:',
     features['features'])

'You selected: ', option

#option = 'prio'
plt = plotter(df, option)
the_plot = st.pyplot(plt)


