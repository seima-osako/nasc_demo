import pandas as pd
import geopandas as gpd

import folium
from folium import *
import streamlit as st
from streamlit_folium import folium_static

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

st.set_page_config(layout="wide")
st.write("## 若穂綿内")

st.sidebar.write("### 閾値")
thresholds = st.sidebar.slider("thresholds", min_value=0.0, max_value=1.0, step=0.1, value=0.5)

df_test = pd.read_csv('data/df_test.csv')
df_test['pred_target'] = df_test['lgbm_proba'].apply(lambda x: 1 if x >= thresholds else 0)

df_test = df_test.sort_values(by='lgbm_proba')
fig = make_subplots(rows=1, cols=2, print_grid=False)
tn, fp, fn, tp = confusion_matrix(df_test['target'], df_test['pred_target']).flatten()
confmat = confusion_matrix(df_test['target'], df_test['pred_target'])

Accuracy  =  round(((tp+tn)/(tp+tn+fp+fn)), 3)*100 # 正解率
Precision =  round(tp/(tp+fp), 3)*100 # 精度
FPR = round(fp/(tn+fp), 3)*100 # 偽陽性率

show_metrics = pd.DataFrame(data=[[Accuracy , Precision, FPR]])
show_metrics = show_metrics.T

trace1 = go.Bar(x = (show_metrics[0].values), 
                y = ['正解率', '精度', '偽陽性率'], text = np.round_(show_metrics[0].values,4),
                textposition = 'auto',
                orientation = 'h', opacity = 0.8, marker=dict(color=['gold', 'lightgreen', 'lightcoral'],line=dict(color='#000000',width=1.5)))
trace2 = px.imshow(confmat, x = ["耕作地 (pred)","非耕作地 (pred)"], y = ["耕作地 (true)","非耕作地 (true)"], text_auto=True).update_layout(showlegend=False)
trace2 = go.Figure(trace2.data, trace2.layout)

fig.append_trace(trace1,1,1)
fig.add_trace(trace2.data[0],1,2)
st.plotly_chart(fig, use_container_width=True)