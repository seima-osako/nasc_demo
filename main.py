import numpy as np
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
st.write("## 若穂綿内の耕作地・非耕作地分類結果")

st.sidebar.write("### 閾値")
thresholds = st.sidebar.slider("thresholds", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

df_test = pd.read_csv('data/df_test.csv')
df_test['pred_target'] = df_test['lgbm_proba'].apply(lambda x: 1 if x >= thresholds else 0)
df_test = df_test.sort_values(by='lgbm_proba')

st.write("### Test区画に対する汎化性能")
fig = make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=('Metrics', 'Confusion Matrix'))
tn, fp, fn, tp = confusion_matrix(df_test['target'], df_test['pred_target']).flatten()
confmat = confusion_matrix(df_test['target'], df_test['pred_target'])

Accuracy  =  round(((tp+tn)/(tp+tn+fp+fn)), 3)*100 # 正解率
Precision =  round(tp/(tp+fp), 3)*100 # 精度
Recall =round(tp/(tp+fn), 3)*100 # 再現率 
FPR = round(fp/(tn+fp), 3)*100 # 偽陽性率

show_metrics = pd.DataFrame(data=[[Accuracy , Precision, Recall, FPR]]).T

trace1 = go.Bar(x = (show_metrics[0].values), 
                y = ['正解率', '精度', '再現率', '偽陽性率'], text = np.round_(show_metrics[0].values,4),
                textposition = 'auto',
                orientation = 'h', opacity = 0.8, marker=dict(color=['gold', 'lightgreen', 'lightcoral', 'lightskyblue'],line=dict(color='#000000',width=1.5)))
trace2 = px.imshow(confmat, x = ["耕作地 (pred)","非耕作地 (pred)"], y = ["耕作地 (true)","非耕作地 (true)"], text_auto=True).update_layout(showlegend=False)
trace2 = go.Figure(trace2.data, trace2.layout)

fig.append_trace(trace1,1,1)
fig.add_trace(trace2.data[0],1,2)
fig['layout'].update(showlegend=False, plot_bgcolor='rgba(240, 240, 240, 0.95)', paper_bgcolor='rgba(240, 240, 240, 0.95)', margin=dict(b=100))
st.plotly_chart(fig, use_container_width=True)


gdf = gpd.read_file('data/polygon_wakaho.geojson')
gdf = gdf.drop(columns=['R3_result', 'R4_result', 'land_cover', 'abandoned_label'])
gdf = pd.merge(gdf, df_test[['OBJECTID', 'R4_result', 'target', 'pred_target', 'lgbm_proba']], on=['OBJECTID'])


st.sidebar.write("### 背景地図")
bm = st.sidebar.radio(
    "Please select basemap",
    ( "Esri-Satellite", "Google-Maps", "Google-Satellite-Hybrid")
)

basemaps = {
    'Google-Maps': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Maps',
        overlay = True,
        control = True
    ),
    'Google-Satellite-Hybrid': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Esri-Satellite': folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
    )
}

st.write("#### True=Pred（青色） / True!=Pred（赤色）")

m = folium.Map(location=[36.61979182743826, 138.27179683757538], zoom_start=14)
basemaps[bm].add_to(m)

for r in gdf.itertuples():
    sim_geo = gpd.GeoSeries(r.geometry).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    if r.target!=r.pred_target:
      geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'color' : 'black', 'fillColor': 'red', 'weight': 2})
    else:
      geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'color' : 'black', 'fillColor': 'blue', 'weight': 2})
    folium.Popup(f'農地調査結果：{r.R4_result}', max_width=1000, max_height=2500).add_to(geo_j)
    geo_j.add_to(m)

folium_static(m, width=1000, height=500)