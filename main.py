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
st.write("## 若穂綿内における耕作地/非耕作地の分類結果")
st.write("")
st.sidebar.write("### 識別閾値（＝非耕作地である確率）")
thresholds = st.sidebar.slider(
    "thresholds", min_value=0.0, max_value=1.0, step=0.01, value=0.5
)

df_test = pd.read_csv("data/df_test.csv")
df_test["pred_target"] = df_test["lgbm_proba"].apply(
    lambda x: 1 if x >= thresholds else 0
)
df_test = df_test.sort_values(by="lgbm_proba")

st.write("### 評価データに対する性能")
fig = make_subplots(
    rows=1,
    cols=2,
    print_grid=False,
    subplot_titles=("Metrics", "Confusion Matrix"),
)
tn, fp, fn, tp = confusion_matrix(df_test["target"], df_test["pred_target"]).flatten()
confmat = confusion_matrix(df_test["target"], df_test["pred_target"])

Accuracy = round(((tp + tn) / (tp + tn + fp + fn)), 3) * 100  # 正解率
Precision = round(tp / (tp + fp), 3) * 100  # 精度
Recall = round(tp / (tp + fn), 3) * 100  # 再現率
FPR = round(fp / (tn + fp), 3) * 100  # 偽陽性率

show_metrics = pd.DataFrame(data=[[Accuracy, Precision, Recall, FPR]]).T

trace1 = go.Bar(
    x=(show_metrics[0].values),
    y=["正解率", "適合率", "再現率", "偽陽性率"],
    text=np.round_(show_metrics[0].values, 4),
    textposition="auto",
    orientation="h",
    opacity=0.8,
    marker=dict(
        color=["gold", "lightgreen", "lightskyblue", "lightcoral"],
        line=dict(color="#000000", width=1.5),
    ),
)
trace2 = px.imshow(
    confmat,
    x=["耕作地 (予測)", "非耕作地 (予測)"],
    y=["耕作地 (正解)", "非耕作地 (正解)"],
    text_auto=True,
)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2.data[0], 1, 2)
fig.update_annotations(font_size=25)
fig.update_layout(
    font_size=20,
    xaxis=dict(title_text="%", tickfont=dict(size=18)),
    yaxis=dict(tickfont=dict(size=18)),
)
st.plotly_chart(fig, use_container_width=True)


gdf = gpd.read_file("data/polygon_wakaho.geojson")
gdf = gdf.drop(columns=["R3_result", "R4_result", "land_cover", "abandoned_label"])
gdf = pd.merge(
    gdf,
    df_test[["OBJECTID", "R4_result", "target", "pred_target", "lgbm_proba"]],
    on=["OBJECTID"],
)


st.sidebar.write("### 背景地図")
bm = st.sidebar.radio(
    "Please select basemap",
    ("Google-Satellite-Hybrid", "Google-Maps", "Google-Terrain", "Esri-Satellite"),
)

basemaps = {
    "Google-Maps": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Maps",
        overlay=True,
        control=True,
    ),
    "Google-Satellite-Hybrid": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ),
    "Google-Terrain": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Terrain",
        overlay=True,
        control=True,
    ),
    "Esri-Satellite": folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=True,
        control=True,
    ),
}

st.write("#### 予測＝正解（青色） / 予測！＝正解（赤色）")

m = folium.Map(location=[36.61979182743826, 138.27179683757538], zoom_start=14)
basemaps[bm].add_to(m)

for r in gdf.itertuples():
    sim_geo = gpd.GeoSeries(r.geometry).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    if r.target != r.pred_target:
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x: {
                "color": "black",
                "fillColor": "red",
                "fillOpacity": 0.7,
                "weight": 1.5,
            },
        )
    else:
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x: {
                "color": "black",
                "fillColor": "blue",
                "weight": 2,
            },
        )
    folium.Popup(
        f"OBJECTID：{r.OBJECTID}<br>令和4年度 農地調査結果：{r.R4_result}<br>非耕作地である確率＝{round(r.lgbm_proba, 3)}",
        max_width=1000,
        max_height=2500,
    ).add_to(geo_j)
    geo_j.add_to(m)

folium_static(m, width=1000, height=500)
