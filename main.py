import numpy as np
import pandas as pd
import geopandas as gpd

import folium
from folium import *
import streamlit as st
from st_aggrid import AgGrid
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


def add_cm_flg(x):
    if x["target"] == 0 and x["pred_target"] == 0:
        return "TN"
    elif x["target"] == 1 and x["pred_target"] == 1:
        return "TP"
    elif x["target"] == 0 and x["pred_target"] == 1:
        return "FP"
    elif x["target"] == 1 and x["pred_target"] == 0:
        return "FN"


df_test = pd.read_csv("data/df_test.csv")
df_test["pred_target"] = df_test["lgbm_proba"].apply(
    lambda x: 1 if x >= thresholds else 0
)
df_test = df_test.sort_values(by="lgbm_proba")
df_test["cm_flg"] = df_test.apply(lambda x: add_cm_flg(x), axis=1)

st.write("### 評価データに対する性能")
st.write(
    """
    - 偽陽性率：全体の耕作地(正解)に対して、間違えて非耕作地と予測した割合
    - 真陰性率：モデルで耕作地と予測した中で、正しく耕作地と判定できた割合
    - 再現率：全体の非耕作地(正解)に対して、正しく捕捉できた割合
    - 適合率：モデルで非耕作地と予測した中で、正しく非耕作地と判定できた割合
"""
)
fig = make_subplots(
    rows=1,
    cols=2,
    print_grid=False,
    subplot_titles=("Metrics", "Confusion Matrix"),
)
tn, fp, fn, tp = confusion_matrix(df_test["target"], df_test["pred_target"]).flatten()
confmat = confusion_matrix(df_test["target"], df_test["pred_target"])

# Accuracy = round(((tp + tn) / (tp + tn + fp + fn)), 3) * 100  # 正解率
Precision = round(tp / (tp + fp), 3) * 100  # 適合率
Recall = round(tp / (tp + fn), 3) * 100  # 再現率
FPR = round(fp / (tn + fp), 3) * 100  # 偽陽性率
TPF = round(tn / (tn + fn), 3) * 100  # 真陰性率

show_metrics = pd.DataFrame(data=[[Precision, Recall, TPF, FPR]]).T

trace1 = go.Bar(
    x=(show_metrics[0].values),
    y=["適合率", "再現率", "真陰性率", "偽陽性率"],
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
    df_test[["OBJECTID", "R4_result", "target", "pred_target", "lgbm_proba", "cm_flg"]],
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

st.write("#### 結果の可視化")
st.write(
    """
    - FN（False Negative）：耕作地と予測した中で、誤って非耕作地と判定してしまった区画
    - FP（False Positive）：非耕作地と予測した中で、誤って耕作地と判定してしまった区画
    - TN（True Negative）：正しく耕作地と予測できた区画
    - TP（True Positive）：正しく非耕作地と予測できた区画

"""
)

cm_option = st.selectbox("可視化する区画を選ぶ", ("FN", "FP", "TN", "TP"))
vis_gdf = gdf[gdf["cm_flg"] == cm_option]

m = folium.Map(location=[36.61979182743826, 138.27179683757538], zoom_start=14)
basemaps[bm].add_to(m)

for r in vis_gdf.itertuples():
    sim_geo = gpd.GeoSeries(r.geometry).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(
        data=geo_j,
        style_function=lambda x: {
            "color": "black",
            "fillColor": "red",
            "fillOpacity": 0.7,
            "weight": 1.5,
        },
    )
    folium.Popup(
        f"OBJECTID：{r.OBJECTID}<br>令和4年度 農地調査結果：{r.R4_result}<br>非耕作地である確率＝{round(r.lgbm_proba, 3)}",
        max_width=1000,
        max_height=2500,
    ).add_to(geo_j)
    geo_j.add_to(m)

folium_static(m, width=1000, height=500)

vis_gdf_show = vis_gdf.drop(columns=["lgbm_proba", "cm_flg", "geometry"])
vis_gdf_show["target"] = vis_gdf_show["target"].apply(
    lambda x: "耕作地" if x == 0 else "非耕作地"
)
vis_gdf_show["pred_target"] = vis_gdf_show["pred_target"].apply(
    lambda x: "耕作地" if x == 0 else "非耕作地"
)
vis_gdf_show = vis_gdf_show.rename(columns={"target": "正解", "pred_target": "予測"})

@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('SJIS')


csv = convert_df(vis_gdf_show)

st.download_button(
   "Download",
   csv,
   f"{cm_option}.csv",
   "text/csv",
   key='download-csv'
)

AgGrid(
    vis_gdf_show,
    theme="streamlit",
    fit_columns_on_grid_load=True,
    height=500,
)
