import os
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
import json
from layer import add_brand_layers

SP_SHEET     = 'tech0_01' 

if 'show_all' not in st.session_state:
    st.session_state['show_all'] = False  

def toggle_show_all():
    st.session_state['show_all'] = not st.session_state['show_all']


def load_data_from_spreadsheet():
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    google_credentials = {
        "type": st.secrets["GOOGLE_CREDENTIALS"]["type"],
        "project_id": st.secrets["GOOGLE_CREDENTIALS"]["project_id"],
        "private_key_id": st.secrets["GOOGLE_CREDENTIALS"]["private_key_id"],
        "private_key": st.secrets["GOOGLE_CREDENTIALS"]["private_key"],
        "client_email": st.secrets["GOOGLE_CREDENTIALS"]["client_email"],
        "client_id": st.secrets["GOOGLE_CREDENTIALS"]["client_id"],
        "auth_uri": st.secrets["GOOGLE_CREDENTIALS"]["auth_uri"],
        "token_uri": st.secrets["GOOGLE_CREDENTIALS"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["GOOGLE_CREDENTIALS"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["GOOGLE_CREDENTIALS"]["client_x509_cert_url"]
    }

    credentials = Credentials.from_service_account_info(
        google_credentials,
        scopes=scopes
    )
    gc = gspread.authorize(credentials)

    SP_SHEET_KEY = st.secrets["SP_SHEET_KEY"] 
    sh  = gc.open_by_key(SP_SHEET_KEY)

    worksheet = sh.worksheet(SP_SHEET) 
    pre_data  = worksheet.get_all_values()
    col_name = pre_data[0][:]
    df = pd.DataFrame(pre_data[1:], columns=col_name) 

    return df

# ブランド地区データ
brand_areas = pd.DataFrame({
    'brand': ['治安の良い地区', '公園の多い地区', '大型の一戸建ての多い地区', '文教地区'],
    'latitude': [35.6895, 35.6995, 35.7095, 35.7195],
    'longitude': [139.6917, 139.7017, 139.7117, 139.7217]
})

# マイナスのブランド地区データ
negative_areas = pd.DataFrame({
    'brand': ['治安の悪い地区', 'ハザードマップの水害危険地区'],
    'latitude': [35.6890, 35.6980],
    'longitude': [139.6900, 139.7000],
    'radius': [1200, 500]  # メートル単位の半径
})

# 仮のポリゴンデータ
polygons = [
    [(35.7095, 139.7007), (35.6695, 139.7427), (35.6295, 139.6917), (35.6595, 139.6717)],  # 治安の良い地区
    [(35.7095, 139.7417), (35.6795, 139.7717), (35.6695, 139.7417), (35.6995, 139.7217)],  # 公園の多い地区
    [(35.7335, 139.7217), (35.7095, 139.7617), (35.6595, 139.7117), (35.7195, 139.6917)],  # 大型の一戸建ての多い地区
    [(35.7805, 139.7217), (35.7495, 139.7417), (35.7295, 139.7117), (35.7695, 139.6817)]   # 文教地区
]

# 地図の初期設定
map_center = [35.6895, 139.6917]
m = folium.Map(location=map_center, zoom_start=12)

# ブランド地区の選択
st.sidebar.title("ブランド地区の選択")
selected_brands = []
for brand in brand_areas['brand']:
    if st.sidebar.checkbox(brand, False):  # デフォルトでオフに設定
        selected_brands.append(brand)

# ネガティブエリアの選択
st.sidebar.title("ネガティブエリアの選択")
selected_negative_brands = []
for brand in negative_areas['brand']:
    if st.sidebar.checkbox(brand, False):  # デフォルトでオフに設定
        selected_negative_brands.append(brand)

# 選択されたブランド地区のみをフィルタリング
selected_brand_areas = brand_areas[brand_areas['brand'].isin(selected_brands)]

# 選択されたインデックスに基づいてポリゴンをフィルタリング
selected_indices = selected_brand_areas.index.tolist()
selected_polygons = [polygons[i] for i in selected_indices]

# 選択されたネガティブエリアのみをフィルタリング
selected_negative_areas = negative_areas[negative_areas['brand'].isin(selected_negative_brands)]

# 選択されたブランド地区とネガティブエリアのレイヤーを追加
m = add_brand_layers(m, selected_brand_areas, selected_polygons, selected_negative_areas)

# 地図を表示
folium_static(m)

def preprocess_dataframe(df):
    df['家賃'] = pd.to_numeric(df['家賃'], errors='coerce')
    df = df.dropna(subset=['家賃'])
    return df

def make_clickable(url, name):
    return f'<a target="_blank" href="{url}">{name}</a>'

def create_map(filtered_df):
    map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    for idx, row in filtered_df.iterrows():
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            popup_html = f"""
            <b>名称:</b> {row['名称']}<br>
            <b>アドレス:</b> {row['アドレス']}<br>
            <b>家賃:</b> {row['家賃']}万円<br>
            <b>間取り:</b> {row['間取り']}<br>
            <a href="{row['物件詳細URL']}" target="_blank">物件詳細</a>
            """
            popup = folium.Popup(popup_html, max_width=400)
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=popup
            ).add_to(m)

    return m

def display_search_results(filtered_df):
    filtered_df['物件番号'] = range(1, len(filtered_df) + 1)
    filtered_df['物件詳細URL'] = filtered_df['物件詳細URL'].apply(lambda x: make_clickable(x, "リンク"))
    display_columns = ['物件番号', '名称', 'アドレス', '階数', '家賃', '間取り', '物件詳細URL']
    filtered_df_display = filtered_df[display_columns]
    st.markdown(filtered_df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

def main():
    df = load_data_from_spreadsheet()
    df = preprocess_dataframe(df)

    st.title('賃貸物件情報の可視化')

    col1, col2 = st.columns([1, 2])

    with col1:
        area = st.radio('■ エリア選択', df['区'].unique())


    with col2:
        price_min, price_max = st.slider(
            '■ 家賃範囲 (万円)', 
            min_value=float(1), 
            max_value=float(df['家賃'].max()),
            value=(float(df['家賃'].min()), float(df['家賃'].max())),
            step=0.1,  
            format='%.1f'
        )

    with col2:
        type_options = st.multiselect('■ 間取り選択', df['間取り'].unique(), default=df['間取り'].unique())


    filtered_df = df[(df['区'].isin([area])) & (df['間取り'].isin(type_options))]
    filtered_df = filtered_df[(filtered_df['家賃'] >= price_min) & (filtered_df['家賃'] <= price_max)]
    filtered_count = len(filtered_df)

    filtered_df['latitude'] = pd.to_numeric(filtered_df['latitude'], errors='coerce')
    filtered_df['longitude'] = pd.to_numeric(filtered_df['longitude'], errors='coerce')
    filtered_df2 = filtered_df.dropna(subset=['latitude', 'longitude'])


    col2_1, col2_2 = st.columns([1, 2])

    with col2_2:
        st.write(f"物件検索数: {filtered_count}件 / 全{len(df)}件")

    if col2_1.button('検索＆更新', key='search_button'):
        st.session_state['filtered_df'] = filtered_df
        st.session_state['filtered_df2'] = filtered_df2
        st.session_state['search_clicked'] = True

    if st.session_state.get('search_clicked', False):
        m = create_map(st.session_state.get('filtered_df2', filtered_df2))
        folium_static(m)

    show_all_option = st.radio(
        "表示オプションを選択してください:",
        ('地図上の検索物件のみ', 'すべての検索物件'),
        index=0 if not st.session_state.get('show_all', False) else 1,
        key='show_all_option'
    )

    st.session_state['show_all'] = (show_all_option == 'すべての検索物件')

    if st.session_state.get('search_clicked', False):
        if st.session_state['show_all']:
            display_search_results(st.session_state.get('filtered_df', filtered_df))  
        else:
            display_search_results(st.session_state.get('filtered_df2', filtered_df2))  


if __name__ == "__main__":
    if 'search_clicked' not in st.session_state:
        st.session_state['search_clicked'] = False
    if 'show_all' not in st.session_state:
        st.session_state['show_all'] = False
    main()