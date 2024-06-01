# layer.py
import folium

def add_brand_layers(m, brand_areas, polygons, negative_areas):
    for idx, row in enumerate(brand_areas.iterrows()):
        row_data = row[1]
        folium.Polygon(
            locations=polygons[idx],
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.2,
            popup=row_data['brand'],
            tooltip=folium.Tooltip(row_data['brand'])
        ).add_to(m)

    # マイナスのブランド地区を赤色の円で表示
    for _, row in negative_areas.iterrows():
        folium.Circle(
            location=(row['latitude'], row['longitude']),
            radius=row['radius'],
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.4,
            popup=row['brand'],
            tooltip=folium.Tooltip(row['brand'])
        ).add_to(m)
    
    return m