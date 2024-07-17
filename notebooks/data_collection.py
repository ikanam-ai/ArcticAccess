#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint
import os

class MapCreator:
    def __init__(self, map_location=[64.5376, 40.5451], zoom_start=10):
        self.map = folium.Map(location=map_location, zoom_start=zoom_start)

    def add_geojson(self, geojson_path, layer_name):
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
            folium.GeoJson(geojson_data, name=layer_name).add_to(self.map)

    def add_markers_from_geojson(self, geojson_path):
        gdf = gpd.read_file(geojson_path)
        
        for index, row in gdf.iterrows():
            station_name = row['name_en']
            geometry = row['geometry']

            if isinstance(geometry, str):
                geometry = MultiPoint(map(float, geometry.strip('MULTIPOINT ').replace('(', '').replace(')', '').split()))

            for point in geometry.geoms:
                lat, lon = point.y, point.x
                folium.Marker([lat, lon], popup=station_name).add_to(self.map)

    def save_map(self, output_path):
        self.map.save(output_path)

# Путь к директории с файлами
# directory_path = ''

# Список файлов для обработки
file_list = [
    'arhangelskaya_oblast.geojson',
    'base_obl_people.geojson',
    'mun_obr_all_bad.geojson',
    'oopt_gp_point.geojson',
    'oopt_gp_poly.geojson',
    'path_to_hexagons.geojson',
    '_Список слоев и описание.xlsx',
    'Модель развития туристической отрасли.xlsx'
]

# Пример использования класса
if __name__ == "__main__":
    map_creator = MapCreator()

    # Добавляем GeoJSON файлы
    for file_name in file_list:
        if file_name.endswith('.geojson'):
            map_creator.add_geojson(os.path.join(directory_path, file_name), file_name)

    # Если есть дополнительные данные для добавления (например, точки метро)
    # map_creator.add_markers_from_geojson(os.path.join(directory_path, 'metro_arkhangelsk.geojson'))

    # Сохраняем карту
    map_creator.save_map('arkhangelsk_map.html')

