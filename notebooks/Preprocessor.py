import pandas as pd
import numpy as np
import geopandas as gpd
from geopy.distance import distance
from tqdm import tqdm
from typing import List, Dict, Tuple


class DatasetProcessor:
    def __init__(self, geojson_files: List[str], config: Dict[str, float]):
        """
        Инициализация класса DatasetProcessor.

        :param geojson_files: Список файлов GeoJSON для обработки.
        :param config: Конфигурация для разбиения на группы.
        """
        self.geojson_files = geojson_files
        self.config = config
        self.gdf = None
        self.df = pd.DataFrame()

    def load_geojson_files(self) -> None:
        """Загрузка GeoJSON файлов и объединение их в один GeoDataFrame."""
        dataframes = []
        for file in self.geojson_files:
            filepath = f"{directory_path}/{file}"
            if file.endswith('.geojson'):
                gdf = gpd.read_file(filepath)
                dataframes.append(gdf)
        self.gdf = pd.concat(dataframes, ignore_index=True)

    def preprocess_geometries(self) -> None:
        """Предобработка геометрий: извлечение широты и долготы."""
        self.gdf['lat'] = self.gdf['geometry'].apply(lambda p: p.y)
        self.gdf['lon'] = self.gdf['geometry'].apply(lambda p: p.x)

    def compute_distances(self) -> None:
        """Вычисление расстояний до объектов по классам."""
        fclasses = self.gdf['fclass'].unique()
        self.df['points'] = [[] for _ in range(len(self.df))]

        for fclass in tqdm(fclasses):
            fclass_points = self.gdf[self.gdf['fclass'] == fclass]
            self.df[f'dist_{fclass}'] = self.df['points'].apply(
                lambda points: [
                    self.find_min_distance(float(p['lat']), float(p['lon']), fclass_points) for p in points
                ]
            )
            self.calculate_distance_statistics(fclass)

    def find_min_distance(self, lat: float, lon: float, fclass_points: gpd.GeoDataFrame) -> float:
        """
        Нахождение минимального расстояния до ближайшей точки заданного класса.

        :param lat: Широта точки.
        :param lon: Долгота точки.
        :param fclass_points: GeoDataFrame с точками заданного класса.
        :return: Минимальное расстояние.
        """
        dists = np.sqrt((fclass_points['lat'].values - lat) ** 2 + (fclass_points['lon'].values - lon) ** 2)
        return np.min(dists)

    def calculate_distance_statistics(self, fclass: str) -> None:
        """Вычисление статистики расстояний для заданного класса."""
        self.df[f'dist_{fclass}_mean'] = self.df[f'dist_{fclass}'].apply(np.mean)
        self.df[f'dist_{fclass}_max'] = self.df[f'dist_{fclass}'].apply(np.max)
        self.df[f'dist_{fclass}_min'] = self.df[f'dist_{fclass}'].apply(np.min)
        self.df[f'dist_{fclass}_median'] = self.df[f'dist_{fclass}'].apply(np.median)
        self.df[f'dist_{fclass}_std'] = self.df[f'dist_{fclass}'].apply(np.std)

    def create_groups(self) -> Dict[str, int]:
        """
        Создание групп по заданным интервалам.

        :return: Словарь с количеством точек в каждой группе.
        """
        x_intervals = self.split_on_intervals(
            self.config['min_xval'],
            self.config['max_xval'],
            self.config['x_ngroups']
        )
        y_intervals = self.split_on_intervals(
            self.config['min_yval'],
            self.config['max_yval'],
            self.config['y_ngroups']
        )
        groups = self.create_group_dict(x_intervals, y_intervals)

        for i in range(len(self.df)):
            points = np.array([[float(x['lat']), float(x['lon'])] for x in self.df['points'].iloc[i]])
            group_values = self.sort_points_into_groups(points[:, 0], points[:, 1], x_intervals, y_intervals, groups)
            for idx, key in enumerate(groups.keys()):
                groups[key] += group_values[idx]

        return groups

    def split_on_intervals(self, min_val: float, max_val: float, n: int) -> List[float]:
        """
        Деление интервала на равные подинтервалы.

        :param min_val: Минимальное значение интервала.
        :param max_val: Максимальное значение интервала.
        :param n: Количество подинтервалов.
        :return: Список значений границ подинтервалов.
        """
        step = (max_val - min_val) / n
        return [min_val + (step * x) for x in range(n + 1)]

    def create_group_dict(self, x_intervals: List[float], y_intervals: List[float]) -> Dict[str, int]:
        """
        Создание словаря групп с инициализацией значений.

        :param x_intervals: Границы по оси X.
        :param y_intervals: Границы по оси Y.
        :return: Словарь групп с нулевыми значениями.
        """
        groups = {}
        x_intervals = np.concatenate([[-np.inf], x_intervals, [np.inf]])
        y_intervals = np.concatenate([[-np.inf], y_intervals, [np.inf]])
        for x_i in range(len(x_intervals) - 1):
            for y_i in range(len(y_intervals) - 1):
                groups[f'x : {x_intervals[x_i]} - {x_intervals[x_i + 1]} | '
                       f'y : {y_intervals[y_i]} - {y_intervals[y_i + 1]}'] = 0
        return groups

    def sort_points_into_groups(self, x_vals: np.ndarray, y_vals: np.ndarray,
                                 x_intervals: List[float], y_intervals: List[float],
                                 groups: Dict[str, int]) -> List[int]:
        """
        Сортировка точек по заданным группам.

        :param x_vals: Значения по оси X.
        :param y_vals: Значения по оси Y.
        :param x_intervals: Границы по оси X.
        :param y_intervals: Границы по оси Y.
        :param groups: Словарь групп.
        :return: Список значений для каждой группы.
        """
        group_values = []
        for x, y in zip(x_vals, y_vals):
            for x_i in range(len(x_intervals) - 1):
                for y_i in range(len(y_intervals) - 1):
                    if (x_intervals[x_i] <= x < x_intervals[x_i + 1] and
                            y_intervals[y_i] <= y < y_intervals[y_i + 1]):
                        groups[f'x : {x_intervals[x_i]} - {x_intervals[x_i + 1]} | '
                               f'y : {y_intervals[y_i]} - {y_intervals[y_i + 1]}'] += 1
        return list(groups.values())

    def add_statistics(self) -> None:
        """Добавление статистики по широте и расстояниям в DataFrame."""
        self.df['mean_lat'], self.df['mean_lon'] = zip(
            *self.df['points'].apply(self.calculate_mean_lat_lon)
        )
        self.df['min_lat'], self.df['max_lat'], self.df['min_lon'], self.df['max_lon'] = zip(
            *self.df['points'].apply(self.calculate_min_max_lat_lon)
        )
        self.df['mean_distance'] = self.df['points'].apply(self.calculate_distances)

    def calculate_mean_lat_lon(self, points: List[Dict[str, float]]) -> Tuple[float, float]:
        """
        Вычисление средней широты и долготы.

        :param points: Список точек с широтой и долготой.
        :return: Средняя широта и долгота.
        """
        lats = [float(point['lat']) for point in points]
        lons = [float(point['lon']) for point in points]
        return np.mean(lats), np.mean(lons)

    def calculate_min_max_lat_lon(self, points: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
        """
        Вычисление минимальной и максимальной широты и долготы.

        :param points: Список точек с широтой и долготой.
        :return: Минимальная и максимальная широта и долгота.
        """
        lats = [float(point['lat']) for point in points]
        lons = [float(point['lon']) for point in points]
        return np.min(lats), np.max(lats), np.min(lons), np.max(lons)

    def calculate_distances(self, points: List[Dict[str, float]]) -> float:
        """
        Вычисление средней дистанции между последовательными точками.

        :param points: Список точек с широтой и долготой.
        :return: Средняя дистанция между точками.
        """
        distances = []
        for i in range(len(points) - 1):
            coord1 = (float(points[i]['lat']), float(points[i]['lon']))
            coord2 = (float(points[i + 1]['lat']), float(points[i + 1]['lon']))
            dist = distance(coord1, coord2).km
            distances.append(dist)
        return np.mean(distances)
    
    def categorize_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Категоризует колонки DataFrame по количеству уникальных значений."""
        
        binary_cols = []
        continuous_cols = []
        few_unique_cols = []
        multi_unique_cols = []

        for col in df.columns:
            unique_values = df[col].nunique()
            
            if unique_values == 2:
                binary_cols.append(col)
            elif 2 < unique_values < 5:
                continuous_cols.append(col)
            elif 4 < unique_values < 10:
                few_unique_cols.append(col)
            elif unique_values >= 10:
                multi_unique_cols.append(col)

        return binary_cols, continuous_cols, few_unique_cols, multi_unique_cols

    def quantile_encode_not_all(self, series: pd.Series) -> pd.Series:
        """Простое квантильное кодирование."""
        return series.rank(pct=True)

    def quantile_encode(self, series: pd.Series) -> pd.Series:
        """Квантильное кодирование с преобразованием в целочисленный формат."""
        return series.rank(pct=True).astype(int)

    def encode_columns(self, method: int) -> pd.DataFrame:
        """
        Кодирование колонок в зависимости от выбранного метода.

        :param method: Метод кодирования (0-5).
        :return: Обработанный DataFrame с закодированными колонками.
        """
        df_encoded = pd.DataFrame()
        self.categorize_columns(df_encoded)

        if method == 0:
            df_encoded = self.df[few_unique_cols].copy()
        elif method in [1, 2]:
            for col in few_unique_cols:
                df_encoded[col] = (self.quantile_encode_not_all if method == 1 else self.quantile_encode)(self.df[col])
        elif method == 3:
            catboost_encoder = ce.CatBoostEncoder()
            for col in few_unique_cols:
                df_encoded[col] = self.quantile_encode(self.df[col])
            df_encoded = catboost_encoder.fit_transform(df_encoded, self.df['value'])
        elif method in [4, 5]:
            for col in few_unique_cols:
                try:
                    df_encoded[col] = self.quantile_encode(self.df[col])
                    df_encoded[col] = pd.Categorical(df_encoded[col]).cat.codes
                except Exception as e:
                    print(f"Error encoding column {col}: {e}")

        return df_encoded    

    def process(self) -> Tuple[Dict[str, int], pd.DataFrame]:
        """
        Основной метод обработки данных.

        :return: Кортеж с группами и обработанным DataFrame.
        """
        self.load_geojson_files()
        self.preprocess_geometries()
        self.compute_distances()
        self.add_statistics()
        groups = self.create_groups()
        return groups, self.df


# Пример использования
# directory_path = ''
file_list = [
    'arhangelskaya_oblast.geojson',
    'base_obl_people.geojson',
    'mun_obr_all_bad.geojson',
    'oopt_gp_point.geojson',
    'oopt_gp_poly.geojson',
    'path_to_hexagons.geojson',
]

config = {
    'min_xval': 64.5,
    'max_xval': 65.5,
    'min_yval': 35.5,
    'max_yval': 37.5,
    'x_ngroups': 3,
    'y_ngroups': 3
}

processor = DatasetProcessor(file_list, config)
groups, processed_df = processor.process()

# Выводим DataFrame
print(processed_df)

