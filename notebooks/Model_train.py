import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce
from typing import Tuple, List
from catboost import Pool

class ModelPipeline:
    def __init__(self, geojson_files: List[str], config: dict):
        self.processor = DatasetProcessor(geojson_files, config)
        self.model = CustomRegressor(
            iterations=2500,
            max_depth=6,
            learning_rate=0.05,
            random_seed=42,
            loss_function='RMSE',
            eval_metric='RMSE',
            verbose=100
        )

    def train(self, df: pd.DataFrame):
        """Обучение модели."""
        groups, processed_df = self.processor.process()
        y = processed_df[self.processor.config['target']]
        X = processed_df.drop(columns=[self.processor.config['target']])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)

        self.model.fit(train_pool, eval_set=test_pool, use_best_model=True, plot=True)

        self.save_model('model_weights.cbm')

        return self.model, X_test, y_test

    def save_model(self, model_path: str):
        """Сохранение весов модели."""
        self.model.save_model(model_path)

# Пример конфигурации
config = {
    'min_xval': 64.5,
    'max_xval': 65.5,
    'min_yval': 35.5,
    'max_yval': 37.5,
    'x_ngroups': 3,
    'y_ngroups': 3,
}

# Использование пайплайна
directory_path = 'path_to_your_data_directory'  # Замените на ваш путь к данным
file_list = [
    'arhangelskaya_oblast.geojson',
    'base_obl_people.geojson',
    'mun_obr_all_bad.geojson',
    'oopt_gp_point.geojson',
    'oopt_gp_poly.geojson',
    'path_to_hexagons.geojson',
]

pipeline = ModelPipeline(file_list, config)

model, X_test, y_test = pipeline.train(pipeline.processor.df)

