import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch 
from typing import Tuple, List
from catboost import Pool
import Preprocessor

class ModelfClassification(nn.Module):
    """Модель для классификации"""
    def __init__(self, input_size, num_layers=4, output_size=2, dropout_prob=0.2):
        super(ModelfClassification, self).__init__()
        self.num_layers = num_layers
        self.conv1d = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(input_size, 128)
        self.batchnorm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.sigm = nn.Sigmoid()
        
        
    def forward(self, x):
        out = self.fc1(out)
        out = self.batchnorm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigm(out)
        return out
    
class ModelfRegression():
    """Модель для оценки привлекательности"""
    def __init__(self, embeddings_array, target_array):
        self.catboost_model = CatBoostRegressor(iterations=2000, depth=7, learning_rate=0.05, task_type="GPU",  devices='0:1',  loss_function='RMSE', eval_metric='RMSE', random_state=42)
        self.catboost_model.fit(embeddings_array, target_array)

    def predict(self, X):
        return catboost_model.predict(X_test)
    

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


