import os
import numpy as np
import pandas as pd
import polars as pl
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
# from data_provider.uea import subsample, interpolate_missing, Normalizer
# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
# from utils.augmentation import run_augmentation_single
from joblib import load

warnings.filterwarnings('ignore')


class ClimSimSeq2Seq(Dataset):
    def __init__(self, args):
        self.args = args
        self.__load_scaler__()
        self.__load_data__()
        self.__load_weight__()

    def __load_scaler__(self):
        self.feature_scaler: StandardScaler = load("/data/home/scv7343/run/climsim_new/dataset/ClimSim/feature_scaler.joblib")
        self.target_scaler: StandardScaler = load("/data/home/scv7343/run/climsim_new/dataset/ClimSim/target_scaler.joblib")
        self.features = self.feature_scaler.feature_names_in_.tolist()
        self.targets = self.target_scaler.feature_names_in_.tolist()
        
    def __load_data__(self):
        nrows = 100_0000 if self.args.debug else None
        df = pl.read_parquet(os.path.join(self.args.root_path, self.args.data_path), n_rows=nrows)
        if self.args.sample_rate is not None:
            df = df.sample(fraction=self.args.sample_rate, shuffle=True, seed=2024)
        df_x, df_y = df[self.features], df[self.targets]
        self.data_x = df_x.to_numpy()
        self.data_y = df_y.to_numpy()
        
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)
        
    def inverse_transform(self, data_y: np.ndarray):
        return self.target_scaler.inverse_transform(data_y.astype(np.float64))
    
    def __load_weight__(self):
        df_weight = pl.read_parquet("/data/home/scv7343/run/climsim_new/dataset/ClimSim/sample_submission.parquet")
        self.weight = df_weight[self.targets].to_numpy()
    
    def multiply_weight(self, data_y):
        return data_y * self.weight