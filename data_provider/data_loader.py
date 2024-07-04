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
from sklearn.model_selection import train_test_split

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
        if self.args.add_feature_engineering:
            feature_scaler: StandardScaler = load("dataset/ClimSim/feature_engineering_feature_scaler.joblib")
            fe_features = feature_scaler.feature_names_in_.tolist()
        if self.args.add_fe_v1:
            fe_v1_scaler : StandardScaler = load("dataset/ClimSim/feature_engineering_version1_feature_scaler.joblib")
            fe_v1_features = fe_v1_scaler.feature_names_in_.tolist()
        self.features = self.feature_scaler.feature_names_in_.tolist()
        self.targets: list = self.target_scaler.feature_names_in_.tolist()
        if self.args.add_feature_engineering:
            self.features = self.features[:9 * 60] + fe_features[:6 * 60] + fe_features[6 * 60:] + self.features[9 * 60:]
            if self.args.add_fe_v1:
                self.features = self.features[:15 * 60] + fe_v1_features[:5 * 60] + self.features[15 * 60:] + fe_v1_features[5 * 60:]
        else:
            if self.args.add_fe_v1:
                self.features = self.features[:9 * 60] + fe_v1_features[:5 * 60] + self.features[9 * 60:] + fe_v1_features[5 * 60:]
        
    def __load_data__(self):
        nrows = None
        df = pl.read_parquet(os.path.join(self.args.root_path, self.args.data_path), n_rows=nrows)
        if self.args.add_feature_engineering:
            df_fe = pl.read_parquet(os.path.join(self.args.root_path, f"feature_engineering_{self.args.data_path}"), n_rows=nrows).drop("sample_id")
            df = pl.concat([df, df_fe], how="horizontal")
        if self.args.add_fe_v1:
            df_fev1 = pl.read_parquet(os.path.join(self.args.root_path, f"feature_engineering_version1_{self.args.data_path}"), n_rows=nrows).drop("sample_id")
            df = pl.concat([df, df_fev1], how="horizontal")
        if self.args.add_val_data:
            val_df = df
        if self.args.sample_rate is not None:
            df = df.sample(fraction=self.args.sample_rate, shuffle=True, seed=2024)
        else:
            if self.args.data_path.startswith("val"):
                df = df.sample(fraction=0.31, shuffle=True, seed=2024) # 625_0000
        if self.args.add_val_data:
            df = pl.concat([val_df, df], how="vertical")
            df = df.filter(df.is_unique())
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
    
    def get_no_grad_targets(self) -> list:
        targets = [f"ptend_q0002_{i}" for i in range(27)]
        no_grad_index = [self.targets.index(x) for x in targets]
        return no_grad_index
    
    def get_last_eight_std_mean(self):
        scale_ = self.target_scaler.scale_[-8:]
        mean_ = self.target_scaler.mean_[-8:]
        return scale_, mean_
    

class ClimSim2D(Dataset):
    def __init__(self, args, flag="train") -> None:
        super().__init__()
        self.args = args
        filenames = os.listdir(self.args.root_path)
        train_filenames, val_filenames = train_test_split(filenames, test_size=0.2, random_state=2024, shuffle=True)
        self.filenames = train_filenames if flag == "train" else val_filenames
        self.__load_scaler__()
        self.__load_weight__()

    def __getitem__(self, index):
        data = np.load(os.path.join(self.args.root_path, self.filenames[index]))
        single_x = data["single_x"]
        single_y = data["single_y"]
        level_x = data["level_x"]
        level_y = data["level_y"]
        return level_x, single_x, level_y, single_y

    def __len__(self):
        return len(self.filenames)
    
    def __load_scaler__(self):
        self.feature_scaler: StandardScaler = load("/data/home/scv7343/run/climsim_new/dataset/ClimSim/feature_scaler.joblib")
        self.target_scaler: StandardScaler = load("/data/home/scv7343/run/climsim_new/dataset/ClimSim/target_scaler.joblib")
        self.features = self.feature_scaler.feature_names_in_.tolist()
        self.targets: list = self.target_scaler.feature_names_in_.tolist()

    def __load_weight__(self):
        df_weight = pl.read_parquet("/data/home/scv7343/run/climsim_new/dataset/ClimSim/sample_submission.parquet")
        self.weight = df_weight[self.targets].to_numpy()

    def get_no_grad_targets(self) -> list:
        targets = [f"ptend_q0002_{i}" for i in range(27)]
        no_grad_index = [self.targets.index(x) for x in targets]
        return no_grad_index
    
    def get_last_eight_std_mean(self):
        scale_ = self.target_scaler.scale_[-8:]
        mean_ = self.target_scaler.mean_[-8:]
        return scale_, mean_
    
    def multiply_weight(self, data_y):
        return data_y * self.weight
    
    def inverse_transform(self, data_y: np.ndarray):
        return self.target_scaler.inverse_transform(data_y.astype(np.float64))