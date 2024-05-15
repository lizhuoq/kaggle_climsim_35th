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


class ClimSim1D(Dataset):
    def __init__(self, args):
        self.args = args
        self.features = [f"{x}_{l}" for x in ["state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", 
                      "state_v", "pbuf_ozone", "pbuf_CH4", "pbuf_N2O"] for l in range(60)] + [
            "cam_in_SNOWHLAND", "cam_in_OCNFRAC", "cam_in_LANDFRAC", "cam_in_ICEFRAC", "cam_in_LWUP", 
            "cam_in_ASDIR", "cam_in_ASDIF", "cam_in_ALDIR", "cam_in_ALDIF", "pbuf_COSZRS", "pbuf_TAUY", 
            "pbuf_TAUX", "pbuf_SHFLX", "pbuf_LHFLX", "pbuf_SOLIN", "state_ps"
        ]
        self.targets = [f"{x}_{l}" 
                for x in ["ptend_t", "ptend_q0001", "ptend_q0002", "ptend_q0003", "ptend_u", 
                            "ptend_v"] for l in range(60)] + [
            "cam_out_NETSW", "cam_out_FLWDS", "cam_out_PRECSC", "cam_out_PRECC", "cam_out_SOLS", "cam_out_SOLL", 
            "cam_out_SOLSD", "cam_out_SOLLD"
        ]
        self.feature_scaler: StandardScaler = load(args.feature_scaler_path)
        self.target_scaler: StandardScaler = load(args.target_scaler_path)
        assert self.feature_scaler.feature_names_in_.tolist() == self.features
        assert self.target_scaler.feature_names_in_.tolist() == self.targets
        self.__load_data__()
        self.__load_weight__()
        
    def __load_data__(self):
        df = pl.read_parquet(os.path.join(self.args.root_path, self.args.data_path))
        df_x, df_y = df[self.features], df[self.targets]
        data_x = self.feature_scaler.transform(df_x.to_numpy())
        data_y = self.target_scaler.transform(df_y.to_numpy())
        self.x_seq, self.x, self.y_seq, self.y = self.__to1d__(data_x, data_y)
        
    def __getitem__(self, index):
        return self.x_seq[index], self.x[index], self.y_seq[index], self.y[index]
    
    def __len__(self):
        return len(self.x_seq)
    
    def __to1d__(self, data_x, data_y):
        x_seq = data_x[:, :9 * 60].reshape(-1, 9, 60)
        x = data_x[:, 9 * 60:]
        y_seq = data_y[:, :6 * 60].reshape(-1, 6, 60)
        y = data_y[:, 6 * 60:]
        return x_seq, x, y_seq, y
        
    def inverse_transform(self, data_y):
        return self.target_scaler.inverse_transform(data_y)
    
    def __load_weight__(self):
        df_weight = pl.read_parquet(self.args.weight_path)
        self.weight = df_weight[self.targets].to_numpy()
    
    def multiply_weight(self, data_y):
        return data_y * self.weight
    

class ClimSimTabular(Dataset):
    def __init__(self, args):
        self.args = args
        self.features = [f"{x}_{l}" for x in ["state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", 
                      "state_v", "pbuf_ozone", "pbuf_CH4", "pbuf_N2O"] for l in range(60)] + [
            "cam_in_SNOWHLAND", "cam_in_OCNFRAC", "cam_in_LANDFRAC", "cam_in_ICEFRAC", "cam_in_LWUP", 
            "cam_in_ASDIR", "cam_in_ASDIF", "cam_in_ALDIR", "cam_in_ALDIF", "pbuf_COSZRS", "pbuf_TAUY", 
            "pbuf_TAUX", "pbuf_SHFLX", "pbuf_LHFLX", "pbuf_SOLIN", "state_ps"
        ]
        self.targets = [f"{x}_{l}" 
                for x in ["ptend_t", "ptend_q0001", "ptend_q0002", "ptend_q0003", "ptend_u", 
                            "ptend_v"] for l in range(60)] + [
            "cam_out_NETSW", "cam_out_FLWDS", "cam_out_PRECSC", "cam_out_PRECC", "cam_out_SOLS", "cam_out_SOLL", 
            "cam_out_SOLSD", "cam_out_SOLLD"
        ]
        self.feature_scaler: StandardScaler = load(args.feature_scaler_path)
        self.target_scaler: StandardScaler = load(args.target_scaler_path)
        assert self.feature_scaler.feature_names_in_.tolist() == self.features
        assert self.target_scaler.feature_names_in_.tolist() == self.targets
        self.__load_data__()
        self.__load_weight__()
        if self.args.simplify_features:
            self.__simplify_features__()
            self.data_x = self.data_x[:, self.simple_feature_index]
        if self.args.simplify_targets:
            self.__simplify_targets__()
            self.data_y = self.data_y[:, self.simple_target_index]
        
    def __load_data__(self):
        df = pl.read_parquet(os.path.join(self.args.root_path, self.args.data_path))
        df_x, df_y = df[self.features], df[self.targets]
        self.data_x = self.feature_scaler.transform(df_x.to_numpy())
        self.data_y = self.target_scaler.transform(df_y.to_numpy())
        
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)
        
    def inverse_transform(self, data_y):
        if self.args.simplify_targets:
            std = self.target_scaler.scale_[self.simple_target_index]
            mean = self.target_scaler.mean_[self.simple_target_index]
            return data_y * std + mean
        return self.target_scaler.inverse_transform(data_y)
    
    def __load_weight__(self):
        df_weight = pl.read_parquet(self.args.weight_path)
        self.weight = df_weight[self.targets].to_numpy()
    
    def multiply_weight(self, data_y):
        if self.args.simplify_targets:
            return data_y * self.weight[:, self.simple_target_index]
        return data_y * self.weight
    
    def __simplify_features__(self):
        df = pd.read_csv(self.args.feature_importance_path)
        features = df[df["feature_importance"] <= self.args.fi_threshold]["feature_name"].to_list()
        self.simple_feature_index = [self.features.index(x) for x in features]

    def __simplify_targets__(self):
        self.simple_target_index = np.nonzero(self.weight[0] != 0)[0].tolist()


class ClimSimSmall(Dataset):
    def __init__(self, args):
        self.args = args
        self.__load_scaler__()
        self.__load_data__()
        self.__load_weight__()

    def __load_scaler__(self):
        self.feature_scaler: StandardScaler = load(self.args.feature_scaler_path)
        self.target_scaler: StandardScaler = load(self.args.target_scaler_path)
        self.features = self.feature_scaler.feature_names_in_.tolist()
        self.targets = self.target_scaler.feature_names_in_.tolist()
        
    def __load_data__(self):
        df = pl.read_parquet(os.path.join(self.args.root_path, self.args.data_path))
        df_x, df_y = df[self.features], df[self.targets]
        self.data_x = self.feature_scaler.transform(df_x.to_numpy())
        self.data_y = self.target_scaler.transform(df_y.to_numpy())

    def __load_weight__(self):
        df_weight = pl.read_parquet(self.args.weight_path)
        self.weight = df_weight[self.targets].to_numpy()
        
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)
        
    def inverse_transform(self, data_y):
        return self.target_scaler.inverse_transform(data_y)
    
    def multiply_weight(self, data_y):
        return data_y * self.weight