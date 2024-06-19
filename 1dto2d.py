import polars as pl
from joblib import load
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='debug')
parser.add_argument('--debug', action='store_true', help='debug', default=False)
args = parser.parse_args()

nrows = 384 if args.debug else None

df_train = pl.read_parquet("dataset/ClimSim/train_set.parquet", n_rows=nrows)
df_val = pl.read_parquet("dataset/ClimSim/val_set.parquet", n_rows=nrows)
df = pl.concat([df_train, df_val], how="vertical")

df = df.with_columns(
    pl.col('sample_id').str.extract(r'train_(\d+)', 1).cast(pl.Int64).alias('id_numeric')
)
df = df.sort("id_numeric")
print(df[['id_numeric']].head(), df[['id_numeric']].tail())
df = df.drop("id_numeric")
feature_scaler: StandardScaler = load("dataset/ClimSim/feature_scaler.joblib")
features: list = feature_scaler.feature_names_in_.tolist()
target_scaler: StandardScaler = load("dataset/ClimSim/target_scaler.joblib")
targets: list = target_scaler.feature_names_in_.tolist()
pbar = tqdm(total=len(df) // 384)
save_dir = "dataset/ClimSim_2D"
os.makedirs(save_dir, exist_ok=True)

for i, chunk in enumerate(df.iter_slices(384)):
    # sample_id = chunk["sample_id"].to_numpy()
    x = chunk[features].to_numpy()
    level_x = x[:, :9 * 60].reshape(-1, 9, 60)
    single_x = x[:, 9 * 60:]
    y = chunk[targets].to_numpy()
    level_y = y[:, :6 * 60].reshape(-1, 6, 60)
    single_y = y[:, 6 * 60:]
    np.savez(os.path.join(save_dir, f"{i}.npz"), level_x=level_x, single_x=single_x, level_y=level_y, single_y=single_y)
    pbar.update()

pbar.close()