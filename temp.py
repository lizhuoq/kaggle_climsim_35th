import polars as pl
import argparse
from joblib import load, dump
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings

# 忽略所有FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='debug')
parser.add_argument('--debug', action='store_true', help='debug', default=False)
args = parser.parse_args()

nrows = 100 if args.debug else None
use_cols = [f'{c}_{i}' for c in ["state_t", 'state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v'] for i in range(60)] + [
    'pbuf_TAUX', "pbuf_TAUY"
]
# use_cols = [f'{c}_{i}' for c in ["state_t", 'state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v'] for i in range(60)] + [
#     'state_ps'
# ]
feature_scaler: StandardScaler = load("dataset/ClimSim/feature_scaler.joblib")
features: list = feature_scaler.feature_names_in_.tolist()
use_index = [features.index(x) for x in use_cols]
scale_ = feature_scaler.scale_[use_index]
mean_ = feature_scaler.mean_[use_index]

train_set = pl.read_parquet("dataset/ClimSim/train_set.parquet", n_rows=nrows, columns=use_cols + ["sample_id"])
val_set = pl.read_parquet("dataset/ClimSim/val_set.parquet", n_rows=nrows, columns=use_cols + ["sample_id"])
test = pl.read_parquet("dataset/ClimSim/test.parquet", n_rows=nrows, columns=use_cols + ["sample_id"])
df = pl.concat([train_set, val_set, test], how="vertical")

# test = pl.read_parquet("dataset/ClimSim/test.parquet", n_rows=nrows, columns=use_cols + ["sample_id"])

df = pl.DataFrame(
    df[use_cols].to_numpy().astype(np.float64) * scale_ + mean_, schema=use_cols
)


def feature_engineering(data: pl.DataFrame) -> pl.DataFrame:
    res = []
    # 温度、湿度梯度
    for c in ["state_t", 'state_q0001', 'state_q0002', 'state_q0003']:
        df = data[[f"{c}_{i}" for i in range(59, -1, -1)]].to_pandas().diff(axis=1).fillna(method="bfill", axis=1)
        df = pl.DataFrame(df.to_numpy(), schema=[f"{c}_grad_{i}" for i in range(59, -1, -1)])
        df = df[[f"{c}_grad_{i}" for i in range(60)]]
        res.append(df)
    
    # 风速
    df = pl.DataFrame(np.sqrt(data[[f"state_u_{i}" for i in range(60)]].to_numpy() ** 2 + data[[f"state_v_{i}" for i in range(60)]].to_numpy() ** 2), 
                    schema=[f'wind_speed_{i}' for i in range(60)])    
    res.append(df)

    # 风梯度
    df = df[[f'wind_speed_{i}' for i in range(59, -1, -1)]].to_pandas().diff(axis=1).fillna(method='bfill', axis=1)
    df = pl.DataFrame(df.to_numpy(), schema=[f'wind_speed_grad_{i}' for i in range(59, -1, -1)])
    df = df[[f'wind_speed_grad_{i}' for i in range(60)]]
    res.append(df)

    # 总热力通量 = 感热 + 潜热
    df = pl.DataFrame(data[["pbuf_LHFLX"]].to_numpy() + data[["pbuf_SHFLX"]].to_numpy(), schema=["pbuf_LHFLX + pbuf_SHFLX"])
    res.append(df)

    # 有效太阳辐射=太阳入射辐射×cos(太阳天顶角)×(1−地表反照率)
    df = pl.DataFrame(data[['pbuf_SOLIN']].to_numpy() * data[['pbuf_COSZRS']].to_numpy() * (1 - data[['cam_in_ASDIR']].to_numpy()), schema=['effective_solar_radiation'])
    res.append(df)

    return pl.concat(res, how="horizontal")


def feature_engineering_version1(data: pl.DataFrame) -> pl.DataFrame:
    res = []
    res = []
    # 温度、湿度梯度 二阶差分
    for c in ["state_t", 'state_q0001', 'state_q0002', 'state_q0003']:
        df = data[[f"{c}_{i}" for i in range(59, -1, -1)]].to_pandas().diff(axis=1).diff(axis=1).fillna(method="bfill", axis=1)
        df = pl.DataFrame(df.to_numpy(), schema=[f"{c}_n2_grad_{i}" for i in range(59, -1, -1)])
        df = df[[f"{c}_n2_grad_{i}" for i in range(60)]]
        res.append(df)

    # 风速
    df = pl.DataFrame(np.sqrt(data[[f"state_u_{i}" for i in range(60)]].to_numpy() ** 2 + data[[f"state_v_{i}" for i in range(60)]].to_numpy() ** 2), 
                    schema=[f'wind_speed_{i}' for i in range(60)])
    
    # 风梯度 二阶差分
    df = df[[f'wind_speed_{i}' for i in range(59, -1, -1)]].to_pandas().diff(axis=1).diff(axis=1).fillna(method='bfill', axis=1)
    df = pl.DataFrame(df.to_numpy(), schema=[f'wind_speed_n2_grad_{i}' for i in range(59, -1, -1)])
    df = df[[f'wind_speed_n2_grad_{i}' for i in range(60)]]
    res.append(df)

    # 表面应力
    df = pl.DataFrame(np.sqrt(data[["pbuf_TAUX"]].to_numpy() ** 2 + data[["pbuf_TAUY"]].to_numpy() ** 2), schema=["surface stress"])
    res.append(df)

    return pl.concat(res, how="horizontal")


# df = feature_engineering(df)
df = feature_engineering_version1(df)
feature_scaler = StandardScaler()
feature_scaler.fit(df)
# dump(feature_scaler, "dataset/ClimSim/feature_engineering_feature_scaler.joblib")

# feature_scaler: StandardScaler = load("dataset/ClimSim/feature_engineering_feature_scaler.joblib")
features = feature_scaler.feature_names_in_.tolist()
dump(feature_scaler, "dataset/ClimSim/feature_engineering_version1_feature_scaler.joblib")
df = pl.DataFrame(feature_scaler.transform(df[features].to_numpy()).astype(np.float32), schema=features)
# df = pl.concat([test[["sample_id"]], df], how="horizontal")

train_out = pl.concat([train_set[["sample_id"]], df[:len(train_set)]], how="horizontal")
val_out = pl.concat([val_set[["sample_id"]], df[len(train_set): len(train_set) + len(val_set)]], how="horizontal")
test_out = pl.concat([test[["sample_id"]], df[len(train_set) + len(val_set):]], how="horizontal")

# train_out.write_parquet("dataset/ClimSim/feature_engineering_train_set.parquet")
# val_out.write_parquet("dataset/ClimSim/feature_engineering_val_set.parquet")
# test_out.write_parquet("dataset/ClimSim/feature_engineering_test.parquet")

train_out.write_parquet("dataset/ClimSim/feature_engineering_version1_train_set.parquet")
val_out.write_parquet("dataset/ClimSim/feature_engineering_version1_val_set.parquet")
test_out.write_parquet("dataset/ClimSim/feature_engineering_version1_test.parquet")

# df.write_parquet("dataset/ClimSim/feature_engineering_test.parquet")