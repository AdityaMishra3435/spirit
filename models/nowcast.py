import json
import pandas as pd
from datasets import load_dataset
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb
import numpy as np
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
SCRATCH_DIR = config["SCRATCH_DIR"]
TRAIN_DATASET_LINK = config["TRAIN_DATASET_LINK"]
TRAIN_EMBEDDINGS_FILE = SCRATCH_DIR + config["TRAIN_EMBEDDINGS_FILE"]
STUDY_NAME = config["STUDY_NAME"]
best_val_nmap = float("inf")
best_model_path = SCRATCH_DIR + "nowcast_model.json"
train_dataset = load_dataset(TRAIN_DATASET_LINK, split="train")
train_dataset_df = pd.DataFrame(train_dataset)
X_train_image = []
with open(TRAIN_EMBEDDINGS_FILE, "r") as file:
    for line in file:
        entry = json.loads(line)
        embedding = entry["embedding"]
        X_train_image.append(embedding)

X_train_image = np.array(X_train_image)
X_train_auxiliary = []
y_train_ghi = []
for index in range(len(train_dataset_df)):
    zenith_angle = train_dataset_df["Zenith_angle"][index]
    azimuth_angle = train_dataset_df["Azimuth_angle"][index]
    panel_tilt = train_dataset_df["physics_panel_tilt"][index]
    panel_orientation = train_dataset_df["physics_panel_orientation"][index]
    aoi = train_dataset_df["physics_aoi"][index]
    auxiliary_data = [
        train_dataset_df["Clear_sky_ghi"][index],
        zenith_angle,
        azimuth_angle,
        panel_tilt,
        panel_orientation,
        aoi,
        train_dataset_df["physics_total_irradiance_tilted"][index],
        np.cos(zenith_angle),
        np.sin(zenith_angle),
        np.cos(azimuth_angle),
        np.sin(azimuth_angle),
        np.cos(panel_tilt),
        np.sin(panel_tilt),
        np.cos(panel_orientation),
        np.sin(panel_orientation),
        np.cos(aoi),
        np.sin(aoi),
    ]
    X_train_auxiliary.append(auxiliary_data)
    y_train_ghi.append(train_dataset_df["Global_horizontal_irradiance"][index])
X_train_auxiliary = np.array(X_train_auxiliary)
y_train_ghi = np.array(y_train_ghi).reshape(-1, 1)


def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def global_normalization(y):
    global_mean = np.mean(y)
    global_std = np.std(y)
    normalized_y = (y - global_mean) / global_std
    return normalized_y, global_mean, global_std


X_train_auxiliary, X_train_mean, X_train_std = normalize_features(X_train_auxiliary)
y_train_ghi, y_train_mean, y_train_std = global_normalization(y_train_ghi)
train_indices = np.arange(len(train_dataset))
X_train_concat = np.concatenate(
    [
        X_train_image,
        X_train_auxiliary,
    ],
    axis=1,
)
X_train = X_train_concat
y_train = y_train_ghi


def evaluate_function(y_true, y_pred):
    y_true = y_true * y_train_std + y_train_mean
    y_pred = y_pred * y_train_std + y_train_mean
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    nmap = mae / np.mean(y_true) * 100
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
        "nMAP": nmap,
    }
    return metrics


def train(X_train, y_train, param):
    global best_val_nmap
    model = xgb.XGBRegressor(**param)
    early_stopping_rounds = 200
    model.set_params(
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=early_stopping_rounds,
    )
    X_k_train, X_k_val, y_k_train, y_k_val = train_test_split(
        X_train, y_train, test_size=0.2
    )
    evals = [(X_k_train, y_k_train), (X_k_val, y_k_val)]
    model.fit(
        X_k_train,
        y_k_train,
        eval_set=evals,
        verbose=False,
    )
    y_k_val_pred = model.predict(X_k_val)
    val_metrics = evaluate_function(y_true=y_k_val, y_pred=y_k_val_pred)
    val_nmap = val_metrics["nMAP"]
    if val_nmap < best_val_nmap:
        best_val_nmap = val_nmap
        model.save_model(best_model_path)
        print(f"New best model saved with nMAP: {best_val_nmap:.4f}")
    return val_metrics


def objective(trial):
    param = {
        "booster": "gbtree",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 8),
        "lambda": trial.suggest_float("lambda", 0, 8),
    }
    val_metrics = train(X_train, y_train, param)
    return val_metrics["nMAP"]


pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10, n_warmup_steps=10, interval_steps=1
)
study = optuna.create_study(
    direction="minimize",
    pruner=pruner,
    study_name=STUDY_NAME,
    storage=f"sqlite:///./database_{STUDY_NAME}.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=1000, show_progress_bar=True)
best_params = study.best_trial.params
print("Best trial hyperparameters:", best_params)
