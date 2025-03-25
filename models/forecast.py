import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
import os
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
SCRATCH_DIR = config["SCRATCH_DIR"]
NUM_PAST_FRAMES = config["NUM_PAST_FRAMES"]
NUM_FUTURE_FRAMES = config["NUM_FUTURE_FRAMES"]
FEATURE_DIM = config["FEATURE_DIM"]
HIDDEN_DIM = config["HIDDEN_DIM"]
NUM_COVARIATE_FEATURES = config["NUM_COVARIATE_FEATURES"]
NUM_RESIDUAL_BLOCKS = config["NUM_RESIDUAL_BLOCKS"]
NUM_ENCODER_LAYERS = config["NUM_ENCODER_LAYERS"]
INIT_LEARNING_RATE = config["INIT_LEARNING_RATE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
NUM_ATTENTION_HEADS = config["NUM_ATTENTION_HEADS"]
BATCH_SIZE = config["BATCH_SIZE"]
WARMUP_STEPS = config["WARMUP_STEPS"]
TOTAL_STEPS = config["TOTAL_STEPS"]
STUDY_NAME = config["STUDY_NAME"]
TRAIN_DATASET_LINKS = config["TRAIN_DATASET_LINKS"]
TRAIN_EMBEDDINGS_FILES = [
    SCRATCH_DIR + file for file in config["TRAIN_EMBEDDINGS_FILES"]
]
TEST_DATASET_LINKS = config["TEST_DATASET_LINKS"]
TEST_EMBEDDINGS_FILES = [SCRATCH_DIR + file for file in config["TEST_EMBEDDINGS_FILES"]]

# create dataset
train_dataset_df = pd.DataFrame()
for train_dataset_link in TRAIN_DATASET_LINKS:
    train_dataset = load_dataset(
        train_dataset_link, split="train", cache_dir=SCRATCH_DIR
    )
    train_dataset_df = pd.concat(
        [train_dataset_df, pd.DataFrame(train_dataset)], ignore_index=True
    )
train_embeddings = []
for embeddings_file in TRAIN_EMBEDDINGS_FILES:
    with open(embeddings_file, "r") as file:
        for line in file:
            entry = json.loads(line)
            embedding = entry["embedding"]
            train_embeddings.append(embedding)
test_dataset_df = pd.DataFrame()
for test_dataset_link in TEST_DATASET_LINKS:
    test_dataset = load_dataset(test_dataset_link, split="train", cache_dir=SCRATCH_DIR)
    test_dataset_df = pd.concat(
        [test_dataset_df, pd.DataFrame(test_dataset)], ignore_index=True
    )
test_embeddings = []
for embeddings_file in TEST_EMBEDDINGS_FILES:
    with open(embeddings_file, "r") as file:
        for line in file:
            entry = json.loads(line)
            embedding = entry["embedding"]
            test_embeddings.append(embedding)


def generate_timeseries_data(dataset_df, embeddings):
    timeseries_data = []
    len_dataset = len(embeddings)
    for i in range(len_dataset - NUM_PAST_FRAMES - NUM_FUTURE_FRAMES):
        if (
            dataset_df.iloc[i]["DATE"]
            != dataset_df.iloc[i + NUM_PAST_FRAMES + NUM_FUTURE_FRAMES - 1]["DATE"]
        ):
            continue

        selected_embeddings = [
            np.array(embeddings[j]) for j in range(i, i + NUM_PAST_FRAMES)
        ]

        future_covariate_vector = []
        for j in range(NUM_FUTURE_FRAMES):
            future_covariate_vector.append(
                dataset_df.iloc[i + NUM_PAST_FRAMES + j]["Clear_sky_ghi"]
            )
            future_covariate_vector.append(
                dataset_df.iloc[i + NUM_PAST_FRAMES + j]["Clear_sky_dni"]
            )
            future_covariate_vector.append(
                dataset_df.iloc[i + NUM_PAST_FRAMES + j][
                    "physics_total_irradiance_tilted"
                ]
            )
            future_covariate_vector.append(
                dataset_df.iloc[i + NUM_PAST_FRAMES + j]["physics_panel_tilt"]
            )
            future_covariate_vector.append(
                dataset_df.iloc[i + NUM_PAST_FRAMES + j]["physics_aoi"]
            )
        future_clear_sky_ghi = [
            dataset_df.iloc[i + NUM_PAST_FRAMES + j]["Clear_sky_ghi"]
            for j in range(NUM_FUTURE_FRAMES)
        ]
        target_actual_ghi = [
            dataset_df.iloc[i + NUM_PAST_FRAMES + j]["Global_horizontal_irradiance"]
            for j in range(NUM_FUTURE_FRAMES)
        ]
        timeseries_data.append(
            (
                selected_embeddings,
                future_covariate_vector,
                future_clear_sky_ghi,
                target_actual_ghi,
            )
        )
    return timeseries_data


def process_timeseries_data(timeseries_data, isTraining):
    if isTraining:
        X, X_future_covariate, X_future_clear_sky, y_ghi = zip(*timeseries_data)
        (
            X,
            X_val,
            X_future_covariate,
            X_future_covariate_val,
            X_future_clear_sky,
            X_future_clear_sky_val,
            y_ghi,
            y_ghi_val,
        ) = train_test_split(
            list(X),
            list(X_future_covariate),
            list(X_future_clear_sky),
            list(y_ghi),
            test_size=0.2,
        )
        X = np.array(X)
        X_future_covariate = np.array(X_future_covariate)
        X_future_clear_sky = np.array(X_future_clear_sky)
        y_ghi = np.array(y_ghi)
        X_val = np.array(X_val)
        X_future_covariate_val = np.array(X_future_covariate_val)
        X_future_clear_sky_val = np.array(X_future_clear_sky_val)
        y_ghi_val = np.array(y_ghi_val)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        X_future_covariate_tensor = torch.tensor(
            X_future_covariate, dtype=torch.float32
        )
        X_future_covariate_val_tensor = torch.tensor(
            X_future_covariate_val, dtype=torch.float32
        )
        X_future_clear_sky_tensor = torch.tensor(
            X_future_clear_sky, dtype=torch.float32
        )
        X_future_clear_sky_val_tensor = torch.tensor(
            X_future_clear_sky_val, dtype=torch.float32
        )
        y_ghi_tensor = torch.tensor(y_ghi, dtype=torch.float32)
        y_ghi_val_tensor = torch.tensor(y_ghi_val, dtype=torch.float32)
        return {
            "X_train_tensor": X_tensor,
            "X_val_tensor": X_val_tensor,
            "X_future_covariate_train_tensor": X_future_covariate_tensor,
            "X_future_covariate_val_tensor": X_future_covariate_val_tensor,
            "X_future_clear_sky_train_tensor": X_future_clear_sky_tensor,
            "X_future_clear_sky_val_tensor": X_future_clear_sky_val_tensor,
            "y_ghi_train_tensor": y_ghi_tensor,
            "y_ghi_val_tensor": y_ghi_val_tensor,
        }

    else:
        X, X_future_covariate, X_future_clear_sky, y_ghi = zip(*timeseries_data)
        X = np.array(X)
        X_future_covariate = np.array(X_future_covariate)
        X_future_clear_sky = np.array(X_future_clear_sky)
        y_ghi = np.array(y_ghi)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_future_covariate_tensor = torch.tensor(
            X_future_covariate, dtype=torch.float32
        )
        X_future_clear_sky_tensor = torch.tensor(
            X_future_clear_sky, dtype=torch.float32
        )
        y_ghi_tensor = torch.tensor(y_ghi, dtype=torch.float32)
        return {
            "X_test_tensor": X_tensor,
            "X_future_covariate_test_tensor": X_future_covariate_tensor,
            "X_future_clear_sky_test_tensor": X_future_clear_sky_tensor,
            "y_ghi_test_tensor": y_ghi_tensor,
        }


def save_model(model, optimizer, scheduler, epoch, best_val_loss, save_dir, model_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    checkpoint_path = os.path.join(save_dir, f"{model_name}")
    torch.save(checkpoint, checkpoint_path)


def load_model(model, optimizer, scheduler, load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, scheduler, checkpoint["epoch"], checkpoint["best_val_loss"]


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x + residual


class TransformerEncoderResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.initial_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, NUM_PAST_FRAMES + 1, HIDDEN_DIM)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_ATTENTION_HEADS,
            dim_feedforward=4 * HIDDEN_DIM,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_ENCODER_LAYERS
        )
        self.concatenation_fc_layer = nn.Linear(
            HIDDEN_DIM + NUM_COVARIATE_FEATURES * NUM_FUTURE_FRAMES + NUM_FUTURE_FRAMES,
            HIDDEN_DIM,
        )
        self.residual_mlp = nn.Sequential(
            *[ResidualMLP(HIDDEN_DIM) for _ in range(NUM_RESIDUAL_BLOCKS)]
        )
        self.final_fc_layer = nn.Linear(HIDDEN_DIM, NUM_FUTURE_FRAMES)

    def forward(self, x, future_covariate, future_clearsky_ghi):
        B, _, _ = x.shape
        x = self.input_projection(x)
        initial_token = self.initial_token.expand(B, -1, -1).to(device)
        x = torch.cat([initial_token, x], dim=1)
        x = x + self.pos_embedding.to(device)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = torch.cat([x, future_covariate, future_clearsky_ghi], dim=-1)
        x = self.concatenation_fc_layer(x)
        x = self.residual_mlp(x)
        x = self.final_fc_layer(x)
        x = x + future_clearsky_ghi
        return x


class BaselineTimeseriesPredictor:
    def __init__(self):
        self.study_name = STUDY_NAME

    def evaluate_function(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        nmap = mae / np.mean(y_true)
        errors = y_true - y_pred
        std_dev = np.std(errors)
        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "nMAP": nmap,
            "standard_deviation": std_dev,
        }
        return metrics

    def get_scheduler(self, optimizer):
        warmup_steps = WARMUP_STEPS
        total_steps = TOTAL_STEPS

        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=0
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        return scheduler

    def train_model(self, train_tensors, trial):
        X_train_tensor = train_tensors["X_train_tensor"]
        X_future_covariate_train_tensor = train_tensors[
            "X_future_covariate_train_tensor"
        ]
        X_future_clear_sky_train_tensor = train_tensors[
            "X_future_clear_sky_train_tensor"
        ]
        y_ghi_train_tensor = train_tensors["y_ghi_train_tensor"]
        X_val_tensor = train_tensors["X_val_tensor"]
        X_future_covariate_val_tensor = train_tensors["X_future_covariate_val_tensor"]
        X_future_clear_sky_val_tensor = train_tensors["X_future_clear_sky_val_tensor"]
        y_ghi_val_tensor = train_tensors["y_ghi_val_tensor"]
        train_tensor_dataset = TensorDataset(
            X_train_tensor,
            X_future_covariate_train_tensor,
            X_future_clear_sky_train_tensor,
            y_ghi_train_tensor,
        )
        train_loader = DataLoader(
            train_tensor_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_tensor_dataset = TensorDataset(
            X_val_tensor,
            X_future_covariate_val_tensor,
            X_future_clear_sky_val_tensor,
            y_ghi_val_tensor,
        )
        val_loader = DataLoader(
            val_tensor_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        model = TransformerEncoderResidualMLP().to(device)
        criterion = nn.L1Loss()
        optimizer = SGD(model.parameters(), lr=INIT_LEARNING_RATE, momentum=0.9)
        scheduler = self.get_scheduler(optimizer)
        best_val_loss = float("inf")
        for epoch in range(NUM_EPOCHS):
            # train loop
            model.train()
            train_loss = 0.0
            for batch_idx, (
                X_batch,
                X_future_covariate_batch,
                X_future_clear_sky_batch,
                y_ghi_batch,
            ) in enumerate(train_loader):
                (
                    X_batch,
                    X_future_covariate_batch,
                    X_future_clear_sky_batch,
                    y_ghi_batch,
                ) = (
                    X_batch.to(device),
                    X_future_covariate_batch.to(device),
                    X_future_clear_sky_batch.to(device),
                    y_ghi_batch.to(device),
                )
                optimizer.zero_grad()
                y_pred = model(
                    X_batch, X_future_covariate_batch, X_future_clear_sky_batch
                )
                loss = criterion(y_pred, y_ghi_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                train_loss += loss.item()
                optimizer.step()
                scheduler.step()
            train_loss /= len(train_loader)
            # validation loop
            model.eval()
            val_loss = 0.0
            y_val_true, y_val_pred = [], []
            with torch.no_grad():
                for (
                    X_batch,
                    X_future_covariate_batch,
                    X_future_clear_sky_batch,
                    y_ghi_batch,
                ) in val_loader:
                    (
                        X_batch,
                        X_future_covariate_batch,
                        X_future_clear_sky_batch,
                        y_ghi_batch,
                    ) = (
                        X_batch.to(device),
                        X_future_covariate_batch.to(device),
                        X_future_clear_sky_batch.to(device),
                        y_ghi_batch.to(device),
                    )

                    y_pred = model(
                        X_batch, X_future_covariate_batch, X_future_clear_sky_batch
                    )
                    loss = criterion(y_pred, y_ghi_batch)
                    val_loss += loss.item()

                    y_val_pred.extend(y_pred.cpu().numpy())
                    y_val_true.extend(y_ghi_batch.cpu().numpy())

            val_loss /= len(val_loader)
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss}, Val Loss: {val_loss}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    save_dir=SCRATCH_DIR,
                    model_name=f"forecast_model.pth",
                )
        model, _, _, _, _ = load_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            load_path=os.path.join(SCRATCH_DIR, f"forecast_model.pth"),
        )
        val_metrics = self.test_model(model, val_tensor_dataset)
        return val_metrics

    def objective(self, train_tensors, trial):
        val_metrics = self.train_model(train_tensors, trial)

        def calculate_average_nmap(metrics):
            nmap_values = [value for key, value in metrics.items() if "nMAP" in key]
            average_nmap = sum(nmap_values) / len(nmap_values)
            return average_nmap

        return calculate_average_nmap(val_metrics)

    def tune_model(self, train_tensors, n_trials=10):
        storage_url = f"sqlite:///./database_{self.study_name}.db"
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5, interval_steps=1
        )
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: self.objective(train_tensors, trial),
            n_trials=n_trials,
            show_progress_bar=True,
        )

    def test_model(self, model, test_tensor_dataset):
        test_loader = DataLoader(
            test_tensor_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        model.eval()
        y_test_true_by_step = [[] for _ in range(NUM_FUTURE_FRAMES)]
        y_test_pred_by_step = [[] for _ in range(NUM_FUTURE_FRAMES)]
        test_loss = 0.0
        criterion = nn.L1Loss()
        with torch.no_grad():
            for (
                X_batch,
                X_future_covariate_batch,
                X_future_clear_sky_batch,
                y_ghi_batch,
            ) in test_loader:
                X_batch = X_batch.to(device)
                X_future_covariate_batch = X_future_covariate_batch.to(
                    device
                )  # shape: [B, NUM_COVARIATE_FEATURES * NUM_FUTURE_FRAMES]
                X_future_clear_sky_batch = X_future_clear_sky_batch.to(
                    device
                )  # shape: [B, NUM_FUTURE_FRAMES]
                y_ghi_batch = y_ghi_batch.to(device)  # shape: [B, NUM_FUTURE_FRAMES]
                y_pred = model(
                    X_batch, X_future_covariate_batch, X_future_clear_sky_batch
                )  # shape: [B, NUM_FUTURE_FRAMES]
                loss = criterion(y_pred, y_ghi_batch)
                test_loss += loss.item()
                y_pred_np = y_pred.cpu().numpy()
                y_true_np = y_ghi_batch.cpu().numpy()
                for step in range(NUM_FUTURE_FRAMES):
                    y_test_pred_by_step[step].extend(y_pred_np[:, step])
                    y_test_true_by_step[step].extend(y_true_np[:, step])
        test_loss /= len(test_loader)
        reported_indices = [5, 11, 17, 23]
        reported_names = ["1hr", "2hr", "3hr", "4hr"]
        combined_metrics = {}
        for i, step in enumerate(reported_indices):
            step_metrics = self.evaluate_function(
                y_true=y_test_true_by_step[step], y_pred=y_test_pred_by_step[step]
            )
            print(f"\nMetrics for {step+1} steps ahead forecast:")
            print(f"MAE: {step_metrics['MAE']:.4f}")
            print(f"RMSE: {step_metrics['RMSE']:.4f}")
            print(f"R2: {step_metrics['R2']:.4f}")
            print(f"nMAP: {step_metrics['nMAP']:.4f}")
            print(f"std: {step_metrics['standard_deviation']:.4f}")
            combined_metrics[f"MAE_{reported_names[i]}"] = step_metrics["MAE"]
            combined_metrics[f"RMSE_{reported_names[i]}"] = step_metrics["RMSE"]
            combined_metrics[f"R2_{reported_names[i]}"] = step_metrics["R2"]
            combined_metrics[f"nMAP_{reported_names[i]}"] = step_metrics["nMAP"]
            combined_metrics[f"std_{reported_names[i]}"] = step_metrics[
                "standard_deviation"
            ]
        return combined_metrics


timeseries_train_data = generate_timeseries_data(train_dataset_df, train_embeddings)
train_tensors = process_timeseries_data(timeseries_train_data, isTraining=True)
predictor = BaselineTimeseriesPredictor()
model = predictor.tune_model(train_tensors, n_trials=10)
# Test
timeseries_test_data = generate_timeseries_data(test_dataset_df, test_embeddings)
test_tensors = process_timeseries_data(timeseries_test_data, isTraining=False)
X_test_tensor = test_tensors["X_train_tensor"]
X_future_covariate_test_tensor = test_tensors["X_future_covariate_test_tensor"]
X_future_clear_sky_test_tensor = test_tensors["X_future_clear_sky_test_tensor"]
y_ghi_test_tensor = test_tensors["y_ghi_test_tensor"]
test_tensor_dataset = TensorDataset(
    X_test_tensor,
    X_future_covariate_test_tensor,
    X_future_clear_sky_test_tensor,
    y_ghi_test_tensor,
)
test_metrics = predictor.test_model(test_tensor_dataset)
