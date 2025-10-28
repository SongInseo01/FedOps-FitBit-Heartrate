from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(features: np.ndarray, window_size: int):
    """Return (num_seq, window, feat), (num_seq,) with label=next-step HR."""
    sequences, labels = [], []
    for idx in range(len(features) - window_size):
        sequences.append(features[idx : idx + window_size])
        labels.append(features[idx + window_size, 0])  # column 0: HeartRate
    return np.stack(sequences), np.asarray(labels)

def load_and_prepare(csv_path: str, window_size: int, batch_size):
    EPS = 1e-6
    df = pd.read_csv(csv_path)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["Id", "Time"]).reset_index(drop=True)

    # 가장 데이터가 많은 참가자 선택
    target_id = df["Id"].value_counts().idxmax()
    target = (
        df.loc[df["Id"] == target_id, ["Time", "Value"]]
        .set_index("Time")
        .resample("1min")
        .mean()
        .ffill()
        .dropna()
        .rename(columns={"Value": "HeartRate"})
        .reset_index()
    )

    # 시간 주기 특성
    minute_of_day = target["Time"].dt.hour * 60 + target["Time"].dt.minute
    target["sin_time"] = np.sin(2 * np.pi * minute_of_day / 1440)
    target["cos_time"] = np.cos(2 * np.pi * minute_of_day / 1440)

    feature_cols = ["HeartRate", "sin_time", "cos_time"]
    features = target[feature_cols].to_numpy(dtype=np.float32)

    X_all, y_all = create_sequences(features, window_size=window_size)
    total = X_all.shape[0]
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)

    # Split
    X_train = torch.tensor(X_all[:train_end], dtype=torch.float32)
    y_train = torch.tensor(y_all[:train_end], dtype=torch.float32)
    X_val = torch.tensor(X_all[train_end:val_end], dtype=torch.float32)
    y_val = torch.tensor(y_all[train_end:val_end], dtype=torch.float32)
    X_test = torch.tensor(X_all[val_end:], dtype=torch.float32)
    y_test = torch.tensor(y_all[val_end:], dtype=torch.float32)

    # Standardize (fit on train)
    feat_mean = X_train.mean(dim=(0, 1), keepdim=True)
    feat_std = X_train.std(dim=(0, 1), keepdim=True)

    X_train = (X_train - feat_mean) / (feat_std + EPS)
    X_val = (X_val - feat_mean) / (feat_std + EPS)
    X_test = (X_test - feat_mean) / (feat_std + EPS)

    # Target normalize
    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_n = (y_train - y_mean) / (y_std + EPS)
    y_val_n = (y_val - y_mean) / (y_std + EPS)
    y_test_n = (y_test - y_mean) / (y_std + EPS)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train_n), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val_n), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test_n), batch_size=batch_size, shuffle=False)

    meta = {
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "target_mean": y_mean,
        "target_std": y_std,
        "input_dim": X_train.shape[-1],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "target_id": int(target_id),
    }
    return train_loader, val_loader, test_loader, meta


# Define MNIST Model    
# class MNISTClassifier(nn.Module):
#     # To properly utilize the config file, the output_size variable must be used in __init__().
#     def __init__(self, output_size):
#         super(MNISTClassifier, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

#         # Fully connected layers
#         self.fc1 = nn.Linear(64 * 7 * 7, 1000)  # Image size is 28x28, reduced to 14x14 and then to 7x7
#         self.fc2 = nn.Linear(1000, output_size)  # 10 output classes (digits 0-9)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)

#         # Flatten the output for the fully connected layers
#         x = x.view(-1, 64 * 7 * 7)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x

class HeartRateLSTM(nn.Module):
    def __init__(self, input_dim: int = 3, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(64, 1) # output_size=1

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)

# class FitbitModel(nn.Module):
#     def __init__(self, input_features: int = 10, output_size: int = 1):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_features, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, output_size),
#         )

#     def forward(self, x):
#         return self.model(x)

# Set the torch train & test
# torch train
# def train_torch():
#     def custom_train_torch(model, train_loader, epochs, cfg):
#         """
#         Train the network on the training set.
#         Model must be the return value.
#         """
#         print("Starting training...")
        
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)

#         model.train()
#         for epoch in range(epochs):
#             with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
#                 for inputs, labels in train_loader:
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     optimizer.zero_grad()
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     loss.backward()
#                     optimizer.step()
                    
#                     pbar.update()  # Update the progress bar for each batch

#         model.to("cpu")
            
#         return model
    
#     return custom_train_torch

WINDOW_SIZE = 30
CSV_PATH = "/home/ubuntu/isfolder/fl_agent_paper/buildmodel/content/heartrate_seconds_merged.csv"
# CSV_PATH = "/app/code/data/heartrate_seconds_merged.csv" # for server
batch_size = 32
train_loader, val_loader, test_loader, meta = load_and_prepare(CSV_PATH, WINDOW_SIZE, batch_size)

def train_torch():
    def custom_train_torch(model, train_loader, epochs, cfg):
        """
        Train the network on the training set.
        Model must be the return value.
        """
        print("Starting Local training...")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.train()
        for epoch in range(epochs):
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    pbar.update()

        model.to("cpu")
            
        return model
    
    return custom_train_torch

def test_torch():

    def _to_tensor_like(x, device):
        # float, np.ndarray, torch.Tensor 모두 허용
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def custom_test_torch(model, test_loader, cfg=None):
        target_mean = meta["target_mean"]
        target_std = meta["target_std"]
        eps = 1e-8
        """
        Evaluate the network on test set (regression).
        Returns:
            - y_true: np.ndarray, shape (N,)
            - y_pred: np.ndarray, shape (N,)
            - metrics: {"MAE": float, "RMSE": float}
        """
        print("Starting evaluation...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        preds_norm, targs_norm = [], []

        with torch.no_grad():
            for xb, yb in tqdm(test_loader, desc="Test", unit="batch", leave=False):
                xb = xb.to(device)
                yb = yb.to(device).float()
                pred = model(xb).squeeze(-1)  # (N, 1)->(N,)
                preds_norm.append(pred.detach())
                targs_norm.append(yb.detach())

        if len(preds_norm) == 0:
            print("Test set empty.")
            model.to("cpu")
            return np.array([]), np.array([]), {"MAE": np.nan, "RMSE": np.nan}

        preds_norm = torch.cat(preds_norm, dim=0)
        targs_norm = torch.cat(targs_norm, dim=0)

        # 역정규화
        tm = _to_tensor_like(target_mean, preds_norm.device)
        ts = _to_tensor_like(target_std, preds_norm.device)

        preds = preds_norm * (ts + float(eps)) + tm
        targs = targs_norm * (ts + float(eps)) + tm

        # CPU/Numpy 변환
        y_pred = preds.detach().cpu().numpy()
        y_true = targs.detach().cpu().numpy()

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)  # squared=True
        rmse = sqrt(mse)

        print(f"Test MAE: {mae:.2f}")
        print(f"Test RMSE: {rmse:.2f}")

        model.to("cpu")
        return y_true, y_pred, {"MAE": mae, "RMSE": rmse}

    return custom_test_torch

# torch test
# def test_torch():
    
#     def custom_test_torch(model, test_loader, cfg):
#         """
#         Validate the network on the entire test set.
#         Loss, accuracy values, and dictionary-type metrics variables are fixed as return values.
#         """
#         """
#         Validate the network on the entire test set.

#         반환 형식(동일):
#             - average_loss: 회귀→ HuberLoss 평균, 분류→ CrossEntropy 평균
#             - score:   회귀→ R²,        분류→ accuracy
#             - metrics: dict
#                 * 회귀: {"MAE", "RMSE", "R2", "PearsonR"}
#                 * 분류: {"f1_score"}
#         """
#         print("Starting evalutation...")
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         model.eval()

#         # 누적 변수
#         total_loss = 0.0
#         num_batches = 0

#         # 분류용
#         correct = 0
#         all_labels_cls = []
#         all_preds_cls = []

#         # 회귀용
#         preds_reg = []
#         trues_reg = []

#         # 작업 유형 자동 판별 플래그(첫 배치에서 결정)
#         task_type = "regression"  # "regression" or "classification"
        
#         with torch.no_grad():
#             with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
#                 for inputs, labels in test_loader:
#                     inputs = inputs.to(device)

#                     # 작업 유형 자동 판별
#                     if task_type is None:
#                         # (1) 라벨 dtype이 float이면 회귀
#                         if labels.dtype in (torch.float32, torch.float64):
#                             task_type = "regression"
#                         else:
#                             task_type = "classification"

#                     # 장치/타입 정렬
#                     if task_type == "regression":
#                         labels = labels.to(device).float()
#                         criterion = nn.HuberLoss(delta=0.5)
#                     else:
#                         labels = labels.to(device).long()
#                         criterion = nn.CrossEntropyLoss()

#                     # forward
#                     outputs = model(inputs)

#                     # 손실 및 예측/정답 누적
#                     if task_type == "regression":
#                         # 출력: (N, 1) 또는 (N,)
#                         yhat = outputs.squeeze(-1)
#                         loss = criterion(yhat, labels)
#                         preds_reg.append(yhat.detach().cpu())
#                         trues_reg.append(labels.detach().cpu())
#                     else:
#                         loss = criterion(outputs, labels)
#                         _, predicted = torch.max(outputs, 1)
#                         correct += (predicted == labels).sum().item()
#                         all_labels_cls.extend(labels.cpu().numpy())
#                         all_preds_cls.extend(predicted.cpu().numpy())

#                     total_loss += loss.item()
#                     num_batches += 1
#                     pbar.update()

#         average_loss = total_loss / max(num_batches, 1)

#         # 지표 계산
#         if task_type == "regression":
#             y_true = torch.cat(trues_reg).numpy()
#             y_pred = torch.cat(preds_reg).numpy()

#             mae = mean_absolute_error(y_true, y_pred)
#             mse = mean_squared_error(y_true, y_pred)
#             rmse = sqrt(mse)
#             r2 = r2_score(y_true, y_pred)
#             if np.std(y_pred) == 0 or np.std(y_true) == 0:
#                 pearson = np.nan
#             else:
#                 pearson = np.corrcoef(y_true, y_pred)[0, 1]

#             metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "PearsonR": pearson}
#             score = r2  # 두 번째 리턴 값: 회귀는 R²를 대표 점수로 사용
#         else:
#             # 분류
#             accuracy = correct / len(test_loader.dataset)
#             f1 = f1_score(all_labels_cls, all_preds_cls, average='weighted')
#             metrics = {"f1_score": f1}
#             score = accuracy

#         model.to("cpu")
#         return average_loss, score, metrics

#     return custom_test_torch
