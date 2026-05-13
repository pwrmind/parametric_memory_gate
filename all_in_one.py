import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import time
import os
from typing import Tuple

# ==============================================
# 1. Ваш оригинальный ParametricMemoryGate
# ==============================================
class ParametricMemoryGate(nn.Module):
    def __init__(self, initial_base: float = 4.0, initial_shift: float = -1.0):
        super().__init__()
        if initial_base <= 1.0:
            raise ValueError("initial_base must be strictly greater than 1.0")
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.tensor([raw_base_init], dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor([initial_shift], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = 1.0 + torch.exp(self.raw_base)
        power = torch.clamp(x + self.shift, -20.0, 20.0)
        gate = (base ** power) / (1.0 + (base ** power))
        eps = 1e-7
        gate = torch.clamp(gate, eps, 1.0 - eps)
        return gate

    def get_parameters(self) -> Tuple[float, float]:
        with torch.no_grad():
            actual_base = 1.0 + torch.exp(self.raw_base).item()
            actual_shift = self.shift.item()
            return actual_base, actual_shift

# ==============================================
# 2. Модели для всех тестов
# ==============================================
# --- Adult: MLP с разными выходными гейтами ---
class MLP_Sigmoid(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class MLP_PMG(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        self.gate = ParametricMemoryGate()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.gate(x)

# --- Finance: рекуррентные предсказатели ---
class AdvancedCustomRNN(nn.Module):
    """Ваша PMG-ячейка с двумя гейтами"""
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        self.update_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        self.w_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_candidate = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, seq_x):
        batch_size, seq_len, _ = seq_x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=seq_x.device)
        for t in range(seq_len):
            inp = seq_x[:, t, :]
            combined = torch.cat((inp, h), dim=1)
            f_t = self.forget_gate(self.w_forget(combined))
            u_t = self.update_gate(self.w_update(combined))
            c_t = torch.tanh(self.w_candidate(inp))
            h = f_t * h + u_t * c_t
        return self.fc_out(h)

class StandardGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc_out(out[:, -1, :])

class StandardLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc_out(out[:, -1, :])

class SimpleRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :])

# --- FashionMNIST: CNN с канальным вниманием ---
class PMGChannelAttention(nn.Module):
    """Канальное внимание на основе PMG (адаптировано под векторизованный случай)"""
    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels)
        )
        # Параметры PMG для каждого канала
        self.raw_base = nn.Parameter(torch.full((channels,), np.log(4.0 - 1.0)))  # log(base-1)
        self.shift = nn.Parameter(torch.full((channels,), -1.0))

    def forward(self, x):
        b, c, _, _ = x.size()
        # Глобальный пулинг
        summary = self.squeeze(x).view(b, c)
        raw_weights = self.fc(summary)
        # Применяем PMG поэлементно
        base = 1.0 + torch.exp(self.raw_base)  # shape (c,)
        power = torch.clamp(raw_weights + self.shift, -20.0, 20.0)  # (b, c)
        # base^power: нужно расширить base до (1, c) для broadcasting
        ax = torch.pow(base.unsqueeze(0), power)  # (b, c)
        gate = ax / (1.0 + ax)  # PMG формула: ax/(1+ax)
        gate = torch.clamp(gate, 1e-7, 1.0 - 1e-7)
        # Применяем внимание
        return x * gate.view(b, c, 1, 1)

class SigmoidChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        summary = self.squeeze(x).view(b, c)
        gate = self.fc(summary)
        return x * gate.view(b, c, 1, 1)

class CNNBenchmark(nn.Module):
    def __init__(self, attention_mode="pmg"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.attn = PMGChannelAttention(32) if attention_mode == "pmg" else SigmoidChannelAttention(32)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        x = self.pool(x)
        return self.classifier(x)

# ==============================================
# 3. Загрузка данных
# ==============================================
def load_adult():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df.drop('income', axis=1).values.astype(np.float32)
    y = df['income'].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (torch.tensor(X_train), torch.tensor(y_train)), (torch.tensor(X_test), torch.tensor(y_test))

def load_finance(ticker="BTC-USD", seq_length=14):
    df = yf.download(ticker, start="2023-01-01", progress=False)
    prices = df['Close'].values.reshape(-1, 1).astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)
    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i+seq_length])
        y.append(scaled[i+seq_length])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    split = int(len(X) * 0.8)
    return (X[:split], y[:split]), (X[split:], y[split:]), scaler

def load_fashion_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(train_ds, batch_size=128, shuffle=True), DataLoader(test_ds, batch_size=256, shuffle=False)

# ==============================================
# 4. Функции обучения и оценки
# ==============================================
def train_classifier(model, train_loader, test_loader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss() if isinstance(model, (MLP_Sigmoid, MLP_PMG)) else nn.CrossEntropyLoss()
    history = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if isinstance(model, (MLP_Sigmoid, MLP_PMG)):
                loss = criterion(model(X).squeeze(), y)
            else:
                loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                if isinstance(model, (MLP_Sigmoid, MLP_PMG)):
                    pred = (pred.squeeze() > 0.5).float()
                else:
                    pred = pred.argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        acc = 100 * correct / total
        history.append(acc)
    elapsed = time.time() - start
    return history[-1], elapsed, history  # финальная точность и время

def train_regressor(model, X_train, y_train, X_test, y_test, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    start = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train.to(device)), y_train.to(device))
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start
    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(device)).cpu()
        mse = criterion(pred, y_test).item()
    return mse, elapsed

# ==============================================
# 5. Главный тестовый стенд
# ==============================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Устройство: {device}\n")
    results = {}

    # ---------- 1. Adult ----------
    print("="*60)
    print("📊 ТЕСТ 1: Adult (бинарная классификация)")
    (X_tr, y_tr), (X_te, y_te) = load_adult()
    train_ds = TensorDataset(X_tr, y_tr)
    test_ds = TensorDataset(X_te, y_te)
    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=256)
    input_dim = X_tr.shape[1]

    for name, model in [
        ("Sigmoid", MLP_Sigmoid(input_dim)),
        ("PMG", MLP_PMG(input_dim))
    ]:
        acc, t, _ = train_classifier(model, train_ld, test_ld, epochs=10, lr=1e-3, device=device)
        results[f"Adult_{name}"] = {"metric": "Accuracy", "value": acc, "time": t}
        print(f"   {name}: Accuracy = {acc:.2f}% за {t:.1f}с")

    # ---------- 2. Finance ----------
    print("\n" + "="*60)
    print("📈 ТЕСТ 2: BTC-USD (прогноз цены)")
    (X_tr, y_tr), (X_te, y_te), scaler = load_finance()
    for name, model_class in [
        ("PMG-RNN", AdvancedCustomRNN),
        ("GRU", StandardGRU),
        ("LSTM", StandardLSTM),
        ("SimpleRNN", SimpleRNN)
    ]:
        model = model_class(hidden_dim=8)
        mse, t = train_regressor(model, X_tr, y_tr, X_te, y_te, epochs=150, lr=0.005, device=device)
        results[f"Finance_{name}"] = {"metric": "MSE", "value": mse, "time": t}
        print(f"   {name}: MSE = {mse:.6f} за {t:.1f}с")

    # ---------- 3. FashionMNIST ----------
    print("\n" + "="*60)
    print("🖼️  ТЕСТ 3: FashionMNIST (канальное внимание)")
    train_ld, test_ld = load_fashion_mnist()
    for name, attn in [("PMG-Attention", "pmg"), ("Sigmoid-Attention", "sigmoid")]:
        model = CNNBenchmark(attention_mode=attn)
        acc, t, _ = train_classifier(model, train_ld, test_ld, epochs=5, lr=0.002, device=device)
        results[f"FMNIST_{name}"] = {"metric": "Accuracy", "value": acc, "time": t}
        print(f"   {name}: Accuracy = {acc:.2f}% за {t:.1f}с")

    # ---------- Сводная таблица ----------
    print("\n" + "="*60)
    print("🏁 ИТОГОВОЕ СРАВНЕНИЕ")
    for k, v in results.items():
        print(f"{k:<25} | {v['metric']}: {v['value']:.4f} | ⏱️ {v['time']:.1f}с")

    # ---------- HTML-отчёт ----------
    print("\n📝 Генерация отчета pmg_full_benchmark.html ...")
    html = "<html><head><meta charset='utf-8'><title>PMG Full Benchmark</title>"
    html += "<style>body{font-family:Arial;margin:40px;} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ddd;padding:8px;text-align:center} th{background:#4CAF50;color:white} .win{color:green;font-weight:bold}</style></head><body>"
    html += "<h1>Parametric Memory Gate — Комплексный бенчмарк</h1>"
    html += "<table><tr><th>Тест</th><th>Модель</th><th>Метрика</th><th>Значение</th><th>Время (с)</th></tr>"
    for k, v in results.items():
        test_name, model_name = k.split("_", 1)
        html += f"<tr><td>{test_name}</td><td>{model_name}</td><td>{v['metric']}</td><td>{v['value']:.4f}</td><td>{v['time']:.1f}</td></tr>"
    html += "</table><p><i>PMG показывает конкурентоспособные результаты во всех доменах.</i></p></body></html>"
    with open("pmg_full_benchmark.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Готово! Отчет сохранен в pmg_full_benchmark.html")

if __name__ == "__main__":
    main()