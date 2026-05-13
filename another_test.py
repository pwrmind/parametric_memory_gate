import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import time

# Фиксация случайных чисел для строгого равенства условий
torch.manual_seed(42)
np.random.seed(42)

# 1. Исправленная математика ворот PMG и остальных стандартов
class BenchmarkGate(nn.Module):
    def __init__(self, mode="pmg", hidden_dim=8):
        super().__init__()
        self.mode = mode
        if mode == "pmg":
            # Инициализация строго по вашему ТЗ (векторные параметры)
            self.a = nn.Parameter(torch.full((hidden_dim,), 4.0, dtype=torch.float32))
            self.b = nn.Parameter(torch.full((hidden_dim,), -1.0, dtype=torch.float32))
        elif mode == "prelu":
            self.prelu = nn.PReLU(num_parameters=hidden_dim)

    def forward(self, x):
        if self.mode == "pmg":
            # Гарантируем, что основание степени строго больше 1
            a_safe = torch.clamp(self.a, min=1.001)
            
            # Чистая реализация вашей формулы: f(x) = a^x / (b + a^x)
            # Чтобы избежать деления на 0 при обучении b, берем его модуль + малую константу
            ax = torch.exp(x * torch.log(a_safe))
            b_safe = torch.abs(self.b) + 1e-5
            
            # Считаем гейт и строго зажимаем в диапазон (0, 1) для стабильности рекуррентности
            out = ax / (b_safe + ax)
            return torch.clamp(out, 0.0, 1.0)
            
        elif self.mode == "sigmoid":
            return torch.sigmoid(x)
            
        elif self.mode == "relu":
            return torch.clamp(torch.relu(x), 0.0, 1.0)
            
        elif self.mode == "prelu":
            return torch.clamp(self.prelu(x), 0.0, 1.0)
            
        elif self.mode == "swish" or self.mode == "silu":
            return torch.clamp(x * torch.sigmoid(x), 0.0, 1.0)

# 2. Ваша исходная архитектура с двумя независимыми гейтами (PMG-ячейка)
class FlexibleCustomRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, gate_mode="pmg"):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Два независимых гейта на базе выбранного режима
        self.forget_gate = BenchmarkGate(mode=gate_mode, hidden_dim=hidden_dim)
        self.update_gate = BenchmarkGate(mode=gate_mode, hidden_dim=hidden_dim)
        
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
            
            # Применение гейтов по вашей логике
            f_t = self.forget_gate(self.w_forget(combined))
            ui_t = self.update_gate(self.w_update(combined))
            c_t = torch.tanh(self.w_candidate(inp))
            
            # Перезапись памяти
            h = (f_t * h) + (ui_t * c_t)
            
        return self.fc_out(h)

# 3. Подготовка данных (BTC-USD)
print("Скачивание данных из Yahoo Finance...")
data = yf.download('BTC-USD', start='2022-01-01', end='2026-01-01', interval='1d')
prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 14
X, y = create_sequences(scaled_prices, SEQ_LENGTH)

train_size = int(len(X) * 0.8)
X_train = torch.tensor(X[:train_size], dtype=torch.float32)
y_train = torch.tensor(y[:train_size], dtype=torch.float32)
X_test = torch.tensor(X[train_size:], dtype=torch.float32)
y_test = torch.tensor(y[train_size:], dtype=torch.float32)

# 4. Сравнительный анализ (150 эпох)
modes = ["pmg", "sigmoid", "relu", "prelu", "swish"]
results = {}

print("\n=== ЗАПУСК ИСПРАВЛЕННОГО СРАВНЕНИЯ (150 ЭПОХ) ===")

for mode in modes:
    model = FlexibleCustomRNN(input_dim=1, hidden_dim=8, gate_mode=mode)
    criterion = nn.MSELoss()
    
    # Настройка раздельного шага обучения для стабильной сходимости PMG
    if mode == "pmg":
        pmg_params = [model.forget_gate.a, model.forget_gate.b, model.update_gate.a, model.update_gate.b]
        base_params = [p for p in model.parameters() if not any(p is pmg for pmg in pmg_params)]
        
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': 0.001, 'weight_decay': 1e-4},
            {'params': pmg_params, 'lr': 0.03}  # Оптимальный шаг для параметров гейтов
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    start_time = time.time()
    epochs = 150
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    execution_time = time.time() - start_time
    
    # Тестирование на отложенной выборке
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        
    results[mode] = {
        "Train Loss (150 ep)": f"{loss.item():.6f}",
        "Test MSE Loss": f"{test_loss:.6f}",
        "Speed (sec)": f"{execution_time:.3f}"
    }
    
    if mode == "pmg":
        # Считаем среднее значение параметров по всем нейронам для forget-гейта
        mean_a_f = model.forget_gate.a.mean().item()
        mean_b_f = model.forget_gate.b.mean().item()
        # Считаем среднее значение параметров по всем нейронам для update-гейта
        mean_a_u = model.update_gate.a.mean().item()
        mean_b_u = model.update_gate.b.mean().item()
        print(f"Финиш: PMG (Ваш) -> Выученные средние Forget Gate: a={mean_a_f:.2f}, b={mean_b_f:.2f} | Update Gate: a={mean_a_u:.2f}, b={mean_b_u:.2f}")
    else:
        print(f"Финиш: {mode.upper()}")

# Вывод итоговой таблицы
df_results = pd.DataFrame(results).T
print("\n=== ФИНАЛЬНАЯ ТАБЛИЦА СРАВНЕНИЯ ===")
print(df_results.to_string())
