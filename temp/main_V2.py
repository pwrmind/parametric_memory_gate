import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Настройка воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

# 1. Ваша кастомная функция активации
class ParametricMemoryGate(nn.Module):
    def __init__(self, initial_base=4.0, initial_shift=-1.0):
        super().__init__()
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.tensor([raw_base_init], dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor([initial_shift], dtype=torch.float32))

    def forward(self, x):
        base = 1.0 + torch.exp(self.raw_base)
        power = torch.clamp(x + self.shift, -20.0, 20.0)
        return (base ** power) / (1.0 + (base ** power))

# 2. Ваша кастомная RNN ячейка
class CustomRNNCell(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_x = nn.Linear(input_dim, hidden_dim)
        self.b_x = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, seq_x):
        batch_size, seq_len, _ = seq_x.size()
        memory = torch.zeros(batch_size, self.hidden_dim, device=seq_x.device)
        relevance = torch.zeros(batch_size, self.hidden_dim, device=seq_x.device)
        
        for t in range(seq_len):
            inp = seq_x[:, t, :]
            relevance = self.b_x(relevance) + self.w_x(inp)
            gate = self.memory_gate(relevance)
            memory = (memory * gate) + inp
            
        return self.fc_out(memory)

# 3. Стандартная модель на базе встроенной PyTorch LSTM
class StandardLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :] # Берем состояние на последнем шаге
        return self.fc_out(last_time_step)

# 4. Подготовка данных AAPL
df = yf.download("AAPL", start="2022-01-01", end="2025-01-01", progress=False)
prices = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

def create_sequences(data, seq_length=10):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        Y.append(data[i + seq_length])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

X, Y = create_sequences(prices_scaled, seq_length=10)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# 5. Функция для обучения моделей
def train_model(model_instance, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=0.01)
    for epoch in range(epochs):
        model_instance.train()
        optimizer.zero_grad()
        loss = criterion(model_instance(X_train), Y_train)
        loss.backward()
        optimizer.step()
    
    model_instance.eval()
    with torch.no_grad():
        test_loss = criterion(model_instance(X_test), Y_test)
    return loss.item(), test_loss.item()

# Запуск баттла архитектур
custom_model = CustomRNNCell(hidden_dim=4)
lstm_model = StandardLSTM(hidden_dim=4)

custom_train, custom_test = train_model(custom_model)
lstm_train, lstm_test = train_model(lstm_model)

print("\n--- РЕЗУЛЬТАТЫ СРАВНЕНИЯ АРХИТЕКТУР ---")
print(f"Ваша кастомная память | Train MSE: {custom_train:.6f} | Test MSE: {custom_test:.6f}")
print(f"Стандартная PyTorch LSTM | Train MSE: {lstm_train:.6f} | Test MSE: {lstm_test:.6f}")
