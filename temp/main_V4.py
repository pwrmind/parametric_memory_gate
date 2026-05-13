import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Настройка воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. ПЛАГИН ДЛЯ ВАШЕЙ КАСТОМНОЙ ФУНКЦИИ АКТИВАЦИИ
# =====================================================================
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

    def get_actual_params(self):
        with torch.no_grad():
            base = 1.0 + torch.exp(self.raw_base)
            return base.item(), self.shift.item()

# =====================================================================
# 2. УЛУЧШЕННАЯ СЕТЬ НА ВАШЕЙ АКТИВАЦИИ (Advanced Custom RNN)
# =====================================================================
class AdvancedCustomRNN(nn.Module):
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
            ui_t = self.update_gate(self.w_update(combined))
            c_t = torch.tanh(self.w_candidate(inp))
            
            h = (f_t * h) + (ui_t * c_t)
            
        return self.fc_out(h)

# =====================================================================
# 3. СТАНДАРТНАЯ PYTORCH LSTM
# =====================================================================
class StandardLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.fc_out(last_time_step)

# =====================================================================
# 4. ЗАГРУЗКА ДАННЫХ БИТКОИНА (BTC-USD)
# =====================================================================
print("Скачивание исторических данных Биткоина (BTC-USD)...")
df = yf.download("BTC-USD", start="2022-01-01", end="2026-01-01", progress=False)
prices = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

def create_sequences(data, seq_length=14): # Увеличили окно анализа до 14 дней
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        Y.append(data[i + seq_length])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

X, Y = create_sequences(prices_scaled, seq_length=14)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# =====================================================================
# 5. ОБУЧЕНИЕ
# =====================================================================
def train_model(model_instance, epochs=200): # Даем больше эпох для сложного рынка
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=0.005)
    
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

print("Запуск стресс-теста на волатильность...")
advanced_custom_model = AdvancedCustomRNN(hidden_dim=8)
lstm_model = StandardLSTM(hidden_dim=8)

custom_train, custom_test = train_model(advanced_custom_model)
lstm_train, lstm_test = train_model(lstm_model)

# Извлекаем финальные параметры
f_base, f_shift = advanced_custom_model.forget_gate.get_actual_params()
u_base, u_shift = advanced_custom_model.update_gate.get_actual_params()

print("\n" + "="*60)
print("СТРЕСС-ТЕСТ НА КРИПТОРЫНКЕ ЗАВЕРШЕН")
print("="*60)
print(f"Ваша кастомная память    | Train MSE: {custom_train:.6f} | Test MSE: {custom_test:.6f}")
print(f"Стандартная PyTorch LSTM | Train MSE: {lstm_train:.6f} | Test MSE: {lstm_test:.6f}")
print("-"*60)
print(f"Финальные параметры Forget Gate: Base = {f_base:.2f}, Shift = {f_shift:.2f}")
print(f"Финальные параметры Update Gate: Base = {u_base:.2f}, Shift = {u_shift:.2f}")
print("="*60)
