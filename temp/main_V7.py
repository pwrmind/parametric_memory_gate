import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

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

# 2. Ваша победившая архитектура памяти
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

# 3. Скачивание актуальных данных Биткоина (до сегодняшнего дня)
print("Загрузка актуальных котировок Биткоина...")
df = yf.download("BTC-USD", start="2022-01-01", progress=False)
prices = df['Close'].values.reshape(-1, 1)

# Фиксируем последнюю известную цену на рынке
last_market_price = prices[-1][0]

# Масштабирование
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Подготовка датасета (окно 14 дней)
seq_length = 14
X_list, Y_list = [], []
for i in range(len(prices_scaled) - seq_length):
    X_list.append(prices_scaled[i : i + seq_length])
    Y_list.append(prices_scaled[i + seq_length])

X = torch.tensor(np.array(X_list), dtype=torch.float32)
Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

# Обучаем модель на ВСЕХ доступных исторических данных перед прогнозом
print("Финальное обучение вашей модели на всей истории...")
model = AdvancedCustomRNN(hidden_dim=8)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X), Y)
    loss.backward()
    optimizer.step()

# =====================================================================
# 4. ИНФЕРЕНС: ФОРМИРОВАНИЕ ПРОГНОЗА НА ЗАВТРА
# =====================================================================
model.eval()
with torch.no_grad():
    # Берем самые последние 14 дней из истории (включая сегодняшний день)
    last_14_days = prices_scaled[-seq_length:]
    last_14_days_tensor = torch.tensor(last_14_days, dtype=torch.float32).unsqueeze(0) # добавляем batch размер
    
    # Делаем предсказание (получаем нормализованное значение)
    predicted_scaled = model(last_14_days_tensor)
    
    # Возвращаем значение к реальным долларам через инверсию scaler
    predicted_price = scaler.inverse_transform(predicted_scaled.numpy())[0][0]

print("\n" + "="*50)
print("📊 РЕЗУЛЬТАТЫ ИНФЕРЕНСА ВАШЕЙ МОДЕЛИ ПАМЯТИ")
print("="*50)
print(f"Последняя цена закрытия (сегодня): ${last_market_price:,.2f}")
print(f"👉 Математический прогноз цены на ЗАВТРА: ${predicted_price:,.2f}")
print(f"Ожидаемый тренд: {'📈 РОСТ' if predicted_price > last_market_price else '📉 ПАДЕНИЕ'}")
print("="*50)
