import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Воспроизводимость
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. АРХИТЕКТУРА ВАШЕЙ ПОБЕДИВШЕЙ ПАМЯТИ
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
# 2. ПОДГОТОВКА ДАННЫХ
# =====================================================================
print("Загрузка данных BTC-USD...")
df = yf.download("BTC-USD", start="2022-01-01", end="2026-01-01", progress=False)
prices = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

seq_length = 14
X_list, Y_list = [], []
for i in range(len(prices_scaled) - seq_length):
    X_list.append(prices_scaled[i : i + seq_length])
    Y_list.append(prices_scaled[i + seq_length])

X = torch.tensor(np.array(X_list), dtype=torch.float32)
Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

trade_prices = prices[seq_length + split:] 

# =====================================================================
# 3. ОБУЧЕНИЕ МОДЕЛИ
# =====================================================================
print("Обучение ядра ИИ торгового бота...")
model = AdvancedCustomRNN(hidden_dim=8)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), Y_train)
    loss.backward()
    optimizer.step()

# =====================================================================
# 4. СИМУЛЯЦИЯ С ФИЛЬТРОМ КРУПНЫХ ТРЕНДОВ
# =====================================================================
print("Запуск оптимизированного робота...")
model.eval()
with torch.no_grad():
    predictions_scaled = model(X_test).numpy()
    predictions = scaler.inverse_transform(predictions_scaled)

cash = 1000.0
initial_balance = cash
crypto_inventory = 0.0
in_position = False
commission_rate = 0.001   # 0.1%
total_trades = 0

# Увеличиваем порог входа: ждем прогноза роста минимум на 1.5%
signal_threshold = 0.015 
# Смягчаем порог выхода: выходим только если прогноз обещает падение глубже -1.0%
exit_threshold = -0.01

for i in range(len(trade_prices) - 1):
    current_price = trade_prices[i][0]
    predicted_next_price = predictions[i][0]
    
    expected_return = (predicted_next_price - current_price) / current_price
    
    if expected_return > signal_threshold and not in_position:
        # BUY
        crypto_inventory = (cash * (1.0 - commission_rate)) / current_price
        cash = 0.0
        in_position = True
        total_trades += 1
        
    elif expected_return < exit_threshold and in_position:
        # SELL
        cash = (crypto_inventory * current_price) * (1.0 - commission_rate)
        crypto_inventory = 0.0
        in_position = False
        total_trades += 1

final_price = trade_prices[-1][0]
final_balance = (crypto_inventory * final_price) if in_position else cash
buy_and_hold_final = (initial_balance / trade_prices[0][0]) * final_price

# =====================================================================
# 5. ОТЧЕТ
# =====================================================================
print("\n" + "="*55)
print("📊 ОТЧЕТ МОДЕРНИЗИРОВАННОГО БЭКТЕСТИНГА")
print("="*55)
print(f"Период симуляции:        {len(trade_prices)} дней")
print(f"Стартовый баланс:        ${initial_balance:,.2f}")
print(f"Совершено сделок:        {total_trades} (вместо 62!)")
print("-"*55)
print(f"🤖 Итоговый баланс БОТА:  ${final_balance:,.2f}")
print(f"📈 Доходность БОТА:       {((final_balance - initial_balance)/initial_balance)*100:.2f}%")
print("-"*55)
print(f"💼 Стратегия 'Купил и держи': ${buy_and_hold_final:,.2f}")
print(f"📊 Доходность 'Купил и держи': {((buy_and_hold_final - initial_balance)/initial_balance)*100:.2f}%")
print("="*55)

if final_balance > buy_and_hold_final:
    print("🏆 УСПЕХ: Новая стратегия вывела вашу память в лидеры рынка!")
else:
    print("📉 Рынок оказался сильнее, требуются дополнительные маркеры тренда.")
