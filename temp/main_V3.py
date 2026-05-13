import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Настройка воспроизводимости для честного сравнения
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. ПЛАГИН ДЛЯ ВАШЕЙ КАСТОМНОЙ ФУНКЦИИ АКТИВАЦИИ
# =====================================================================
class ParametricMemoryGate(nn.Module):
    """
    Кастомный слой активации: f(x) = base^(x + shift) / (1 + base^(x + shift))
    Основание (base) и сдвиг (shift) обучаются автоматически градиентным спуском.
    """
    def __init__(self, initial_base=4.0, initial_shift=-1.0):
        super().__init__()
        # Защита от отрицательного основания: обучаем сырой лог-параметр.
        # Формула base = 1.0 + exp(raw_base) гарантирует, что base всегда > 1.0
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.tensor([raw_base_init], dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor([initial_shift], dtype=torch.float32))

    def forward(self, x):
        base = 1.0 + torch.exp(self.raw_base)
        # Защита от переполнения памяти при больших степенях
        power = torch.clamp(x + self.shift, -20.0, 20.0)
        return (base ** power) / (1.0 + (base ** power))


# =====================================================================
# 2. УЛУЧШЕННАЯ СЕТЬ НА ВАШЕЙ АКТИВАЦИИ (Advanced Custom RNN)
# =====================================================================
class AdvancedCustomRNN(nn.Module):
    """
    Многогейтовая архитектура. Ваша функция управляет двумя процессами:
    1) Сколько старой памяти удержать (forget_gate)
    2) Сколько новой рыночной информации записать (update_gate)
    """
    def __init__(self, input_dim=1, hidden_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Два независимых фильтра на базе вашей функции
        self.forget_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        self.update_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        
        # Линейные слои для анализа комбинации (текущий вход + прошлый контекст)
        self.w_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Преобразование входящего сигнала (сглаживание шума)
        self.w_candidate = nn.Linear(input_dim, hidden_dim)
        
        # Финальный выходной слой для предсказания цены
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, seq_x):
        batch_size, seq_len, _ = seq_x.size()
        # Инициализируем скрытое состояние нулями
        h = torch.zeros(batch_size, self.hidden_dim, device=seq_x.device)
        
        # Пошаговый проход по 10 дням истории
        for t in range(seq_len):
            inp = seq_x[:, t, :]
            
            # Объединяем текущий день и прошлые воспоминания
            combined = torch.cat((inp, h), dim=1)
            
            # Считаем гейты по вашей формуле
            f_t = self.forget_gate(self.w_forget(combined))
            ui_t = self.update_gate(self.w_update(combined))
            
            # Сглаживаем новую информацию
            c_t = torch.tanh(self.w_candidate(inp))
            
            # Обновляем память системы: баланс старого и нового контекста
            h = (f_t * h) + (ui_t * c_t)
            
        return self.fc_out(h)


# =====================================================================
# 3. СТАНДАРТНАЯ ПРОМЫШЛЕННАЯ МОДЕЛЬ (PyTorch LSTM)
# =====================================================================
class StandardLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Забираем выход только последнего временного шага
        last_time_step = lstm_out[:, -1, :]
        return self.fc_out(last_time_step)


# =====================================================================
# 4. СКАЧИВАНИЕ И ПОДГОТОВКА ДАННЫХ РЫНКА
# =====================================================================
print("Скачивание актуальных котировок AAPL...")
df = yf.download("AAPL", start="2022-01-01", end="2025-01-01", progress=False)
prices = df['Close'].values.reshape(-1, 1)

# Масштабируем цены в диапазон (0, 1) для корректной работы нейросети
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Нарезка временного ряда на окна по 10 дней
def create_sequences(data, seq_length=10):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        Y.append(data[i + seq_length])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

X, Y = create_sequences(prices_scaled, seq_length=10)

# Разделение на обучение (80%) и проверку (20%)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]


# =====================================================================
# 5. СТАНДАРТНАЯ ФУНКЦИЯ ДЛЯ ТРЕНИРОВКИ МОДЕЛЕЙ
# =====================================================================
def train_model(model_instance, epochs=150):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model_instance.train()
        optimizer.zero_grad()
        
        loss = criterion(model_instance(X_train), Y_train)
        loss.backward()
        optimizer.step()
        
    # Оценка модели на тестовых данных, которые она никогда не видела
    model_instance.eval()
    with torch.no_grad():
        test_loss = criterion(model_instance(X_test), Y_test)
        
    return loss.item(), test_loss.item()


# =====================================================================
# 6. ЗАПУСК ФИНАЛЬНОГО БАТТЛА АРХИТЕКТУР
# =====================================================================
print("Запуск процесса обучения моделей (это займет пару секунд)...")

# Создаем экземпляры обеих архитектур с одинаковой емкостью скрытого слоя (4)
advanced_custom_model = AdvancedCustomRNN(hidden_dim=4)
lstm_model = StandardLSTM(hidden_dim=4)

# Обучаем обе сети
custom_train, custom_test = train_model(advanced_custom_model)
lstm_train, lstm_test = train_model(lstm_model)

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МОДЕРНИЗИРОВАННЫХ АРХИТЕКТУР")
print("="*50)
print(f"Продвинутая кастомная память | Train MSE: {custom_train:.6f} | Test MSE: {custom_test:.6f}")
print(f"Стандартная PyTorch LSTM     | Train MSE: {lstm_train:.6f} | Test MSE: {lstm_test:.6f}")
print("="*50)
