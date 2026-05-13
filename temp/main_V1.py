import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Установка воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. КАСТОМНЫЙ ПЛАГИН PYTORCH ДЛЯ ВАШЕЙ ФУНКЦИИ АКТИВАЦИИ
# =====================================================================
class ParametricMemoryGate(nn.Module):
    """
    Кастомный слой активации: f(x) = base^(x + shift) / (1 + base^(x + shift))
    base и shift являются обучаемыми параметрами.
    """
    def __init__(self, initial_base=4.0, initial_shift=-1.0):
        super().__init__()
        # Чтобы base не ушел в отрицательные значения, обучаем сырой лог-параметр
        # base = 1.0 + exp(raw_base) -> гарантирует base > 1.0
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.tensor([raw_base_init], dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor([initial_shift], dtype=torch.float32))

    def forward(self, x):
        base = 1.0 + torch.exp(self.raw_base)
        # Ограничиваем степень для предотвращения переполнения (Inf/NaN)
        power = torch.clamp(x + self.shift, -20.0, 20.0)
        
        gate = (base ** power) / (1.0 + (base ** power))
        return gate

    def get_actual_base(self):
        with torch.no_grad():
            return (1.0 + torch.exp(self.raw_base)).item()

# =====================================================================
# 2. РЕКУРРЕНТНАЯ ЯЧЕЙКА ПАМЯТИ НА БАЗЕ ВАШЕГО ПЛАГИНА
# =====================================================================
class CustomRNNCell(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Линейные преобразования для обновления актуальности (relevance)
        self.w_x = nn.Linear(input_dim, hidden_dim)
        self.b_x = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Наш кастомный плагин активации памяти
        self.memory_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        
        # Выходной линейный слой для финального прогноза цены
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, seq_x):
        # seq_x имеет форму: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = seq_x.size()
        
        # Инициализация скрытых состояний нулями
        memory = torch.zeros(batch_size, self.hidden_dim, device=seq_x.device)
        relevance = torch.zeros(batch_size, self.hidden_dim, device=seq_x.device)
        
        # Проход по временной последовательности (шаг за шагом)
        for t in range(seq_len):
            inp = seq_x[:, t, :] # Текущий временной шаг
            
            # Обновление актуальности по формуле: relevance = b_x(rel) + w_x(inp)
            relevance = self.b_x(relevance) + self.w_x(inp)
            
            # Фильтрация через ваш адаптивный гейт
            gate = self.memory_gate(relevance)
            
            # Обновление состояния памяти системы
            memory = (memory * gate) + inp
            
        # Возвращаем финальное предсказание из накопленной памяти
        output = self.fc_out(memory)
        return output

# =====================================================================
# 3. ЗАГРУЗКА И ПОДГОТОВКА РЕАЛЬНЫХ ФИНАНСОВЫХ ДАННЫХ
# =====================================================================
print("Скачивание котировок AAPL...")
# Берем исторические дневные данные акций Apple за последние несколько лет
df = yf.download("AAPL", start="2022-01-01", end="2025-01-01")
prices = df['Close'].values.reshape(-1, 1)

# Нормализация данных в диапазон (0, 1) для стабильности градиентов
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Функция для генерации датасета скользящего окна
def create_sequences(data, seq_length=10):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        Y.append(data[i + seq_length])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

# Окно анализа — последние 10 дней для предсказания цены на 11-й день
X, Y = create_sequences(prices_scaled, seq_length=10)

# Разделение на обучающую и тестовую выборки (80% / 20%)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# =====================================================================
# 4. ЦИКЛ ОБУЧЕНИЯ МОДЕЛИ В PYTORCH
# =====================================================================
model = CustomRNNCell(input_dim=1, hidden_dim=1)
criterion = nn.MSELoss() # Среднеквадратичная ошибка
optimizer = optim.Adam(model.parameters(), lr=0.01) # Используем Adam

epochs = 100
print("\n--- СТАРТ ОБУЧЕНИЯ НА РЕАЛЬНОМ РЫНКЕ ---")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        actual_base = model.memory_gate.get_actual_base()
        actual_shift = model.memory_gate.shift.item()
        print(f"Эпоха {epoch+1:03d} | Train Loss: {loss.item():.6f} | "
              f"Изученная Base: {actual_base:.3f} | Изученный Shift: {actual_shift:.3f}")

# =====================================================================
# 5. ТЕСТИРОВАНИЕ НА ОТЛОЖЕННЫХ ДАННЫХ
# =====================================================================
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, Y_test)
    print(f"\n--- ТЕСТ ЗАВЕРШЕН ---")
    print(f"Финальная ошибка на тесте (MSE): {test_loss.item():.6f}")
