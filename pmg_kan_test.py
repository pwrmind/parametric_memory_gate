import torch
import torch.nn as nn
import numpy as np
import time

# Фиксация сида для строгой воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

# 1. Исправленный параметрический гейт PMG для KAN (работает со всеми знаками x)
class PMGKANActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Параметры создаются для каждой отдельной связи между входом и выходом
        self.a = nn.Parameter(torch.full((in_features, out_features), 4.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.full((in_features, out_features), -1.0, dtype=torch.float32))
        # Вес масштаба, позволяющий функции аппроксимировать отрицательные значения
        self.w = nn.Parameter(torch.randn(in_features, out_features) * 0.1)

    def forward(self, x):
        # x имеет размерность [batch_size, in_features]
        # Расширяем x до [batch_size, in_features, 1] для связи с out_features
        x_unsq = x.unsqueeze(-1)
        
        # Стабилизация параметров
        a_safe = torch.clamp(self.a, min=1.001, max=20.0)
        b_safe = torch.abs(self.b) + 1e-5
        
        # МАТЕМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ: 
        # Использование torch.pow вместо экспоненты корректно считает a^x при x < 0.
        # Если x = -2, то a^-2 = 1 / (a^2), функция больше не падает в глухой ноль.
        ax = torch.pow(a_safe, x_unsq)
        pmg = ax / (b_safe + ax)
        
        # Возвращаем масштабированный сигнал для каждой связи
        return pmg * self.w

# 2. Полносвязный слой PMG-KAN
class PM_KANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Каждая связь имеет свою функцию PMG
        self.pmg_kan = PMGKANActivation(in_features, out_features)
        # Смещение (bias) на выходе слоя
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Получаем тензор: [batch_size, in_features, out_features]
        activated_connections = self.pmg_kan(x)
        # Суммируем по входным признакам согласно теореме Колмогорова-Арнольда
        return torch.sum(activated_connections, dim=1) + self.bias

# Сеть PMG-KAN
class PMG_KAN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = PM_KANLayer(in_dim, hidden_dim)
        self.layer2 = PM_KANLayer(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)

# 3. Классический MLP для честного сравнения
class BaselineMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(), # Промышленный стандарт (Swish)
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# 4. Генерация данных: аппроксимация сложной нелинейной функции
X = torch.randn(3000, 5)
# Целевая функция со скрещенными нелинейными членами (включает знаки, синусы, квадраты)
Y = (torch.sin(X[:, 0] * X[:, 1]) + torch.exp(X[:, 2] * 0.5) + torch.abs(X[:, 3]) - X[:, 4]**2).unsqueeze(-1)

X_train, X_test = X[:2400], X[2400:]
y_train, y_test = Y[:2400], Y[2400:]

# 5. Тестирование моделей
print("=== ЗАПУСК ИСПРАВЛЕННОГО БЕНЧМАРКА: PMG-KAN ПРОТИВ MLP ===")

# --- Тест 1: PMG-KAN ---
kan_model = PMG_KAN(in_dim=5, hidden_dim=16, out_dim=1)
criterion = nn.MSELoss()

# Настройка оптимизатора с разделенным LR для параметров гейтов PMG
pmg_params = [kan_model.layer1.pmg_kan.a, kan_model.layer1.pmg_kan.b, 
              kan_model.layer2.pmg_kan.a, kan_model.layer2.pmg_kan.b]
base_params = [p for p in kan_model.parameters() if not any(p is pmg for pmg in pmg_params)]

kan_optimizer = torch.optim.AdamW([
    {'params': base_params, 'lr': 0.01, 'weight_decay': 1e-4},
    {'params': pmg_params, 'lr': 0.04}
])

start_kan = time.time()
for epoch in range(250):
    kan_model.train()
    kan_optimizer.zero_grad()
    loss = criterion(kan_model(X_train), y_train)
    loss.backward()
    kan_optimizer.step()

kan_time = time.time() - start_kan
kan_model.eval()
with torch.no_grad():
    kan_test_loss = criterion(kan_model(X_test), y_test).item()

# --- Тест 2: MLP ---
mlp_model = BaselineMLP(in_dim=5, hidden_dim=16, out_dim=1)
mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=0.01, weight_decay=1e-4)

start_mlp = time.time()
for epoch in range(250):
    mlp_model.train()
    mlp_optimizer.zero_grad()
    loss = criterion(mlp_model(X_train), y_train)
    loss.backward()
    mlp_optimizer.step()

mlp_time = time.time() - start_mlp
mlp_model.eval()
with torch.no_grad():
    mlp_test_loss = criterion(mlp_model(X_test), y_test).item()

# Финальный вывод результатов
print(f"\nРезультаты PMG-KAN:")
print(f"-> Test MSE: {kan_test_loss:.5f} | Время: {kan_time:.2f} сек")
print(f"Выученные 'a' (среднее слоя 1): {kan_model.layer1.pmg_kan.a.mean().item():.2f}")
print(f"Выученные 'b' (среднее слоя 1): {kan_model.layer1.pmg_kan.b.mean().item():.2f}")

print(f"\nРезультаты Baseline MLP (SiLU):")
print(f"-> Test MSE: {mlp_test_loss:.5f} | Время: {mlp_time:.2f} сек")
