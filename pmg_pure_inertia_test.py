import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# =====================================================================
# 1. ЧИСТАЯ РЕАЛИЗАЦИЯ PMG ИЗ ВАШЕГО ПЕРВОГО СКРИПТА
# =====================================================================
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
        return torch.clamp(gate, eps, 1.0 - eps)

    def get_parameters(self):
        with torch.no_grad():
            actual_base = 1.0 + torch.exp(self.raw_base).item()
            actual_shift = self.shift.item()
            return actual_base, actual_shift

# =====================================================================
# 2. ИСПЫТАТЕЛЬНЫЙ СТЕНД И АВТОМАТИЧЕСКИЙ ПОДБОР ПАРАМЕТРОВ
# =====================================================================
def main():
    timesteps = 300
    t = np.arange(timesteps)
    
    # Ступенчатый входной сигнал (Скачок на 100-м шаге)
    input_signal = np.zeros(timesteps)
    input_signal[100:] = 1.0 

    # --- 1. ВЫЧИСЛЕНИЕ ДЕЛЬТЫ (СКОРОСТИ ИЗМЕНЕНИЯ СИГНАЛА) ---
    delta_signal = np.zeros(timesteps)
    for i in range(1, timesteps):
        delta_signal[i] = input_signal[i] - input_signal[i-1]

    # --- 2. РАСЧЕТ ИНДУСТРИАЛЬНОГО ЭТАЛОНА (ФИЛЬТР КАЛМАНА) ---
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[0.0]])
    kf.A = np.array([[1.0]])
    kf.H = np.array([[1.0]])
    kf.R = np.array([[0.1**2]])
    kf.Q = np.array([[0.005]])   
    kf.P *= 1.0

    kf_outputs_raw = []
    for z in input_signal:
        kf.predict()
        kf.update(z)
        kf_outputs_raw.append(kf.x)
        
    # ИСПРАВЛЕНО: Принудительно выпрямляем в одномерный массив (300,)
    kf_outputs = np.array(kf_outputs_raw).flatten()

    # --- 3. АВТОМАТИЧЕСКИЙ ПОДБОР ПАРАМЕТРОВ ДЛЯ ДИФФЕРЕНЦИАЛЬНОЙ PMG ---
    X_train_delta = torch.tensor(delta_signal, dtype=torch.float32)
    # ИСПРАВЛЕНО: Тензоры теперь строго одинаковой формы (300,)
    Y_target = torch.tensor(kf_outputs, dtype=torch.float32)
    
    pmg_model = ParametricMemoryGate(initial_base=4.0, initial_shift=-0.5)
    optimizer = optim.Adam(pmg_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("=== [ЭТАП 1] ОБУЧЕНИЕ ДИФФЕРЕНЦИАЛЬНОЙ PMG МОДЕЛИ ===")
    pmg_model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        preds = pmg_model(X_train_delta)
        loss = criterion(preds, Y_target)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            b, s = pmg_model.get_parameters()
            print(f"Эпоха {epoch+1:3d} | Ошибка аппроксимации (MSE): {loss.item():.6f} | base: {b:.4f} | shift: {s:.4f}")

    pmg_model.eval()
    opt_base, opt_shift = pmg_model.get_parameters()
    
    with torch.no_grad():
        pmg_final_outputs = pmg_model(X_train_delta).numpy()

    print("\n=== [ЭТАП 2] СРАВНИТЕЛЬНЫЙ АНАЛИЗ ФОРМЫ СИГНАЛА ===")
    print(f"Финал обучения PMG -> Оптимальные параметры: base = {opt_base:.4f}, shift = {opt_shift:.4f}")
    print("-" * 75)
    print(f"Поведение дифференциального PMG-фильтра:")
    print(f" • Статика до удара (шаг 50, delta=0.0):       {pmg_final_outputs[50]:.4f}")
    print(f" • Момент удара (шаг 100, delta=1.0):          {pmg_final_outputs[100]:.4f}")
    print(f" • Полка после удара (шаг 150, delta=0.0):     {pmg_final_outputs[150]:.4f}")
    print("-" * 75)

    # --- ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---
    plt.figure(figsize=(12, 7))
    plt.plot(t, input_signal, label='Входной сигнал (Ступенька 0.0 -> 1.0)', color='red', linewidth=2, linestyle=':')
    plt.plot(t, delta_signal, label='Дельта входа (Скорость изменения \\Delta x)', color='magenta', linewidth=1.5)
    plt.plot(t, kf_outputs, label='Фильтр Калмана (Временная инерция)', color='blue', linewidth=2)
    plt.plot(t, pmg_final_outputs, label='Дифференциальная PMG (На дельте входа)', color='green', linewidth=2.5)

    plt.title('Тест дифференциальной модификации: Реакция на скорость изменения сигнала', fontsize=12)
    plt.xlabel('Шаги времени (Итерации)', fontsize=10)
    plt.ylabel('Амплитуда', fontsize=10)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()
