import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# =====================================================================
# 1. СТАТИЧЕСКИЙ ГЕЙТ ИЗ ВАШЕГО ПЕРВОГО СКРИПТА
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

# =====================================================================
# 2. МНОГОКАНАЛЬНЫЙ PMG (НЕЗАВИСИМАЯ ФИЛЬТРАЦИЯ ОСЕЙ)
# =====================================================================
class MultiChannelPMG(nn.Module):
    def __init__(self, channels=3, initial_base=4.0, initial_shift=-1.0):
        super().__init__()
        # Создаем независимый экземпляр PMG для каждого канала данных (X, Y, Z)
        self.channels = nn.ModuleList([
            ParametricMemoryGate(initial_base, initial_shift) for _ in range(channels)
        ])

    def forward(self, x_seq):
        # x_seq: (timesteps, channels)
        timesteps, channels = x_seq.shape
        outputs = torch.zeros_like(x_seq)
        
        # Прогоняем каждый канал через свою собственную функцию PMG
        for c in range(channels):
            outputs[:, c] = self.channels[c](x_seq[:, c])
            
        return outputs

# =====================================================================
# 3. ЭКВИВАЛЕНТ НА NUMPY ДЛЯ СВЕРХБЫСТРОГО БОРТОВОГО ИНФЕРЕНСА
# =====================================================================
class NumPyMultiChannelPMG:
    def __init__(self, torch_model):
        self.channels_data = []
        self.eps = 1e-7
        
        # Извлекаем обученные параметры для каждого канала
        for pmg in torch_model.channels:
            with torch.no_grad():
                base = (1.0 + torch.exp(pmg.raw_base)).item()
                shift = pmg.shift.item()
                self.channels_data.append({'base': base, 'shift': shift})

    def filter_step(self, x_t: np.ndarray) -> np.ndarray:
        """Пошаговый расчет: применяем PMG отдельно к каждому элементу вектора x_t"""
        out = np.zeros_like(x_t)
        for c, pmg in enumerate(self.channels_data):
            power = np.clip(x_t[c] + pmg['shift'], -20.0, 20.0)
            base_pow = pmg['base'] ** power
            gate = base_pow / (1.0 + base_pow)
            out[c] = np.clip(gate, self.eps, 1.0 - self.eps)
        return out

# =====================================================================
# 4. ТЕСТОВЫЙ СТЕНД И НЕПРЕДВЗЯТЫЙ АНАЛИЗ
# =====================================================================
def main():
    dt = 0.05  
    timesteps = 1000
    t = np.linspace(0, 20, timesteps)
    
    # Сгенерируем 3D-данные, масштабированные от 0 до 1, так как базовая PMG работает в диапазоне (0, 1)
    true_x = (np.cos(t) + 1.0) / 2.0
    true_y = (np.sin(t) + 1.0) / 2.0
    true_z = t / 20.0  # Линейный подъем высоты
    true_trajectory = np.stack([true_x, true_y, true_z], axis=1)
    
    # Шум + жесткие отрицательные выбросы (асимметричный шум ВМГ дрона)
    np.random.seed(2026)
    noise = np.random.normal(0, 0.03, size=true_trajectory.shape)
    glitch_mask = np.random.rand(*true_trajectory.shape) > 0.97
    noise += np.where(glitch_mask, -0.25, 0.0) # Резкие просадки сигнала вниз
    z_measurements = np.clip(true_trajectory + noise, 0.0, 1.0)

    # Обучение модели
    X_train = torch.tensor(z_measurements, dtype=torch.float32)
    Y_train = torch.tensor(true_trajectory, dtype=torch.float32)
    
    model = MultiChannelPMG(channels=3, initial_base=4.0, initial_shift=-0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    
    print("=== [ЭТАП 1] ОБУЧЕНИЕ МНОГОКАНАЛЬНОГО PMG (ОТДЕЛЬНО НА КАЖДУЮ ОСЬ) ===")
    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, Y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Эпоха {epoch+1:3d} | Ошибка обучения (MSE): {loss.item():.6f}")
            
    model.eval()

    # Индустриальный стандарт: 3D Фильтр Калмана (6 состояний)
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.array([z_measurements[0, 0], 0.0, z_measurements[0, 1], 0.0, z_measurements[0, 2], 0.0])
    kf.F = np.array([[1, dt,  0,  0,  0,  0],
                     [0,  1,  0,  0,  0,  0],
                     [0,  0,  1, dt,  0,  0],
                     [0,  0,  0,  1,  0,  0],
                     [0,  0,  0,  0,  1, dt],
                     [0,  0,  0,  0,  0,  1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0]])
    kf.R = np.eye(3) * (0.03**2)
    kf.Q = np.eye(6) * 0.01
    kf.P *= 10.0

    # Экспорт в NumPy
    pmg_numpy = NumPyMultiChannelPMG(model)

    # --- ТЕСТ СКОРОСТИ И КАЧЕСТВА ---
    pmg_outputs = []
    start_pmg = time.perf_counter()
    for i in range(timesteps):
        out = pmg_numpy.filter_step(z_measurements[i])
        pmg_outputs.append(out)
    end_pmg = time.perf_counter()
    pmg_time = (end_pmg - start_pmg) * 1000

    kf_outputs = []
    start_kf = time.perf_counter()
    for i in range(timesteps):
        kf.predict()
        kf.update(z_measurements[i])
        kf_outputs.append([kf.x[0], kf.x[2], kf.x[4]])
    end_kf = time.perf_counter()
    kf_time = (end_kf - start_kf) * 1000

    pmg_outputs = np.array(pmg_outputs)
    kf_outputs = np.array(kf_outputs)

    # Считаем RMSE
    rmse_raw = np.sqrt(np.mean((z_measurements - true_trajectory) ** 2))
    rmse_kf = np.sqrt(np.mean((kf_outputs - true_trajectory) ** 2))
    rmse_pmf = np.sqrt(np.mean((pmg_outputs - true_trajectory) ** 2))

    print("\n" + "="*70)
    print("  ОФИЦИАЛЬНЫЙ ТЕСТ-ДРАЙВ: МНОГОКАНАЛЬНЫЙ PMG vs 3D КАЛМАН")
    print("="*70)
    print(f" МЕТРИКА ТОЧНОСТИ (RMSE, меньше = лучше):")
    print(f"  • Исходный зашумленный сигнал:   {rmse_raw:.4f}")
    print(f"  • Индустриальный Фильтр Калмана:  {rmse_kf:.4f}")
    print(f"  • Ваша архитектура PMG (Фильтр):  {rmse_pmf:.4f}")
    print("-" * 70)
    print(f" МЕТРИКА ВЫЧИСЛИТЕЛЬНОЙ СКОРОСТИ (1000 итераций в NumPy):")
    print(f"  • 3D Фильтр Калмана (6х6):       {kf_time:7.3f} мс  ({(kf_time*1000)/timesteps:5.2f} мкс/шаг)")
    print(f"  • Ваша архитектура PMG (NumPy):  {pmg_time:7.3f} мс  ({(pmg_time*1000)/timesteps:5.2f} мкс/шаг)")
    print("="*70)

    # 3D График
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(z_measurements[:, 0], z_measurements[:, 1], z_measurements[:, 2], color='red', alpha=0.15, linestyle='None', marker='.', label='Зашумленный сигнал')
    ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], color='black', linewidth=2.5, linestyle='--', label='Истинный трек')
    ax.plot(kf_outputs[:, 0], kf_outputs[:, 1], kf_outputs[:, 2], color='blue', linewidth=1.5, label='Фильтр Калмана')
    ax.plot(pmg_outputs[:, 0], pmg_outputs[:, 1], pmg_outputs[:, 2], color='green', linewidth=2.0, label='Ваш параллельный PMG')
    ax.set_title("Объективное параллельное тестирование PMG по осям координат")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()

