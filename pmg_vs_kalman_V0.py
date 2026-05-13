import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from filterpy.kalman import KalmanFilter

# =====================================================================
# 1. ОРИГИНАЛЬНАЯ РЕАЛИЗАЦИЯ PMG НА PYTORCH (ВАШ КОД)
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
        gate = torch.clamp(gate, eps, 1.0 - eps)
        return gate

    def get_parameters(self) -> Tuple[float, float]:
        with torch.no_grad():
            actual_base = 1.0 + torch.exp(self.raw_base).item()
            actual_shift = self.shift.item()
            return actual_base, actual_shift

# =====================================================================
# 2. ОПТИМИЗИРОВАННАЯ NUMPY-РЕАЛИЗАЦИЯ PMG ДЛЯ РЕАЛЬНОГО ВРЕМЕНИ
# =====================================================================
class NumPyParametricMemoryGate:
    def __init__(self, base: float = 4.0, shift: float = -1.0):
        self.base = base
        self.shift = shift
        self.eps = 1e-7

    def forward_step(self, x: float) -> float:
        """Сверхбыстрый пошаговый расчет без оверхеда PyTorch"""
        power = np.clip(x + self.shift, -20.0, 20.0)
        base_pow = self.base ** power
        gate = base_pow / (1.0 + base_pow)
        return np.clip(gate, self.eps, 1.0 - self.eps)

# =====================================================================
# 3. СЛУЖЕБНЫЙ СКРИПТ СРАВНЕНИЯ И СНЯТИЯ МЕТРИК СКОРОСТИ
# =====================================================================
def run_ultimate_benchmark():
    # Настройка симуляции (100 Гц телеметрия, 1000 шагов)
    dt = 0.01  
    timesteps = 1000
    t = np.linspace(0, timesteps * dt, timesteps)
    
    # Истинный сигнал (уровень связи LQ / RSSI)
    true_signal = 0.9 - 0.4 * (np.exp(-(t - 5)**2 / 2))
    
    # Генерация асимметричного шума FPV-дрона (просадки вниз)
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, size=timesteps)
    noise += np.where(np.random.rand(timesteps) > 0.95, -0.3, 0.0) 
    z_measurements = np.clip(true_signal + noise, 0.0, 1.0)

    # Инициализация моделей
    init_base = 5.0
    init_shift = -0.1
    
    pmg_torch = ParametricMemoryGate(initial_base=init_base, initial_shift=init_shift)
    pmg_torch.eval()
    
    # Получаем точные веса для NumPy-версии, чтобы графики совпали на 100%
    learned_base, learned_shift = pmg_torch.get_parameters()
    pmg_numpy = NumPyParametricMemoryGate(base=learned_base, shift=learned_shift)

    torch_input = torch.tensor(z_measurements, dtype=torch.float32)

    # Фильтр Калмана
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[0.9]])
    kf.A = np.array([[1.0]])
    kf.H = np.array([[1.0]])
    kf.R = np.array([[0.05**2]])
    kf.Q = np.array([[0.001]])
    kf.P *= 1.0

    # ---------------- BENCHMARK 1: Фильтр Калмана ----------------
    kf_outputs = []
    start_kf = time.perf_counter()
    for z in z_measurements:
        kf.predict()
        kf.update(z)
        kf_outputs.append(kf.x[0, 0])
    end_kf = time.perf_counter()

    # ---------------- BENCHMARK 2: Ваша PMG (Пошагово в PyTorch) ----------------
    pmg_torch_step_outputs = []
    start_pmg_torch_step = time.perf_counter()
    with torch.no_grad():
        for i in range(timesteps):
            val = torch_input[i].unsqueeze(0) 
            out = pmg_torch(val)
            pmg_torch_step_outputs.append(out.item())
    end_pmg_torch_step = time.perf_counter()
    
    # ---------------- BENCHMARK 3: Ваша PMG (Пакетно в PyTorch) ----------------
    start_pmg_torch_batch = time.perf_counter()
    with torch.no_grad():
        pmg_torch_batch_outputs = pmg_torch(torch_input).numpy()
    end_pmg_torch_batch = time.perf_counter()

    # ---------------- BENCHMARK 4: Оптимизированная PMG (Пошагово в NumPy) ----------------
    pmg_np_outputs = []
    start_pmg_np = time.perf_counter()
    for i in range(timesteps):
        out = pmg_numpy.forward_step(z_measurements[i])
        pmg_np_outputs.append(out)
    end_pmg_np = time.perf_counter()

    # Вычисление метрик времени в миллисекундах и микросекундах
    kf_total = (end_kf - start_kf) * 1000
    torch_step_total = (end_pmg_torch_step - start_pmg_torch_step) * 1000
    torch_batch_total = (end_pmg_torch_batch - start_pmg_torch_batch) * 1000
    np_step_total = (end_pmg_np - start_pmg_np) * 1000

    # Вывод результатов в консоль
    print("\n" + "="*65)
    print("  СРАВНИТЕЛЬНЫЙ БЕНЧМАРК СКОРОСТИ ВЫПОЛНЕНИЯ (1000 итераций)")
    print("="*65)
    print(f" 1. Фильтр Калмана (Пошагово):        {kf_total:7.3f} мс  ({(kf_total*1000)/timesteps:5.2f} мкс/шаг)")
    print(f" 2. Ваша PMG (Пошагово в PyTorch):    {torch_step_total:7.3f} мс  ({(torch_step_total*1000)/timesteps:5.2f} мкс/шаг)")
    print(f" 3. Ваша PMG (Пакетно в PyTorch):     {torch_batch_total:7.3f} мс  ({(torch_batch_total*1000)/timesteps:5.2f} мкс/шаг)")
    print(f" 4. Оптимизированная PMG (NumPy):     {np_step_total:7.3f} мс  ({(np_step_total*1000)/timesteps:5.2f} мкс/шаг)")
    print("="*65)

    # Визуализация графиков
    plt.figure(figsize=(13, 6.5))
    plt.plot(t, z_measurements, label='Входной зашумленный сигнал телеметрии', color='red', alpha=0.25, linestyle='None', marker='.')
    plt.plot(t, true_signal, label='Истинный чистый трек', color='black', linewidth=2, linestyle='--')
    plt.plot(t, kf_outputs, label='Фильтр Калмана (Инерционный)', color='blue', linewidth=1.5)
    plt.plot(t, pmg_np_outputs, label='Ваш PMG (Оптимизированный NumPy)', color='green', linewidth=2)
    
    plt.title('Сравнение фильтрации данных: Фильтр Калмана vs Parametric Memory Gate (PMG)', fontsize=12)
    plt.xlabel('Время (секунды)', fontsize=10)
    plt.ylabel('Амплитуда сигнала (Телеметрия)', fontsize=10)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    run_ultimate_benchmark()
