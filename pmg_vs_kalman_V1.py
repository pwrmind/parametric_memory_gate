import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# =====================================================================
# 1. РЕКУРРЕНТНАЯ ЯЧЕЙКА PMC НА PYTORCH (ДЛЯ ОБУЧЕНИЯ)
# =====================================================================
class ParametricMemoryCell(nn.Module):
    def __init__(self, initial_base: float = 4.0, initial_shift: float = -0.5):
        super().__init__()
        if initial_base <= 1.0:
            raise ValueError("initial_base must be strictly greater than 1.0")
            
        # Оптимизация в log-space для стабильности градиентов
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.tensor([raw_base_init], dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor([initial_shift], dtype=torch.float32))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход по всей последовательности данных (векторизованно)
        x_seq: тензор размерности (timesteps)
        """
        base = 1.0 + torch.exp(self.raw_base)
        eps = 1e-7
        
        timesteps = x_seq.size(0)
        outputs = torch.zeros(timesteps)
        
        # Начальное состояние памяти (первая точка измерения)
        h_t = x_seq[0]
        outputs[0] = h_t
        
        # Рекуррентный цикл фильтрации
        for t in range(1, timesteps):
            x_t = x_seq[t]
            
            # Гейт активации зависит от текущего входа и разности с прошлой памятью
            # Это позволяет гейту реагировать на скорость изменения сигнала
            power = torch.clamp((x_t - h_t) + self.shift, -20.0, 20.0)
            gate = (base ** power) / (1.0 + (base ** power))
            gate = torch.clamp(gate, eps, 1.0 - eps)
            
            # Обновление памяти (Линейная интерполяция между прошлым и настоящим)
            h_t = gate * h_t + (1.0 - gate) * x_t
            outputs[t] = h_t
            
        return outputs

    def get_parameters(self) -> Tuple[float, float]:
        with torch.no_grad():
            actual_base = 1.0 + torch.exp(self.raw_base).item()
            actual_shift = self.shift.item()
            return actual_base, actual_shift

# =====================================================================
# 2. ОПТИМИЗИРОВАННАЯ ЯЧЕЙКА PMC НА NUMPY (ДЛЯ РЕАЛЬНОГО ВРЕМЕНИ)
# =====================================================================
class NumPyParametricMemoryCell:
    def __init__(self, base: float, shift: float):
        self.base = base
        self.shift = shift
        self.eps = 1e-7
        self.h_t = None

    def reset(self, initial_value: float):
        """Сброс памяти при старте приема телеметрии"""
        self.h_t = initial_value

    def filter_step(self, x_t: float) -> float:
        """Сверхбыстрый расчет одного шага на борту дрона"""
        if self.h_t is None:
            self.h_t = x_t
            return x_t
            
        power = np.clip((x_t - self.h_t) + self.shift, -20.0, 20.0)
        base_pow = self.base ** power
        gate = base_pow / (1.0 + base_pow)
        gate = np.clip(gate, self.eps, 1.0 - self.eps)
        
        # Обновление состояния памяти
        self.h_t = gate * self.h_t + (1.0 - gate) * x_t
        return self.h_t

# =====================================================================
# 3. СЦЕНАРИЙ: ГЕНЕРАЦИЯ, ОБУЧЕНИЕ И БЕНЧМАРК
# =====================================================================
def main():
    # --- 1. Генерация полетного датасета ---
    dt = 0.01
    timesteps = 1000
    t = np.linspace(0, timesteps * dt, timesteps)
    
    # Истинный трек телеметрии (плавное изменение)
    true_signal = 0.9 - 0.4 * (np.exp(-(t - 5)**2 / 2))
    
    # Асимметричные жесткие помехи от ВМГ (винтомоторной группы)
    np.random.seed(42)
    noise = np.random.normal(0, 0.04, size=timesteps)
    noise += np.where(np.random.rand(timesteps) > 0.94, -0.25, 0.0)
    z_measurements = np.clip(true_signal + noise, 0.0, 1.0)

    # Подготовка тензоров для PyTorch
    X_train = torch.tensor(z_measurements, dtype=torch.float32)
    Y_train = torch.tensor(true_signal, dtype=torch.float32)

    # --- 2. Инициализация и Обучение PMC модели ---
    pmc_model = ParametricMemoryCell(initial_base=3.0, initial_shift=-0.1)
    optimizer = optim.Adam(pmc_model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    print("=== СТАРТ ОБУЧЕНИЯ ПАРАМЕТРОВ ГЕЙТА ===")
    b_start, s_start = pmc_model.get_parameters()
    print(f"Стартовые параметры: base = {b_start:.3f}, shift = {s_start:.3f}")
    
    pmc_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = pmc_model(X_train)
        loss = criterion(predictions, Y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            b, s = pmc_model.get_parameters()
            print(f"Эпоха {epoch+1:3d} | Loss (MSE): {loss.item():.6f} | base: {b:.3f} | shift: {s:.3f}")

    # Фиксация обученных весов
    pmc_model.eval()
    learned_base, learned_shift = pmc_model.get_parameters()
    print("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
    print(f"Оптимальные параметры: base = {learned_base:.4f}, shift = {learned_shift:.4f}")

    # --- 3. Тестирование скорости на NumPy версии с обученными весами ---
    pmc_numpy = NumPyParametricMemoryCell(base=learned_base, shift=learned_shift)
    
    # Замер времени выполнения
    pmc_np_outputs = []
    start_time = time.perf_counter()
    
    pmc_numpy.reset(z_measurements[0])
    for i in range(timesteps):
        out = pmc_numpy.filter_step(z_measurements[i])
        pmc_np_outputs.append(out)
        
    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000
    
    print("\n" + "="*60)
    print(f" БЕНЧМАРК ОПТИМИЗИРОВАННОЙ РЕКУРРЕНТНОЙ PMC (NumPy)")
    print("="*60)
    print(f" Общее время (1000 шагов):      {total_time_ms:.3f} мс")
    print(f" Скорость одной итерации:       {(total_time_ms * 1000) / timesteps:.2f} мкс/шаг")
    print("="*60)

    # --- 4. Визуализация качества фильтрации памяти ---
    plt.figure(figsize=(13, 6.5))
    plt.plot(t, z_measurements, label='Зашумленный вход (Телеметрия)', color='red', alpha=0.25, linestyle='None', marker='.')
    plt.plot(t, true_signal, label='Целевой чистый трек', color='black', linewidth=2, linestyle='--')
    plt.plot(t, pmc_np_outputs, label='Выход PMC (Рекуррентный NumPy фильтр)', color='darkgreen', linewidth=2)
    
    plt.title('Адаптивная рекуррентная фильтрация с помощью Parametric Memory Cell (PMC)', fontsize=12)
    plt.xlabel('Время (секунды)')
    plt.ylabel('Амплитуда')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    main()
