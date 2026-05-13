import numpy as np
import matplotlib.pyplot as plt

# 1. Генерация цикличных данных (синусоида с шумом)
np.random.seed(42)
timesteps = 100
time_series = np.sin(np.linspace(0, 20, timesteps)) * 10 + np.random.normal(0, 0.5, timesteps)

X_data = time_series[:-1]
Y_data = time_series[1:]

# 2. Инициализация весов и ПАРАМЕТРОВ ФУНКЦИИ АКТИВАЦИИ
w_x = 0.253   # Стартуем с прошлых обученных значений
b_x = -0.051
w_out = 0.254

# Параметры вашей функции активации
# Обучаем b_raw, чтобы получить base = 1 + exp(1.1) ≈ 4.0
b_raw = 1.10
shift = -1.0

# Настройки обучения
lr = 0.001
epochs = 50

print("--- СТАРТ АДАПТИВНОГО ОБУЧЕНИЯ ПАМЯТИ ---")

for epoch in range(epochs):
    # Прямой расчет текущего основания
    base = 1.0 + np.exp(b_raw)
    ln_base = np.log(base)
    
    memory = 0.0
    relevance = 0.0
    
    states_mem = []
    states_rel = []
    states_gate = []
    predictions = []
    
    # --- ПРЯМОЙ ХОД (Forward Pass) ---
    for t in range(len(X_data)):
        inp = X_data[t]
        
        relevance = (relevance * b_x) + (inp * w_x)
        
        # Ваша функция: base^(relevance + shift) / (1 + base^(relevance + shift))
        power = relevance + shift
        power = np.clip(power, -20, 20) # защита от переполнения
        
        gate = (base ** power) / (1.0 + (base ** power))
        memory = (memory * gate) + inp
        pred = memory * w_out
        
        states_rel.append(relevance)
        states_gate.append(gate)
        states_mem.append(memory)
        predictions.append(pred)
        
    predictions = np.array(predictions)
    loss = np.mean((predictions - Y_data) ** 2)
    
    # --- ОБРАТНЫЙ ХОД (Backward Pass) ---
    dw_x, db_x, dw_out = 0.0, 0.0, 0.0
    db_raw, dshift = 0.0, 0.0
    
    for t in reversed(range(len(X_data))):
        error = predictions[t] - Y_data[t]
        dw_out += error * states_mem[t]
        
        d_mem = error * w_out
        
        if t > 0:
            d_gate = d_mem * states_mem[t-1]
            
            # Производная функции активации по аргументу (relevance + shift)
            # d_gate / d_power = gate * (1 - gate) * ln(base)
            g = states_gate[t]
            d_power = d_gate * g * (1.0 - g) * ln_base
            
            # Градиенты для весов сети
            dw_x += d_power * X_data[t]
            db_x += d_power * states_rel[t-1]
            
            # Градиенты для параметров САМОЙ функции
            dshift += d_power # производная по shift равна 1
            
            # Производная по base: gate * (1 - gate) * (power / base)
            d_base = d_gate * g * (1.0 - g) * ( (states_rel[t] + shift) / base )
            # Цепное правило для b_raw (так как base = 1 + exp(b_raw))
            db_raw += d_base * np.exp(b_raw)

    # Обновление всех параметров
    n = len(X_data)
    w_out -= lr * (dw_out / n)
    w_x -= lr * (dw_x / n)
    b_x -= lr * (db_x / n)
    shift -= lr * (dshift / n)
    b_raw -= lr * (db_raw / n)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Эпоха {epoch+1:02d} | Loss: {loss:.4f} | Base: {base:.3f} | Shift: {shift:.3f} | w_x: {w_x:.3f}")

print("\n--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---")



# # 1. Ваша кастомная функция активации и её производная
# def custom_activation(x):
#     # Клиппируем x, чтобы избежать переполнения (OverflowError) при больших степенях
#     x = np.clip(x, -20, 20)
#     return (4**(x - 1)) / (1 + 4**(x - 1))

# def custom_derivative(x):
#     # Производная вашей функции для обратного распространения ошибки
#     x = np.clip(x, -20, 20)
#     fx = custom_activation(x)
#     return fx * (1 - fx) * np.log(4)

# # 2. Генерация данных: цикличный сигнал (например, температура или продажи)
# np.random.seed(42)
# timesteps = 100
# # Базовый паттерн: повторяющиеся волны + небольшой шум
# time_series = np.sin(np.linspace(0, 20, timesteps)) * 10 + np.random.normal(0, 0.5, timesteps)

# # Исходные данные (X) и целевые значения (Y) - предсказать следующий шаг
# X_data = time_series[:-1]
# Y_data = time_series[1:]

# # 3. Инициализация обучаемых весов памяти
# w_x = 0.5   # Вес входного сигнала
# b_x = 0.1   # Вес затухания прошлой актуальности
# w_out = 0.2 # Вес выходного линейного слоя для предсказания

# # Скорость обучения (Learning Rate)
# lr = 0.001
# epochs = 50

# print("--- СТАРТ ОБУЧЕНИЯ ПАМЯТИ ---")

# for epoch in range(epochs):
#     # Инициализация состояний для текущей эпохи
#     memory = 0.0
#     relevance = 0.0
    
#     # Списки для сохранения истории шагов (нужны для бэкпропа)
#     states_mem = []
#     states_rel = []
#     states_gate = []
#     predictions = []
    
#     # --- ПРЯМОЙ ХОД (Forward Pass) ---
#     for t in range(len(X_data)):
#         inp = X_data[t]
        
#         # Обновление актуальности с учетом весов
#         relevance = (relevance * b_x) + (inp * w_x)
#         gate = custom_activation(relevance)
        
#         # Обновление памяти
#         memory = (memory * gate) + inp
        
#         # Финальное предсказание (линейная проекция из памяти)
#         pred = memory * w_out
        
#         # Сохраняем состояния
#         states_rel.append(relevance)
#         states_gate.append(gate)
#         states_mem.append(memory)
#         predictions.append(pred)
        
#     # Считаем общую ошибку (MSE)
#     predictions = np.array(predictions)
#     loss = np.mean((predictions - Y_data) ** 2)
    
#     # --- ОБРАТНЫЙ ХОД (Backward Pass / Градиентный спуск) ---
#     # Инициализируем градиенты нулями
#     dw_x, db_x, dw_out = 0.0, 0.0, 0.0
    
#     # Ошибка идет от конца к началу последовательности (BPTT)
#     for t in reversed(range(len(X_data))):
#         error = predictions[t] - Y_data[t]
        
#         # Градиент для выходного веса
#         dw_out += error * states_mem[t]
        
#         # Ошибка, пришедшая на слой памяти
#         d_mem = error * w_out
        
#         # Градиент через вашу функцию активации (Gate)
#         # На сколько изменение 'relevance' повлияло на итоговую ошибку
#         if t > 0:
#             d_gate = d_mem * states_mem[t-1]
#             d_rel = d_gate * custom_derivative(states_rel[t])
            
#             dw_x += d_rel * X_data[t]
#             db_x += d_rel * states_rel[t-1]

#     # Корректируем веса (усредняем градиент по длине выборки)
#     n = len(X_data)
#     w_out -= lr * (dw_out / n)
#     w_x -= lr * (dw_x / n)
#     b_x -= lr * (db_x / n)
    
#     if (epoch + 1) % 10 == 0 or epoch == 0:
#         print(f"Эпоха {epoch+1:02d} | Ошибка (Loss): {loss:.4f} | w_x: {w_x:.3f} | b_x: {b_x:.3f} | w_out: {w_out:.3f}")

# print("\n--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---")



# def memory_gate(x):
#     # Ваша функция активации (основание 4, сдвиг -1)
#     return (4**(x - 1)) / (1 + 4**(x - 1))

# # Имитируем состояние старой памяти (вектор из 4-х воспоминаний)
# old_memory = np.array([10.0, 50.0, 100.0, 20.0])

# # Индексы актуальности для каждого воспоминания (от очень важного до забытого)
# relevance = np.array([2.0, 1.0, 0.0, -2.0]) 

# # Применяем вашу функцию как фильтр
# gate_values = memory_gate(relevance)
# filtered_memory = old_memory * gate_values

# print("Коэффициенты удержания: ", np.round(gate_values, 3))
# print("Что осталось в памяти:  ", np.round(filtered_memory, 2))
# # Вывод покажет, как красиво 100 превратилось в фоновые 20, а 20 сжалось до архивных 0.3



# def custom_activation(x):
#     return (4**(x - 1)) / (1 + 4**(x - 1))

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def relu(x):
#     return np.maximum(0, x)

# # Test values
# x_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
# print("Custom:", custom_activation(x_vals))
# print("Sigmoid:", sigmoid(x_vals))
# print("ReLU:", relu(x_vals))

# def custom_activation(x):
#     return (4**(x - 1)) / (1 + 4**(x - 1))

# class CustomMemoryCell:
#     def __init__(self):
#         # Начальное состояние памяти системы
#         self.memory = 0.0
#         # Актуальность текущего воспоминания
#         self.relevance = 0.0

#     def step(self, input_signal, importance_impulse):
#         """
#         Один тактовыи шаг работы памяти
#         importance_impulse: > 0 если сигнал важен сейчас, < 0 если игнорируется
#         """
#         # 1. Изменяем индекс актуальности на основе внешнего импульса
#         self.relevance += importance_impulse
        
#         # 2. Считаем коэффициент удержания по ВАШЕИ формуле
#         gate = custom_activation(self.relevance)
        
#         # 3. Обновляем память: часть старои памяти удерживается + добавляется новая инфо
#         self.memory = (self.memory * gate) + input_signal
        
#         return self.memory, gate, self.relevance

# # --- Симуляция процесса ---
# cell = CustomMemoryCell()

# # Поток входящих сигналов (например, показания датчиков или эмбеддинги слов)
# stream = [10, 10, 10, 0, 0, 0, 0]
# # Импульсы внимания: сначала данные очень важны (+1.5), потом фокус теряется (-1.0)
# impulses = [1.5, 1.5, 1.5, -1.0, -1.0, -1.0, -1.0]

# print(f"{'Шаг':<5} | {'Вход':<5} | {'Импульс':<7} | {'Актуальность (X)':<16} | {'Gate (Удержание)':<16} | {'Состояние памяти':<10}")
# print("-" * 80)

# for i, (inp, imp) in enumerate(zip(stream, impulses)):
#     mem, g, rel = cell.step(inp, imp)
#     print(f"{i+1:<5} | {inp:<5} | {imp:<7} | {rel:<16.2f} | {g:<16.4f} | {mem:<10.2f}")
