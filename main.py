import torch
# Импортируем ваш кастомный слой
from parametric_memory_gate import ParametricMemoryGate

# 1. Инициализация слоя
my_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)

# 2. Прямой проход через тензор данных
mock_inputs = torch.tensor([-2.0, 0.0, 1.0, 3.0], dtype=torch.float32)
outputs = my_gate(mock_inputs)

print("Входные сигналы: ", mock_inputs.numpy())
print("Выход из гейта:  ", outputs.detach().numpy())
print("Текущее состояние слоя: ", my_gate)
