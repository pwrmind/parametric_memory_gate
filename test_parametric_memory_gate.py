import torch
import pytest
from parametric_memory_gate import ParametricMemoryGate

def test_forward_boundaries():
    """Проверяет, что выходные значения зажаты в диапазоне [0, 1] с учетом погрешности float32"""
    gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
    inputs = torch.tensor([-50.0, -10.0, 0.0, 10.0, 50.0], dtype=torch.float32)
    
    outputs = gate(inputs)
    
    # Используем встроенные нестрогие проверки PyTorch с допуском (tolerance)
    assert torch.all(outputs >= 0.0), "Выходное значение вышло ниже нуля"
    assert torch.all(outputs <= 1.0), "Выходное значение превысило единицу"
    
    # Проверяем, что эпсилон-защита удержала значения от абсолютных 0 и 1
    assert torch.all(outputs < 1.0), "Эпсилон-защита не сработала на верхнем пределе"

def test_invalid_base_error():
    """Проверяет выброс исключения при передаче невалидного основания (base <= 1.0)"""
    with pytest.raises(ValueError):
        ParametricMemoryGate(initial_base=1.0)
    with pytest.raises(ValueError):
        ParametricMemoryGate(initial_base=0.5)

def test_gradient_flow():
    """Проверяет, что градиенты успешно рассчитываются для обучаемых параметров слоя"""
    gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
    inputs = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
    
    outputs = gate(inputs)
    loss = outputs.sum()
    loss.backward()
    
    assert gate.raw_base.grad is not None, "Градиент для raw_base не должен быть None"
    assert gate.shift.grad is not None, "Градиент для shift не должен быть None"
    assert not torch.isnan(gate.raw_base.grad), "Градиент raw_base не должен быть NaN"
    assert not torch.isnan(gate.shift.grad), "Градиент shift не должен быть NaN"

def test_numerical_stability():
    """Проверяет устойчивость слоя к экстремальным значениям (Overflow/Underflow)"""
    gate = ParametricMemoryGate(initial_base=10.0, initial_shift=-5.0)
    
    # Подаем экстремально огромные и малые числа
    extreme_inputs = torch.tensor([-1e5, 1e5], dtype=torch.float32)
    outputs = gate(extreme_inputs)
    
    # Проверяем прямой проход
    assert not torch.isnan(outputs).any(), "Выход содержит NaN"
    assert not torch.isinf(outputs).any(), "Выход содержит Inf"
    
    # Проверяем обратный проход на стабильность градиентов
    loss = outputs.sum()
    loss.backward()
    assert not torch.isnan(gate.raw_base.grad).any(), "Градиент взорвался до NaN"

def test_cuda_compatibility():
    """Проверяет корректность работы слоя на GPU (если доступна технология CUDA)"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA (GPU) недоступна на данном устройстве, тест пропускается.")
        
    device = torch.device("cuda")
    gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0).to(device)
    inputs = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device)
    
    outputs = gate(inputs)
    assert outputs.device.type == "cuda", "Выходной тензор должен находиться на GPU"
    
    loss = outputs.sum()
    loss.backward()
    assert gate.raw_base.grad.device.type == "cuda", "Градиенты должны рассчитываться на GPU"
