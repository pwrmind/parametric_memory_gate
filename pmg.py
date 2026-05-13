import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Фиксация случайных чисел для воспроизводимости
torch.manual_seed(42)

# =====================================================================
# 1. ЭТАЛОННАЯ РЕАЛИЗАЦИЯ PARAMETRIC MEMORY GATE (PMG)
# =====================================================================
class ParametricMemoryGate(nn.Module):
    """
    Производственная реализация Parametric Memory Gate (PMG).
    Вычисляет функцию: f(x) = a^x / (b_safe + a^x)
    
    Используется в механизмах взвешивания (Gating), внимания (Attention) 
    и рекуррентных структурах памяти (RNN Gates).
    """
    def __init__(self, dim: int = 1, initial_a: float = 4.0, initial_b: float = -1.0):
        super().__init__()
        # dim = 1: один профиль на слой. dim > 1: независимый профиль на каждый канал/нейрон.
        self.a = nn.Parameter(torch.full((dim,), float(initial_a), dtype=torch.float32))
        self.b = nn.Parameter(torch.full((dim,), float(initial_b), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Стабилизация основания степени для исключения переполнения float32
        a_safe = torch.clamp(self.a, min=1.001, max=20.0)
        # Стабилизация смещения для исключения деления на 0
        b_safe = torch.abs(self.b) + 1e-5
        
        # Вычисление степени с полной поддержкой отрицательных входов (x < 0)
        ax = torch.pow(a_safe, x)
        
        # Финальный расчет гейта со строгим ограничением диапазона ворот (0, 1)
        out = ax / (b_safe + ax)
        return torch.clamp(out, min=0.0, max=1.0)

    def extra_repr(self) -> str:
        return f"a_mean={self.a.mean().item():.4f}, b_mean={self.b.mean().item():.4f}"


# =====================================================================
# 2. ИНДУСТРИАЛЬНЫЕ ШАБЛОНЫ ИНТЕГРАЦИИ PMG
# =====================================================================

# --- Вариант А: Интеграция в сбалансированную рекуррентную ячейку (RNN) ---
class PMGRNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget_gate = ParametricMemoryGate(dim=hidden_dim)
        self.update_gate = ParametricMemoryGate(dim=hidden_dim)
        
        self.w_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_candidate = nn.Linear(input_dim, hidden_dim)

    def forward(self, seq_x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((seq_x, h), dim=1)
        f_t = self.forget_gate(self.w_forget(combined))
        ui_t = self.update_gate(self.w_update(combined))
        c_t = torch.tanh(self.w_candidate(seq_x))
        return (f_t * h) + (ui_t * c_t)


# --- Вариант Б: Интеграция в GLU-блок современных LLM/Трансформеров ---
class PMG_GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, dim * 2)
        self.w_up = nn.Linear(dim, dim * 2)
        self.w_down = nn.Linear(dim * 2, dim)
        self.pmg = ParametricMemoryGate(dim=dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Адаптивная замена стандартных SiLU / Swish гейтов
        return self.w_down(self.w_up(x) * self.pmg(self.w_gate(x)))


# --- Вариант В: Интеграция в Канальное Внимание Сверточных Сетей (CNN) ---
class PMGChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels)
        )
        self.pmg = ParametricMemoryGate(dim=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        summary = x.view(b, c, -1).mean(dim=2) # Global Average Pooling
        raw_weights = self.fc(summary)
        gate = self.pmg(raw_weights)
        return x * gate.view(b, c, 1, 1)


# =====================================================================
# 3. КОМПЛЕКСНАЯ НЕЙРОСЕТЬ ДЛЯ ДЕМОНСТРАЦИОННОГО ТЕСТА
# =====================================================================
class ProductionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Встраиваем эталонный блок внимания PMG
            PMGChannelAttention(32),
            nn.MaxPool2d(2)
        )
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc_out(self.conv_layers(x))


# =====================================================================
# 4. ЗАПУСК И СКРИПТ ОБУЧЕНИЯ
# =====================================================================
if __name__ == "__main__":
    print("=== Инициализация эталонного пайплайна PMG ===")
    
    # Подготовка данных (FashionMNIST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    
    model = ProductionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # ПРАВИЛО ОПТИМИЗАЦИИ PMG: Разделение параметров и повышение LR для гейтов
    # Извлекаем параметры 'a' и 'b' из всех внутренних модулей PMG
    pmg_params = [p for n, p in model.named_parameters() if '.pmg.' in n]
    base_params = [p for n, p in model.named_parameters() if '.pmg.' not in n]
    
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 0.002, 'weight_decay': 1e-4},
        {'params': pmg_params, 'lr': 0.03} # Повышенный шаг обучения
    ])
    
    print("\nСтарт демонстрационного обучения (3 эпохи)...")
    model.train()
    start_time = time.time()
    
    for epoch in range(3):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Эпоха [{epoch+1}/3] | Ср. Loss обучения: {running_loss/len(train_loader):.4f}")
        
    print(f"Обучение завершено за: {time.time() - start_time:.2f} сек")
    
    # Оценка качества модели
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"\nИтоговая точность на тесте (Test Accuracy): {100 * correct / total:.2f}%")
    
    # Демонстрация выученных средних значений гейта
    # Извлекаем модуль внимания из архитектуры напрямую
    trained_pmg = model.conv_layers[7].pmg
    print(f"Выученные средние параметры PMG слоя внимания -> a: {trained_pmg.a.mean().item():.3f}, b: {trained_pmg.b.mean().item():.3f}")
