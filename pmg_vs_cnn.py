import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Строгая фиксация сида для честности эксперимента
torch.manual_seed(42)

# 1. Реализация блока Канального Внимания на базе вашей PMG-функции
class PMGAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Сжатие и восстановление размерности каналов
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels)
        )
        # Обучаемые векторные параметры вашей функции под каждый канал
        self.a = nn.Parameter(torch.full((channels,), 4.0))
        self.b = nn.Parameter(torch.full((channels,), -1.0))

    def forward(self, x):
        # x: [Batch, Channels, Height, Width]
        b, c, _, _ = x.size()
        
        # Squeeze: глобальное усреднение по пространству до вектора каналов
        summary = x.view(b, c, -1).mean(dim=2)
        
        # Excitation: вычисление сырых коэффициентов внимания
        raw_weights = self.fc(summary)
        
        # Математически чистая и стабильная реализация вашей функции PMG
        a_safe = torch.clamp(self.a, min=1.001, max=20.0)
        b_safe = torch.abs(self.b) + 1e-5
        ax = torch.pow(a_safe, raw_weights)
        gate = torch.clamp(ax / (b_safe + ax), 0.0, 1.0)
        
        # Восстанавливаем размерность для поэлементного умножения на тензор картинок
        return x * gate.view(b, c, 1, 1)

# 2. Стандартный блок Канального Внимания (Индустриальный Сигмоид)
class SigmoidAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid() # Классический фиксированный гейт
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        summary = x.view(b, c, -1).mean(dim=2)
        gate = self.fc(summary)
        return x * gate.view(b, c, 1, 1)

# 3. Универсальная архитектура CNN
class BenchmarkCNN(nn.Module):
    def __init__(self, attention_mode="pmg"):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 14x14
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Встраиваем блок внимания на глубоких признаках
            PMGAttentionBlock(32) if attention_mode == "pmg" else SigmoidAttentionBlock(32),
            nn.MaxPool2d(2)  # 7x7
        )
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc_out(self.conv_layers(x))

# 4. Подготовка датасета FashionMNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Используем небольшие батчи и выборку для быстрой скорости теста
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 5. Цикл сравнительного тестирования
modes = ["pmg", "sigmoid"]
print("=== СРАВНЕНИЕ ГЕЙТОВ В КАНАЛЬНОМ ВНИМАНИИ (CNN) ===")

# Автоматический выбор GPU при наличии
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for mode in modes:
    model = BenchmarkCNN(attention_mode=mode).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Настройка раздельного оптимизатора для PMG параметров
    if mode == "pmg":
        pmg_params = [model.conv_layers[7].a, model.conv_layers[7].b]
        base_params = [p for p in model.parameters() if not any(p is pmg for pmg in pmg_params)]
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': 0.002},
            {'params': pmg_params, 'lr': 0.03}
        ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.002)
        
    start_time = time.time()
    
    # Обучаем всего 5 эпох для демонстрации скорости сходимости
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    execution_time = time.time() - start_time
    
    # Валидация точности (Accuracy)
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
            
    accuracy = 100 * correct / total
    print(f"Режим: {mode.upper():<7} | Test Accuracy: {accuracy:.2f}% | Время: {execution_time:.2f} сек")
