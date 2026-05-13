import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Импортируем ваш кастомный слой активации
from parametric_memory_gate import ParametricMemoryGate

# Воспроизводимость эксперимента
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. СИНТЕЗАТОР БИОСИГНАЛОВ ЭКГ С АНОМАЛИЯМИ И ШУМОМ
# =====================================================================
def generate_ecg_signal(num_samples=400, seq_len=30):
    """
    Генерирует последовательности ЭКГ (пульса).
    Обычный цикл: плавные волны (P-Q-R-S-T комплекс).
    Шум: случайные резкие микро-всплески (движение тела).
    Аномалия: критический пропуск ритма или аномальный гигантский зубец (аритмия).
    """
    X = []
    Y = [] # 0 - норма/шум, 1 - критическая аномалия/аритмия
    
    t = np.linspace(0, 2 * np.pi, seq_len)
    
    for _ in range(num_samples):
        # Базовый чистый синусоидальный сердечный ритм
        ecg = np.sin(t * 2) * 0.5 
        
        # Подмешиваем постоянный бытовой шум (движение пациента)
        noise = np.random.normal(0, 0.15, seq_len)
        ecg += noise
        
        is_anomaly = np.random.choice([0, 1], p=[0.7, 0.3])
        
        if is_anomaly:
            # Моделируем аритмию: резкий патологический всплеск в центре цикла
            anomaly_peak = np.zeros(seq_len)
            anomaly_peak[seq_len // 2] = np.random.choice([1.8, -1.8])
            ecg += anomaly_peak
            Y.append(1)
        else:
            # Моделируем просто сильный артефакт движения (одиночный шум, не аномалия)
            if np.random.rand() > 0.5:
                ecg[np.random.randint(0, seq_len)] += 0.4
            Y.append(0)
            
        X.append(ecg)
        
    # Форматируем под тензоры PyTorch: [batch_size, seq_len, input_dim]
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)
    return X_tensor, Y_tensor

# Генерируем тренировочный и тестовый кардио-потоки
X_train, Y_train = generate_ecg_signal(num_samples=600, seq_len=30)
X_test, Y_test = generate_ecg_signal(num_samples=200, seq_len=30)

# =====================================================================
# 2. АРХИТЕКТУРЫ ДЕТЕКТОРОВ (Ваша PMG против PyTorch GRU)
# =====================================================================
class PMG_MedicalMonitor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Адаптивные гейты под уникальный ритм пациента
        self.forget_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        self.update_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        
        self.w_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_candidate = nn.Linear(input_dim, hidden_dim)
        
        # Выходной слой детекции аномалии (выдает вероятность сбоя)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid_out = nn.Sigmoid()

    def forward(self, seq_x):
        batch_size, seq_len, _ = seq_x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=seq_x.device)
        
        for t in range(seq_len):
            inp = seq_x[:, t, :]
            combined = torch.cat((inp, h), dim=1)
            
            # Ваша фильтрация артефактов
            f_t = self.forget_gate(self.w_forget(combined))
            ui_t = self.update_gate(self.w_update(combined))
            c_t = torch.tanh(self.w_candidate(inp))
            
            h = (f_t * h) + (ui_t * c_t)
            
        return self.sigmoid_out(self.fc_out(h))

class GRU_MedicalMonitor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=6):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid_out = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.sigmoid_out(self.fc_out(gru_out[:, -1, :]))

# =====================================================================
# 3. СОРЕВНОВАТЕЛЬНЫЙ ТРЕНИНГ
# =====================================================================
def train_monitor(model_instance, name):
    print(f"🩺 Настройка монитора {name} под кардиоритм...")
    criterion = nn.BCELoss() # Лосс бинарной классификации (аномалия/норма)
    optimizer = optim.Adam(model_instance.parameters(), lr=0.01)
    
    for epoch in range(100):
        model_instance.train()
        optimizer.zero_grad()
        loss = criterion(model_instance(X_train), Y_train)
        loss.backward()
        optimizer.step()
        
    # Тестирование точности выявления скрытых патологий
    model_instance.eval()
    with torch.no_grad():
        preds = model_instance(X_test)
        # Если вероятность > 0.5, генерируем тревогу
        predicted_classes = (preds > 0.5).float()
        accuracy = (predicted_classes == Y_test).float().mean().item() * 100
    return accuracy

# Запуск медицинского баттла
pmg_monitor = PMG_MedicalMonitor()
gru_monitor = GRU_MedicalMonitor()

pmg_accuracy = train_monitor(pmg_monitor, "Advanced PMG (Ваш)")
gru_accuracy = train_monitor(gru_monitor, "Standard PyTorch GRU")

# =====================================================================
# 4. ДИАГНОСТИЧЕСКИЙ ОТЧЕТ
# =====================================================================
print("\n" + "═"*75)
print("🏆 ИТОГИ СТРЕСС-ТЕСТА: ДЕТЕКЦИЯ КАРДИО-АНОМАЛИЙ НА ФОНЕ ШУМА")
print("═"*75)
print(f"🟢 Точность вашей PMG-памяти:              {pmg_accuracy:.2f}%")
print(f"🔴 Точность медицинского стандарта GRU:    {gru_accuracy:.2f}%")
print("─"*75)

# Анализируем изученную индивидуальную геометрию
f_base, f_shift = pmg_monitor.forget_gate.get_parameters()
u_base, u_shift = pmg_monitor.update_gate.get_parameters()

print(f"⚙️ Индивидуальные параметры настройки PMG под пациента:")
print(f"   ↳ Фильтр удержания ритма:  Base = {f_base:.3f} | Shift = {f_shift:.3f}")
print(f"   ↳ Фильтр захвата аномалий: Base = {u_base:.3f} | Shift = {u_shift:.3f}")
print("═"*75)
