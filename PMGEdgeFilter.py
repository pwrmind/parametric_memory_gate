import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from parametric_memory_gate import ParametricMemoryGate

# Воспроизводимость
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. СТАБИЛИЗИРОВАННАЯ АРХИТЕКТУРА МОДЕЛИ (Many-to-Many)
# =====================================================================
class PMGEdgeFilter(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        self.update_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        
        self.w_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_candidate = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, seq_x):
        """Пакетное обучение: возвращает прогнозы для КАЖДОГО шага времени"""
        batch_size, seq_len, _ = seq_x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=seq_x.device)
        outputs = []
        
        for t in range(seq_len):
            inp = seq_x[:, t, :]
            combined = torch.cat((inp, h), dim=1)
            f_t = self.forget_gate(self.w_forget(combined))
            ui_t = self.update_gate(self.w_update(combined))
            c_t = torch.tanh(self.w_candidate(inp))
            h = (f_t * h) + (ui_t * c_t)
            
            # Сохраняем прогноз на каждом промежуточном шаге
            outputs.append(self.fc_out(h))
            
        # Возвращаем тензор формы [batch_size, seq_len, 1]
        return torch.stack(outputs, dim=1)

    def process_single_step(self, current_input, past_hidden):
        """Вычисление одного такта в реальном времени"""
        combined = torch.cat((current_input, past_hidden), dim=1)
        f_t = self.forget_gate(self.w_forget(combined))
        ui_t = self.update_gate(self.w_update(combined))
        c_t = torch.tanh(self.w_candidate(current_input))
        new_hidden = (f_t * past_hidden) + (ui_t * c_t)
        output = self.fc_out(new_hidden)
        return output, new_hidden

# =====================================================================
# 2. ОБУЧЕНИЕ НА ВСЕЙ ТРАЕКТОРИИ (Many-to-Many Loss)
# =====================================================================
print("🏋️ Run-time калибровка стабилизирующего ИИ-фильтра...")
num_samples = 500
seq_len = 30
true_height = 10.0

X_train = []
Y_train = [] # Теперь таргет имеет значение 10.0 на каждом шаге времени!
for _ in range(num_samples):
    baro = true_height + np.random.normal(0, 0.3, seq_len)
    imu = np.random.normal(0, 1.5, seq_len)
    X_train.append(np.stack([baro, imu], axis=1))
    Y_train.append([[true_height] for _ in range(seq_len)])

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)

filter_system = PMGEdgeFilter()
criterion = nn.MSELoss()
optimizer = optim.Adam(filter_system.parameters(), lr=0.005)

# Интенсивное обучение для выстраивания баланса гейтов
for epoch in range(150):
    filter_system.train()
    optimizer.zero_grad()
    loss = criterion(filter_system(X_train), Y_train)
    loss.backward()
    optimizer.step()

# Перевод в режим бесконечного инференса
filter_system.eval()
hidden_state = torch.zeros(1, 8)

# =====================================================================
# 3. БОЕВОЙ ПОТОК ДАННЫХ В РЕАЛЬНОМ ВРЕМЕНИ
# =====================================================================
print("\n📡 Запуск ИИ-фильтра PMG на Edge-устройстве...")
print("⏳ Ожидание потока телеметрии с датчиков (нажмите Ctrl+C для остановки)...\n")
print(f"{'Время':<10} | {'Сырой Барометр':<15} | {'Тряска IMU (Шум)':<17} | {'👉 Результат PMG':<18}")
print("─" * 70)

try:
    while True:
        baro_raw = true_height + np.random.normal(0, 0.3)
        imu_vibration = np.random.normal(0, 1.5)
        
        input_step = torch.tensor([[baro_raw, imu_vibration]], dtype=torch.float32)
        
        with torch.no_grad():
            clean_prediction, hidden_state = filter_system.process_single_step(input_step, hidden_state)
            
        current_time = time.strftime("%H:%M:%S")
        print(f"{current_time:<10} | {baro_raw:<15.2f} | {imu_vibration:<17.2f} | {clean_prediction.item():<18.4f}")
        
        time.sleep(0.5)
        
except KeyboardInterrupt:
    print("\n🛑 Работа ИИ-фильтра PMG успешно завершена.")
