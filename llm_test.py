import torch
import torch.nn as nn
import time

# 1. Реализация параметрического гейта для слоев общего назначения
class PMGGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Индивидуальные параметры для каждой фичи/канала
        self.a = nn.Parameter(torch.full((dim,), 4.0))
        self.b = nn.Parameter(torch.full((dim,), -1.0))

    def forward(self, x):
        a_safe = torch.clamp(self.a, min=1.001)
        b_safe = torch.abs(self.b) + 1e-5
        ax = torch.exp(x * torch.log(a_safe))
        return torch.clamp(ax / (b_safe + ax), 0.0, 1.0)

# 2. Реализация Gated MLP (Аналог блока трансформера)
class GatedMLP(nn.Module):
    def __init__(self, dim, mode="pmg"):
        super().__init__()
        self.mode = mode
        self.w_gate = nn.Linear(dim, dim * 2)
        self.w_up = nn.Linear(dim, dim * 2)
        self.w_down = nn.Linear(dim * 2, dim)
        
        if mode == "pmg":
            self.gate_act = PMGGate(dim * 2)
        elif mode == "gelu":
            self.gate_act = nn.GELU()
        elif mode == "silu":
            self.gate_act = nn.SiLU()
        elif mode == "sigmoid":
            self.gate_act = torch.sigmoid

    def forward(self, x):
        # Логика GLU: ворота зависят от w_gate, а контент от w_up
        gate_space = self.w_gate(x)
        content_space = self.w_up(x)
        
        # Применяем выбранный гейт
        activated_gate = self.gate_act(gate_space)
        
        # Поэлементное умножение (Gating)
        hidden = content_space * activated_gate
        return self.w_down(hidden)

# 3. Синтетический тест на сложную нелинейную зависимость
torch.manual_seed(42)
X = torch.randn(2000, 64)  # 2000 объектов, 64 признака
# Целевая функция со сложными взаимодействиями
Y = torch.sin(X[:, :32]).sum(dim=1, keepdim=True) * torch.cos(X[:, 32:]).sum(dim=1, keepdim=True)

# Разделение на train/test
X_train, X_test = X[:1600], X[1600:]
y_train, y_test = Y[:1600], Y[1600:]

modes = ["pmg", "silu", "gelu", "sigmoid"]
print("=== СРАВНЕНИЕ GLU БЛОКОВ В СОВРЕМЕННЫХ СЕТЯХ ===")

for mode in modes:
    model = nn.Sequential(
        GatedMLP(dim=64, mode=mode),
        nn.Linear(64, 1)
    )
    criterion = nn.MSELoss()
    
    if mode == "pmg":
        pmg_params = [model[0].gate_act.a, model[0].gate_act.b]
        base_params = [p for p in model.parameters() if not any(p is pmg for pmg in pmg_params)]
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': 0.01},
            {'params': pmg_params, 'lr': 0.05}
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        
    start = time.time()
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        
    # Валидация
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test), y_test).item()
        
    print(f"Режим: {mode.upper():<7} | Test MSE: {test_loss:.5f} | Время: {time.time()-start:.2f} сек")
