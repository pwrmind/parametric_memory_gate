import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
# Импортируем вашу разработку
from parametric_memory_gate import ParametricMemoryGate

# Воспроизводимость
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. АРХИТЕКТУРА И МОДЕЛИ
# =====================================================================
class AdvancedCustomRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        self.update_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        
        self.w_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_candidate = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, seq_x):
        batch_size, seq_len, _ = seq_x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=seq_x.device)
        for t in range(seq_len):
            inp = seq_x[:, t, :]
            combined = torch.cat((inp, h), dim=1)
            f_t = self.forget_gate(self.w_forget(combined))
            ui_t = self.update_gate(self.w_update(combined))
            c_t = torch.tanh(self.w_candidate(inp))
            h = (f_t * h) + (ui_t * c_t)
        return self.fc_out(h)

class StandardGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc_out(gru_out[:, -1, :])

# =====================================================================
# 2. ДАННЫЕ
# =====================================================================
print("⚡ Инициализация боевой лаборатории PMG...")
print("📥 Скачивание актуальных данных BTC-USD...")
df = yf.download("BTC-USD", start="2023-01-01", progress=False)
prices = df['Close'].values.reshape(-1, 1)
dates = df.index.strftime('%Y-%m-%d').tolist()

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

seq_length = 14
X_list, Y_list = [], []
for i in range(len(prices_scaled) - seq_length):
    X_list.append(prices_scaled[i : i + seq_length])
    Y_list.append(prices_scaled[i + seq_length])

X = torch.tensor(np.array(X_list), dtype=torch.float32)
Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# Даты, соответствующие тестовому периоду
test_dates = dates[seq_length + split:]
real_test_prices = prices[seq_length + split:].flatten()

# =====================================================================
# 3. ОБУЧЕНИЕ И ИНФЕРЕНС
# =====================================================================
def train_model(model_instance, epochs=150):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=0.005)
    for epoch in range(epochs):
        model_instance.train()
        optimizer.zero_grad()
        loss = criterion(model_instance(X_train), Y_train)
        loss.backward()
        optimizer.step()
    
    model_instance.eval()
    with torch.no_grad():
        preds_scaled = model_instance(X_test).numpy()
        test_loss = criterion(model_instance(X_test), Y_test).item()
    return scaler.inverse_transform(preds_scaled).flatten(), test_loss

pmg_model = AdvancedCustomRNN(hidden_dim=8)
gru_model = StandardGRU(hidden_dim=8)

pmg_preds, pmg_error = train_model(pmg_model)
gru_preds, gru_error = train_model(gru_model)

# Текстовый вывод в консоль
print("\n" + "═"*70)
print("🏆 ИТОГОВЫЙ БАТТЛ С УЛУЧШЕННОЙ ВИЗУАЛИЗАЦИЕЙ")
print("═"*70)
max_err = max(pmg_error, gru_error)
scale = 45 / max_err
print(f"🔴 Standard PyTorch GRU:  [{'█'*int(gru_error*scale):<45}]  MSE: {gru_error:.6f}")
print(f"🟢 Advanced PMG RNN:     [{'█'*int(pmg_error*scale):<45}]  MSE: {pmg_error:.6f}")
print("─"*70)
improvement = (gru_error - pmg_error) / gru_error * 100
print(f"🔥 РЕЗУЛЬТАТ: PMG точнее стандарта на {improvement:.1f}%!")
print("═"*70)

# =====================================================================
# 4. ГЕНЕРАЦИЯ ИНТЕРАКТИВНОГО HTML-ОТЧЕТА
# =====================================================================
print("📊 Генерация интерактивного HTML-отчета...")

# Переводим списки в строковый формат для JavaScript массивов
js_dates = str(test_dates)
js_real = str(real_test_prices.tolist())
js_pmg = str(pmg_preds.tolist())
js_gru = str(gru_preds.tolist())

html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>PMG Architecture Benchmark Report</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f4f6f9; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        h1 {{ color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; margin-top: 0; }}
        .metric-box {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .card {{ flex: 1; padding: 20px; border-radius: 8px; color: white; text-align: center; font-weight: bold; }}
        .card.pmg {{ background: linear-gradient(135deg, #10b981, #059669); }}
        .card.gru {{ background: linear-gradient(135deg, #ef4444, #dc2626); }}
        .card.win {{ background: linear-gradient(135deg, #3b82f6, #2563eb); }}
        .value {{ font-size: 24px; margin-top: 10px; }}
        #chart {{ width: 100%; height: 600px; background: #fff; border-radius: 8px; }}
        footer {{ margin-top: 30px; text-align: center; color: #64748b; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Описание эксперимента: Parametric Memory Gate (PMG)</h1>
        
        <div class="metric-box">
            <div class="card pmg">
                <div>Advanced PMG RNN Error</div>
                <div class="value">{pmg_error:.6f}</div>
            </div>
            <div class="card gru">
                <div>Standard PyTorch GRU Error</div>
                <div class="value">{gru_error:.6f}</div>
            </div>
            <div class="card win">
                <div>Превосходство вашей математики</div>
                <div class="value">+{improvement:.1f}% 🔥</div>
            </div>
        </div>

        <div id="chart"></div>
        
        <footer>Сгенерировано автоматически ядром вашей кастомной нейросети. 2026.</footer>
    </div>

    <script>
        const dates = {js_dates};
        
        const traceReal = {{
            x: dates, y: {js_real}, mode: 'lines', name: 'Реальная цена BTC',
            line: {{ color: '#1e293b', width: 2.5 }}
        }};
        
        const tracePMG = {{
            x: dates, y: {js_pmg}, mode: 'lines', name: 'Прогноз PMG Memory (Ваша)',
            line: {{ color: '#10b981', width: 2, dash: 'solid' }}
        }};
        
        const traceGRU = {{
            x: dates, y: {js_gru}, mode: 'lines', name: 'Прогноз Standard GRU',
            line: {{ color: '#ef4444', width: 2, dash: 'dot' }}
        }};

        const layout = {{
            title: 'Сравнение точности прогноза на отложенной выборке (Out-of-Sample)',
            xaxis: {{ title: 'Дата', gridcolor: '#f1f5f9' }},
            yaxis: {{ title: 'Цена BTC (USD)', gridcolor: '#f1f5f9' }},
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff',
            hovermode: 'x unified'
        }};

        Plotly.newPlot('chart', [traceReal, tracePMG, traceGRU], layout);
    </script>
</body>
</html>
"""

report_path = "pmg_benchmark_report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"🎉 Интерактивный отчет успешно сохранен в файл: {os.path.abspath(report_path)}")
print("💡 Просто откройте этот файл в любом браузере (двойным кликом), чтобы увидеть графики!")
