import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from typing import Tuple
import time

# =====================================================================
# 1. ВАШ ParametricMemoryGate (встроен прямо в скрипт)
# =====================================================================
class ParametricMemoryGate(nn.Module):
    def __init__(self, initial_base: float = 4.0, initial_shift: float = -1.0):
        super().__init__()
        if initial_base <= 1.0:
            raise ValueError("initial_base must be strictly greater than 1.0")
        raw_base_init = np.log(initial_base - 1.0)
        self.raw_base = nn.Parameter(torch.tensor([raw_base_init], dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor([initial_shift], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = 1.0 + torch.exp(self.raw_base)
        power = torch.clamp(x + self.shift, -20.0, 20.0)
        gate = (base ** power) / (1.0 + (base ** power))
        eps = 1e-7
        gate = torch.clamp(gate, eps, 1.0 - eps)
        return gate

    def get_parameters(self) -> Tuple[float, float]:
        with torch.no_grad():
            actual_base = 1.0 + torch.exp(self.raw_base).item()
            actual_shift = self.shift.item()
            return actual_base, actual_shift

# =====================================================================
# 2. МОДЕЛИ ДЛЯ СРАВНЕНИЯ
# =====================================================================
class AdvancedCustomRNN(nn.Module):
    """Ваша PMG-архитектура: два независимых параметрических гейта"""
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

class StandardLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc_out(lstm_out[:, -1, :])

class SimpleRNN(nn.Module):
    """Классическая RNN с Tanh (без гейтов)"""
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        return self.fc_out(rnn_out[:, -1, :])

# =====================================================================
# 3. ЗАГРУЗКА ДАННЫХ
# =====================================================================
def load_data(ticker="BTC-USD", seq_length=14, test_ratio=0.2):
    print(f"📥 Скачивание данных {ticker}...")
    df = yf.download(ticker, start="2023-01-01", progress=False)
    if df.empty:
        raise ValueError("Не удалось загрузить данные. Проверьте тикер и интернет-соединение.")
    prices = df['Close'].values.reshape(-1, 1).astype(np.float32)
    dates = df.index.strftime('%Y-%m-%d').tolist()

    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    X_list, Y_list = [], []
    for i in range(len(prices_scaled) - seq_length):
        X_list.append(prices_scaled[i : i + seq_length])
        Y_list.append(prices_scaled[i + seq_length])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    test_dates = dates[seq_length + split:]
    real_test_prices = prices[seq_length + split:].flatten()
    return X_train, X_test, Y_train, Y_test, scaler, test_dates, real_test_prices

# =====================================================================
# 4. ОБУЧЕНИЕ
# =====================================================================
def train_model(model, X_train, Y_train, X_test, Y_test, epochs=150, lr=0.005):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), Y_train)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test).numpy()
        test_loss = criterion(model(X_test), Y_test).item()
    return preds_scaled, test_loss, train_time

# =====================================================================
# 5. ГЛАВНЫЙ БЛОК СРАВНЕНИЯ
# =====================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Параметры эксперимента
    TICKER = "BTC-USD"          # Можно заменить на "AAPL", "^GSPC" и т.д.
    SEQ_LEN = 14
    HIDDEN_DIM = 8
    EPOCHS = 150
    LR = 0.005

    # Загрузка данных
    X_train, X_test, Y_train, Y_test, scaler, test_dates, real_prices = load_data(
        ticker=TICKER, seq_length=SEQ_LEN
    )

    # Создание моделей
    models = {
        "🟢 Advanced PMG RNN": AdvancedCustomRNN(input_dim=1, hidden_dim=HIDDEN_DIM),
        "🔴 Standard GRU":     StandardGRU(input_dim=1, hidden_dim=HIDDEN_DIM),
        "🔵 Standard LSTM":    StandardLSTM(input_dim=1, hidden_dim=HIDDEN_DIM),
        "⚪ Simple RNN (tanh)": SimpleRNN(input_dim=1, hidden_dim=HIDDEN_DIM)
    }

    results = {}
    print("\n" + "="*70)
    print("🚀 СТАРТ БАТТЛА: PMG vs INDUSTRY STANDARDS")
    print("="*70)

    for name, model in models.items():
        print(f"\n⏳ Обучение {name}...")
        preds_scaled, mse, ttime = train_model(model, X_train, Y_train, X_test, Y_test,
                                                epochs=EPOCHS, lr=LR)
        # Обратное масштабирование предсказаний
        preds_real = scaler.inverse_transform(preds_scaled).flatten()
        results[name] = {
            "predictions": preds_real,
            "mse": mse,
            "time": ttime
        }
        print(f"   ✓ Завершено за {ttime:.1f} сек, Test MSE = {mse:.6f}")

    # Вывод таблицы сравнения в консоль
    print("\n" + "═"*70)
    print("🏆 ИТОГОВЫЙ РЕЙТИНГ")
    print("═"*70)
    # Сортируем по MSE (меньше = лучше)
    sorted_models = sorted(results.items(), key=lambda x: x[1]["mse"])
    best_mse = sorted_models[0][1]["mse"]
    for rank, (name, res) in enumerate(sorted_models, 1):
        mse = res["mse"]
        improvement = (best_mse - mse) / best_mse * 100 if rank > 1 else 0.0
        bar_len = int(30 * mse / best_mse)  # масштаб для визуализации
        bar = '█' * max(1, bar_len)
        improv_str = f"  [лучше лидера на {improvement:.1f}%]" if rank > 1 else "  [АБСОЛЮТНЫЙ ПОБЕДИТЕЛЬ]"
        print(f"{rank}. {name:<25} MSE={mse:.6f} {bar}{improv_str}")
    print("─"*70)
    print(f"⚡ Лидер — {sorted_models[0][0]} с ошибкой {best_mse:.6f}")
    print("═"*70)

    # =================================================================
    # 6. ГЕНЕРАЦИЯ ИНТЕРАКТИВНОГО HTML-ОТЧЕТА (Plotly)
    # =================================================================
    print("\n📊 Генерация интерактивного отчета...")
    traces_js = []
    colors = {
        "🟢 Advanced PMG RNN": "#10b981",
        "🔴 Standard GRU": "#ef4444",
        "🔵 Standard LSTM": "#3b82f6",
        "⚪ Simple RNN (tanh)": "#6b7280"
    }
    for name, res in results.items():
        traces_js.append({
            "x": test_dates,
            "y": res["predictions"].tolist(),
            "mode": "lines",
            "name": name,
            "line": {"color": colors.get(name, "#000000"), "width": 2}
        })
    # Добавим реальные цены
    traces_js.append({
        "x": test_dates,
        "y": real_prices.tolist(),
        "mode": "lines",
        "name": "Реальная цена",
        "line": {"color": "#1e293b", "width": 2.5}
    })

    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>PMG Benchmark — {TICKER}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 30px; background: #f8fafc; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 30px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
        h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
        .card-row {{ display: flex; gap: 20px; margin: 25px 0; }}
        .card {{ flex: 1; padding: 18px; border-radius: 12px; color: white; font-weight: bold; }}
        .card.pmg {{ background: linear-gradient(135deg, #10b981, #059669); }}
        .card.gru {{ background: linear-gradient(135deg, #ef4444, #dc2626); }}
        .card.lstm {{ background: linear-gradient(135deg, #3b82f6, #2563eb); }}
        .card.rnn {{ background: linear-gradient(135deg, #6b7280, #4b5563); }}
        .value {{ font-size: 24px; margin-top: 8px; }}
        #chart {{ width: 100%; height: 550px; }}
        footer {{ margin-top: 25px; text-align: center; color: #64748b; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Parametric Memory Gate vs Индустриальные стандарты</h1>
        <p>Данные: {TICKER}, горизонт {SEQ_LEN} дней, скрытый размер {HIDDEN_DIM}</p>
        <div class="card-row">
            {''.join(f'<div class="card {cls}"><div>{name}</div><div class="value">MSE: {results[name]["mse"]:.6f}</div></div>' 
                        for name, cls in zip(results.keys(), ['pmg','gru','lstm','rnn']))}
        </div>
        <div id="chart"></div>
        <footer>Сгенерировано ядром кастомной PMG-нейросети</footer>
    </div>
    <script>
        var data = {traces_js};
        var layout = {{
            title: 'Прогноз vs Реальная цена (тестовая выборка)',
            xaxis: {{ title: 'Дата' }},
            yaxis: {{ title: 'Цена ({TICKER})' }},
            hovermode: 'x unified',
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        }};
        Plotly.newPlot('chart', data, layout);
    </script>
</body>
</html>"""

    report_path = "pmg_benchmark_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Отчет сохранен: {os.path.abspath(report_path)}")
    print("   Откройте его в браузере для детальной визуализации.")

if __name__ == "__main__":
    main()