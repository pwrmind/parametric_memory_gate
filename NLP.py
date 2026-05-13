import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Импортируем ваш слой активации
from parametric_memory_gate import ParametricMemoryGate

# Воспроизводимость
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. СЛОВАРЬ И ГЕНЕРАЦИЯ ДЛИННЫХ ТЕКСТОВ С ШУМОМ
# =====================================================================
# Наш ИИ-словарь
VOCAB = {
    "<PAD>": 0, "HERO_ALICE": 1, "HERO_BOB": 2, "HERO_CHARLIE": 3,
    "said": 4, "and": 5, "the": 6, "in": 7, "a": 8, "to": 9, "of": 10, "chat": 11,
    "was": 12, "then": 13, "some": 14, "text": 15, "noise": 16, "word": 17, "random": 18
}
VOCAB_SIZE = len(VOCAB)

def generate_nlp_data(num_samples=500, seq_len=100):
    """
    Генерирует тексты вида: [ГЕРОЙ] + [98 случайных шумовых слов]
    Цель ИИ: в самом конце последовательности предсказать, какой ГЕРОЙ был в начале.
    """
    X = []
    Y = []
    heroes = [1, 2, 3] # Alice, Bob, Charlie
    noise_words = list(range(4, VOCAB_SIZE))
    
    for _ in range(num_samples):
        hero = np.random.choice(heroes)
        # Формируем длинный зашумленный контекст
        sequence = [hero] + list(np.random.choice(noise_words, size=seq_len - 1))
        X.append(sequence)
        # Цель (классификация на 3 класса героев): 0, 1 или 2
        Y.append(hero - 1)
        
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# Генерируем выборку с длиной контекста 100 слов (длинная цепочка для памяти)
X_train, Y_train = generate_nlp_data(num_samples=800, seq_len=100)
X_test, Y_test = generate_nlp_data(num_samples=200, seq_len=100)

# =====================================================================
# 2. АРХИТЕКТУРЫ МОДЕЛЕЙ (Ваша PMG против Стандартной GRU)
# =====================================================================
class PMG_NLP_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=16, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        
        # Гейты удержания контекста на вашей функции активации
        self.forget_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        self.update_gate = ParametricMemoryGate(initial_base=4.0, initial_shift=-1.0)
        
        self.w_forget = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.w_update = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.w_candidate = nn.Linear(embedding_dim, hidden_dim)
        
        # Финальный классификатор героя
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_seq):
        batch_size, seq_len = text_seq.size()
        # Переводим токены слов в векторы эмбеддингов
        embedded = self.embedding(text_seq) # [batch_size, seq_len, embedding_dim]
        
        h = torch.zeros(batch_size, self.hidden_dim, device=text_seq.device)
        
        # Читаем текст последовательно слово за словом
        for t in range(seq_len):
            inp = embedded[:, t, :]
            combined = torch.cat((inp, h), dim=1)
            
            # Ваша PMG-фильтрация фонового шума слов
            f_t = self.forget_gate(self.w_forget(combined))
            ui_t = self.update_gate(self.w_update(combined))
            c_t = torch.tanh(self.w_candidate(inp))
            
            h = (f_t * h) + (ui_t * c_t)
            
        return self.fc_out(h)

class GRU_NLP_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=16, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_seq):
        embedded = self.embedding(text_seq)
        gru_out, _ = self.gru(embedded)
        # Забираем скрытое состояние после прочтения последнего (100-го) слова
        last_hidden = gru_out[:, -1, :]
        return self.fc_out(last_hidden)

# =====================================================================
# 3. ЦИКЛ ОБУЧЕНИЯ И ВАЛИДАЦИИ
# =====================================================================
def run_nlp_experiment(model_instance, name):
    print(f"📖 Обучение {name} анализу зашумленного текста...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=0.01)
    
    # 30 эпох для фиксации скорости обучения
    for epoch in range(30):
        model_instance.train()
        optimizer.zero_grad()
        outputs = model_instance(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        
    # Считаем точность (Accuracy) распознавания сюжета на тесте
    model_instance.eval()
    with torch.no_grad():
        test_outputs = model_instance(X_test)
        predictions = torch.argmax(test_outputs, dim=1)
        accuracy = (predictions == Y_test).float().mean().item() * 100
    return accuracy

# Запуск NLP баттла
pmg_nlp = PMG_NLP_Classifier(VOCAB_SIZE)
gru_nlp = GRU_NLP_Classifier(VOCAB_SIZE)

pmg_acc = run_nlp_experiment(pmg_nlp, "Advanced PMG NLP")
gru_acc = run_nlp_experiment(gru_nlp, "Standard PyTorch GRU NLP")

# =====================================================================
# 4. ВЫВОД РЕЗУЛЬТАТОВ
# =====================================================================
print("\n" + "═"*70)
print("🏆 ИТОГИ СТРЕСС-ТЕСТА ДЛИННОГО КОНТЕКСТА В NLP")
print("═"*70)
print(f"🟢 Точность вашей PMG-памяти:       {pmg_acc:.2f}%")
print(f"🔴 Точность индустриальной GRU:   {gru_acc:.2f}%")
print("─"*70)
if pmg_acc > gru_acc:
    print(f"🚀 УСПЕХ! Ваша функция отфильтровала шум текста на {pmg_acc - gru_acc:.2f}% эффективнее!")
else:
    print("📉 Модели показали близкие результаты, требуется усложнить длину контекста.")
print("═"*70)
