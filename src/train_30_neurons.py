import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Фиксация для воспроизводимости
np.random.seed(42)
torch.manual_seed(42)

# ================= 1. ПОДГОТОВКА ДАННЫХ =================
X = np.linspace(-5, 5, 200).reshape(-1, 1)
y = np.sin(5 * X) * np.exp(X / 2)  # ВАША ФУНКЦИЯ

# Нормализация Min-Max
X_min, X_max = -5.0, 5.0
y_min, y_max = y.min(), y.max()
X_norm = (X - X_min) / (X_max - X_min)
y_norm = (y - y_min) / (y_max - y_min)

# Разбиение 60/30/10
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_norm, y_norm, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.333, random_state=42)

# Тензоры
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)
X_all_t = torch.FloatTensor(X_norm)

# ================= 2. СЕТЬ =================
class Net(nn.Module):
    def __init__(self, hidden, init_range=5.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.uniform_(self.fc1.weight, -init_range, init_range)
        nn.init.uniform_(self.fc1.bias, -init_range, init_range)

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h), h

model = Net(hidden=30, init_range=5.0)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)  # Stochastic GD

epochs = 40000
train_losses, val_losses, test_losses = [], [], []
w_hidden_hist, w_out_hist = [], []

# ==========================================
# СОСТОЯНИЕ ДО ОБУЧЕНИЯ (сохраняем явно)
# ==========================================
model.eval()
with torch.no_grad():
    y_before_train, h_before_train = model(X_all_t)
    
# Сохраняем веса "До", чтобы позже можно было восстановить состояние для графика
weights_before = {k: v.clone() for k, v in model.state_dict().items()}

print("Обучение сети с 30 нейронами (Stochastic SGD)...")
for epoch in range(epochs):
    model.train()
    idx = np.random.randint(0, len(X_train_t))
    optimizer.zero_grad()
    out, _ = model(X_train_t[idx:idx+1])
    loss = criterion(out, y_train_t[idx:idx+1])
    
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"⚠️ Обнаружен NaN на эпохе {epoch}. Останов.")
        break
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    
    if epoch % 50 == 0:
        w_hidden_hist.append(model.fc1.weight.detach().numpy().flatten().copy())
        w_out_hist.append(model.fc2.weight.detach().numpy().flatten().copy())
    
    model.eval()
    with torch.no_grad():
        tl = criterion(model(X_train_t)[0], y_train_t).item()
        vl = criterion(model(X_val_t)[0], y_val_t).item()
        tel = criterion(model(X_test_t)[0], y_test_t).item()
    train_losses.append(tl)
    val_losses.append(vl)
    test_losses.append(tel)
    
    if (epoch + 1) % 1000 == 0:
        print(f"Эпоха {epoch+1}/{epochs} | Train: {tl:.6f} | Val: {vl:.6f}")

# Состояние ПОСЛЕ обучения
model.eval()
with torch.no_grad():
    y_pred_after, hidden_after = model(X_all_t)
    y_pred_train, _ = model(X_train_t)
    y_pred_val, _ = model(X_val_t)
    y_pred_test, _ = model(X_test_t)

# Восстанавливаем состояние "До" для гарантированной отрисовки
model.load_state_dict(weights_before)
model.eval()
with torch.no_grad():
    y_pred_before, hidden_before = model(X_all_t)

# ================= 3. МЕТРИКИ =================
def get_metrics(y_true, y_pred):
    err = y_true - y_pred
    return np.max(np.abs(err)), np.std(err), np.sqrt(np.mean(err**2))

t_max, t_std, t_rmse = get_metrics(y_train, y_pred_train.numpy())
v_max, v_std, v_rmse = get_metrics(y_val, y_pred_val.numpy())
te_max, te_std, te_rmse = get_metrics(y_test, y_pred_test.numpy())

print("\n=== ЭТАП 2: 30 НЕЙРОНОВ ===")
print(f"{'Выборка':<15} | {'Макс. абс. ошибка':<20} | {'С.к.о.':<15} | {'RMSE':<15}")
print("-" * 75)
print(f"{'Обучающая':<15} | {t_max:<20.4f} | {t_std:<15.4f} | {t_rmse:<15.4f}")
print(f"{'Валидационная':<15} | {v_max:<20.4f} | {v_std:<15.4f} | {v_rmse:<15.4f}")
print(f"{'Тестовая':<15} | {te_max:<20.4f} | {te_std:<15.4f} | {te_rmse:<15.4f}")

# ================= 4. ГРАФИКИ =================
plt.figure(figsize=(14, 16))

# 1. y(x): Истинная, До, После (ГАРАНТИРОВАННО ВИДНО)
plt.subplot(3, 2, 1)
plt.plot(X_norm, y_norm, 'b-', linewidth=2.5, label='Истинная функция', zorder=3)
plt.plot(X_norm, y_pred_before.numpy(), 'm--', linewidth=3.0, label='До обучения (30н)', zorder=4)
plt.plot(X_norm, y_pred_after.numpy(), 'r--', linewidth=2.0, label='После обучения (30н)', zorder=2)
plt.title('Зависимость выхода y(x) сети от входа')
plt.xlabel('Нормированный вход X')
plt.ylabel('Нормированный выход Y')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.4)

# 2. Веса скрытого слоя
plt.subplot(3, 2, 2)
wh = np.array(w_hidden_hist)
for i in range(wh.shape[1]):
    plt.plot(wh[:, i], linewidth=0.8, alpha=0.7)
plt.title('Зависимость синаптических коэффициентов w(τ) \nскрытого слоя от времени')
plt.grid(True, alpha=0.4)

# 3. Скрытые нейроны ДО обучения
plt.subplot(3, 2, 3)
for i in range(30):
    plt.plot(X_norm, hidden_before.numpy()[:, i], linewidth=1.0, alpha=0.6)
plt.title('Зависимость выходов yk(x) скрытого слоя ДО обучения')
plt.grid(True, alpha=0.4)

# 4. Скрытые нейроны ПОСЛЕ обучения
plt.subplot(3, 2, 4)
for i in range(30):
    plt.plot(X_norm, hidden_after.numpy()[:, i], linewidth=1.0, alpha=0.6)
plt.title('Зависимость выходов yk(x) скрытого слоя ПОСЛЕ обучения')
plt.grid(True, alpha=0.4)

# 5. Ошибка от времени
plt.subplot(3, 2, 5)
plt.plot(train_losses, 'b-', linewidth=1.5, label='Train')
plt.plot(val_losses, 'g-', linewidth=1.5, label='Validation')
plt.plot(test_losses, 'r-', linewidth=1.5, label='Test')
plt.title('Зависимость ошибки сети E(τ) от времени обучения')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.4)

# 6. Веса выходного нейрона
plt.subplot(3, 2, 6)
wo = np.array(w_out_hist)
for i in range(wo.shape[1]):
    plt.plot(wo[:, i], linewidth=0.8, alpha=0.7)
plt.title('Зависимость синаптических коэффициентов w(τ) \nвыходного нейрона от времени')
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('plots_stage2_30neurons_FINAL.png', dpi=300, bbox_inches='tight')
print("\n✅ Все 6 графиков сохранены в plots_stage2_30neurons_FINAL.png")
plt.show()