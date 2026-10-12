import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error

# 1. Данные
np.random.seed(42)
torch.manual_seed(42)
X = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
y = np.sin(3 * X)
X_norm = (X - X.min()) / (X.max() - X.min())
y_norm = (y - y.min()) / (y.max() - y.min())

X_train_val, X_test, y_train_val, y_test = train_test_split(X_norm, y_norm, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.333, random_state=42)

X_train_t, y_train_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
X_val_t, y_val_t = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
X_test_t, y_test_t = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
X_all_t = torch.FloatTensor(X_norm)

# 2. Сеть (СЛАБАЯ)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 5) # Всего 5 нейронов!
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(5, 1)
        # Стандартная слабая инициализация от -1 до 1
        nn.init.uniform_(self.fc1.weight, -1.0, 1.0)
        nn.init.uniform_(self.fc1.bias, -1.0, 1.0)

    def forward(self, x):
        hidden = self.act(self.fc1(x))
        out = self.fc2(hidden)
        return out, hidden

model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Сохраняем состояние ДО обучения
model.eval()
with torch.no_grad():
    y_pred_before, hidden_before = model(X_all_t)

# 3. Обучение
epochs = 1000
train_losses, val_losses, test_losses = [], [], []
weights_hidden, weights_out = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred, _ = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        weights_hidden.append(model.fc1.weight.detach().numpy().flatten().copy())
        weights_out.append(model.fc2.weight.detach().numpy().flatten().copy())

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t)[0], y_val_t)
        test_loss = criterion(model(X_test_t)[0], y_test_t)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    test_losses.append(test_loss.item())

# Сохраняем состояние ПОСЛЕ обучения
model.eval()
with torch.no_grad():
    y_pred_after, hidden_after = model(X_all_t)
    y_pred_train, _ = model(X_train_t)
    y_pred_val, _ = model(X_val_t)
    y_pred_test, _ = model(X_test_t)

# === ВЫВОД МЕТРИК ДЛЯ ТАБЛИЦЫ ===
def get_metrics(y_true, y_pred):
    err = y_true - y_pred
    return np.max(np.abs(err)), np.std(err), np.sqrt(np.mean(err**2))

t_max, t_std, t_rmse = get_metrics(y_train, y_pred_train.numpy())
v_max, v_std, v_rmse = get_metrics(y_val, y_pred_val.numpy())
te_max, te_std, te_rmse = get_metrics(y_test, y_pred_test.numpy())

print(f"{'Выборка':<15} | {'Макс. абс. ошибка':<20} | {'С.к.о. ошибки':<20} | {'RMSE':<20}")
print("-" * 80)
print(f"{'Обучающая':<15} | {t_max:<20.4f} | {t_std:<20.4f} | {t_rmse:<20.4f}")
print(f"{'Валидационная':<15} | {v_max:<20.4f} | {v_std:<20.4f} | {v_rmse:<20.4f}")
print(f"{'Тестовая':<15} | {te_max:<20.4f} | {te_std:<20.4f} | {te_rmse:<20.4f}")

# === ОТРИСОВКА ГРАФИКОВ ДЛЯ ОТЧЕТА ===
plt.figure(figsize=(12, 18))

# График 1: Выход сети
plt.subplot(4, 1, 1)
plt.plot(X_norm, y_norm, 'b-', label='Истинная функция')
plt.plot(X_norm, y_pred_before.numpy(), 'g--', label='До обучения')
plt.plot(X_norm, y_pred_after.numpy(), 'r--', label='После обучения')
plt.title('y(x)')
plt.legend()

# График 2: Скрытые нейроны (До и После)
plt.subplot(4, 2, 3)
plt.plot(X_norm, hidden_before.numpy())
plt.title('Выходы до обучения')
plt.subplot(4, 2, 4)
plt.plot(X_norm, hidden_after.numpy())
plt.title('Выходы после обучения')

# График 3: Ошибка
plt.subplot(4, 1, 3)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.plot(test_losses, label='Test')
plt.title('Зависимость ошибки от времени обучения')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.yscale('log')
plt.legend()

# График 4: Веса
plt.subplot(4, 2, 7)
plt.plot(np.array(weights_hidden))
plt.title('Веса скрытого слоя')
plt.xlabel('Эпоха (x10)')
plt.subplot(4, 2, 8)
plt.plot(np.array(weights_out))
plt.title('Веса выходного нейрона')
plt.xlabel('Эпоха (x10)')

plt.tight_layout()
plt.show()
