import numpy as np
import matplotlib.pyplot as plt

# Генерируем 2000 точек от -3 до 3
X = np.linspace(-3, 3, 2000)
# Ваша функция
y = np.sin(5 * X) * np.exp(X / 2)

# Строим график
plt.figure(figsize=(8, 4))
plt.plot(X, y, 'b-', linewidth=2, label='y = sin(5x) * exp(x/2)')
plt.title('График аппроксимируемой функции')
plt.xlabel('Вход X')
plt.ylabel('Выход Y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('func_graph.png', dpi=300)
plt.show()