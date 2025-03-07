import torch
import torch.nn as nn
import torch.optim as optim


# Определяем  нейронную сеть
class myNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(myNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Первый полносвязный слой
        self.relu = nn.ReLU()  # Функция активации ReLU
        self.fc2 = nn.Linear(hidden_size, output_size)  # Выходной слой

    def forward(self, x):
        x = self.fc1(x)  # Применяем первый слой
        x = self.relu(x)  # Применяем функцию активации
        x = self.fc2(x)  # Применяем выходной слой
        return x


# Задаем размеры входного, скрытого и выходного слоев
input_size = 3  # Количество входных признаков
hidden_size = 5  # Количество нейронов в скрытом слое
output_size = 1  # Размер выходного слоя

# Создаем экземпляр модели
model = myNN(input_size, hidden_size, output_size)

# Определяем функцию потерь (MSE - среднеквадратичная ошибка)
criterion = nn.MSELoss()

# Оптимизатор (Стохастический градиентный спуск с шагом 0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Генерируем случайные входные данные и ожидаемые выходные значения
X = torch.randn(10, input_size)  # 10 примеров, каждый с 3 признаками
y = torch.randn(10, output_size)  # 10 целевых значений

# Обучение модели
epochs = 100  # Количество эпох (итераций)
for epoch in range(epochs):
    optimizer.zero_grad()  # Обнуляем градиенты перед следующим шагом
    outputs = model(X)  # Прямой проход (forward pass)
    loss = criterion(outputs, y)  # Вычисляем ошибку
    loss.backward()  # Обратное распространение ошибки (backpropagation)
    optimizer.step()  # Обновление весов модели

    if epoch % 10 == 0:  # Выводим ошибку каждые 10 эпох
        print(f"Эпоха {epoch}, Ошибка: {loss.item():.4f}")

# Сохранение обученной модели
torch.save(model.state_dict(), "model.pth")
print("Модель успешно сохранена.")
