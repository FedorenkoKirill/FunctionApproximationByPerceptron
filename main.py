import matplotlib.pyplot as plt
import numpy as np
# создаём модель нейросети, используя Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


# задаём линейную функцию, которую попробуем приблизить нашей нейронной сетью
def f(x):
    return 2 * np.sin(x) + 5


def baseline_model():
    model = Sequential()
    # Hidden - Layers
    model.add(Dense(100, input_dim=1, activation='relu'))
    # Output- Layer
    model.add(Dense(1, input_dim=100, activation='linear'))
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

if __name__ == "__main__":
    # накидываем тысячу точек от -3 до 3
    x = np.linspace(-3, 3, 1000).reshape(-1, 1)
    f = np.vectorize(f)
    # вычисляем вектор значений функции
    y = f(x)

    # тренируем сеть
    model = baseline_model()
    model.fit(x, y, epochs=100, verbose=0)

    # отрисовываем результат приближения нейросетью поверх исходной функции
    plt.scatter(x, y, color='black', antialiased=True)
    plt.plot(x, model.predict(x), color='magenta', linewidth=2, antialiased=True)
    plt.show()

    # выводим веса на экран
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)

