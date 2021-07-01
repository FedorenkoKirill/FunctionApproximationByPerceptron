import matplotlib.pyplot as plt
import numpy as np
# создаём модель нейросети, используя Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


# задаём линейную функцию, которую попробуем приблизить нашей нейронной сетью
def f(x):
    return x * np.sin(x * 2 * np.pi) if x < 0 else -x * np.sin(x * np.pi) + np.exp(x / 2) - np.exp(0)


def baseline_model():
    model = Sequential()
    # Hidden - Layers
    model.add(Dense(70, input_dim=1, activation='tanh', init='he_normal'))
    model.add(Dense(70, input_dim=70, activation='tanh', init='he_normal'))
    # Output- Layer
    model.add(Dense(1, input_dim=70, activation='linear', init='he_normal'))
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

if __name__ == "__main__":
    # накидываем тысячу точек от -4 до 4
    x = np.linspace(-4, 4, 2000).reshape(-1, 1)
    f = np.vectorize(f)
    # вычисляем вектор значений функции
    y = f(x)

    # тренируем сеть
    model = baseline_model()
    model.fit(x, y, epochs=400, verbose=1)

    # отрисовываем результат приближения нейросетью поверх исходной функции
    plt.scatter(x, y, color='black', antialiased=True)
    plt.plot(x, model.predict(x), color='magenta', linewidth=2, antialiased=True)
    plt.show()

    # выводим веса на экран
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)

