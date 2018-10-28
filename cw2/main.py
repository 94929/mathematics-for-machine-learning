
import numpy as np
import matplotlib.pyplot as plt

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

def plot_dataset(X, Y):
    plt.scatter(X, Y)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('A Regression Dataset')

    axes = plt.gca()
    axes.set_xlim([-0.3, 1.3])
    plt.show()

if __name__ == '__main__':
    plot_dataset(X, Y)

