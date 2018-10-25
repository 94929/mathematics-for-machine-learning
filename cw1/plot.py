
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm

from answers import grad_f2
from answers import grad_f3

B = np.array([[3, -1], [-1, 3]])
a = np.array([1, 0])
b = np.array([0, -1])

def f1(x):
    x1, x2 = x
    func = 4*x1**2 + 4*x2**2 - 2*x1*x2 -x1 -x2
    return func

def f2(x):
    x1, x2 = x
    func = np.sin(x1**2+x2**2-2*x1+1) + 3*x1**2+3*x2**2-2*x1*x2-2*x1+6*x2+3
    return func

def f3(x):
    x1, x2 = x
    term_1 = np.exp(-x1**2 -x2**2 +2*x1 -1)
    term_2 = np.exp(-3*x1**2 -3*x2**2 +2*x1*x2 +2*x1 -6*x2 -3)
    term_3 = 0.1*np.log(0.01*x1**2 + 0.01*x2**2 + 0.0001)
    func = 1 - term_1 - term_2 + term_3
    return func

def gradient_descent(f, x, step_size):

    # declare empty lists for steps and function values
    x1s, x2s = [], []

    # use correct gradient function
    grad_f = grad_f2 if f == f2 else grad_f3

    # gradient descent algorithm 
    for i in range(50):

        x1, x2 = x
        x1s.append(x1)
        x2s.append(x2)

        x = x - step_size * grad_f(x)

    # return steps and function values for plotting
    return np.column_stack((x1s, x2s))

def plot_contour_2d(f, path, minima):

    # preparing the configuration values
    xmin, xmax, xstep = -2.5, 2.5, .2
    ymin, ymax, ystep = -2.5, 2.5, .2
    x, y = np.meshgrid(
                np.arange(xmin, xmax + xstep, xstep), 
                np.arange(ymin, ymax + ystep, ystep)
            )
    z = f(np.array([x, y]))
    minima_ = minima.reshape(-1, 1)

    # plotting methods
    fig, ax = plt.subplots(figsize=(10, 6))

    lvls = np.logspace(-1, 3, 250)
    print(lvls)
    ax.contour(x, y, z, levels=lvls, norm=LogNorm(), cmap=plt.cm.jet)
    ax.quiver(path[0,:-1], path[1,:-1], 
            path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], 
            scale_units='xy', angles='xy', scale=1, color='k')
    ax.plot(*minima_, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    plt.show()

if __name__ == '__main__':
    # decide which function to use
    f = f3

    # obtain steps (i.e. path) by running the gradient descent algorithm
    path = gradient_descent(f, x=np.array([1, -1]), step_size=.1).T
    print('path x1', path[0,:])
    print('path x2', path[1,:])

    # set minima of the given function (should be obtained prior to plot)
    minima = np.array([0.14086436, -0.82233133])

    # plot contour (2d)
    plot_contour_2d(f, path, minima)

