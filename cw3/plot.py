
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm

from answers import lml
from answers import grad_lml

# use the dataset from the previous cw
N = 25
X = Phi = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

def gradient_descent(step, step_size):

    # declare empty lists for steps and function values
    alphas, betas = [], []

    # gradient descent algorithm 
    for i in range(100):

        alpha, beta = step
        alphas.append(alpha)
        betas.append(beta)

        # maximizing gradient descent algorithm
        step = step + step_size*grad_lml(alpha, beta, Phi, Y)

    # return steps and function values for plotting
    return np.column_stack((alphas, betas)).T

def poly_func(order, x):
    a = np.ones((x.shape[0], order+1))
    for i in range(1, order+1):
        a[:, i] = a[:, i-1] * x.ravel()
    return a

def plot_contour_2d(f, path):
    
    # define the basis function
    Phi = poly_func(1, X)

    # preparing the configuration values
    # range should begin from non-zero as ln 0 is not defined
    xmin, xmax, xstep = .1, 3, .1
    ymin, ymax, ystep = .1, 3, .1
    x, y = np.meshgrid(
                np.arange(xmin, xmax + xstep, xstep), 
                np.arange(ymin, ymax + ystep, ystep)
            )

    # ensure x and y have the same shape
    assert x.shape == y.shape

    # fill z-values (i.e. the function values)
    # TODO: check whether the 3 lines of code below works exactly as expected
    xys = [list(zip(x_i, y_i)) for x_i, y_i in zip(x, y)]
    flatten_xys = [xys[i][j] for i in range(len(xys)) for j in range(len(xys[0]))]
    z = np.array([f(x_i, y_i, Phi, Y) for x_i, y_i in flatten_xys]).reshape(x.shape)

    maxima = np.array([path[0, -1], path[1, -1]]).reshape(-1, 1)

    # plotting methods
    fig, ax = plt.subplots(figsize=(10, 6))

    lvls = np.logspace(-1, 3, 250)
    ax.contour(x, y, z, levels=lvls, norm=LogNorm(), cmap=plt.cm.jet)
    ax.quiver(path[0,:-1], path[1,:-1], 
            path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], 
            scale_units='xy', angles='xy', scale=1, color='k')
    ax.plot(*maxima, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((xmin, xmax))

    plt.show()

if __name__ == '__main__':
    # obtain steps (i.e. path) by running the gradient descent algorithm
    path = gradient_descent(step=np.array([.75, .75]), step_size=.1)
    #print('Path a:', path[0,:])
    #print('Path b:', path[1,:])
    print('Maxima a:', path[0, -1])
    print('Maxima b:', path[1, -1])

    # plot contour (2d)
    plot_contour_2d(lml, path)

