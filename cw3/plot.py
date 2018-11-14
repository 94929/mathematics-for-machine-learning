
import matplotlib.pyplot as plt
import numpy as np

from answers import lml
from answers import grad_lml
from prev_answers import poly_func
from prev_answers import trig_func

# use the dataset from the previous cw
N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

# define the basis function
Phi = poly_func(1, X)

def gradient_descent(step, step_size):

    # declare empty lists for steps and function values
    alphas, betas = [], []

    # gradient descent algorithm 
    for i in range(1000):

        alpha, beta = step
        alphas.append(alpha)
        betas.append(beta)

        # maximizing gradient descent algorithm
        step = step + step_size*grad_lml(alpha, beta, Phi, Y)

    # return steps and function values for plotting
    return np.column_stack((alphas, betas)).T

def plot_contour_2d(f, path):
    
    # preparing the configuration values
    # range should begin from non-zero as ln 0 is not defined
    xmin, xmax, xstep = .1, 2.5, .1
    ymin, ymax, ystep = .1, 2.5, .1
    x, y = np.meshgrid(
                np.arange(xmin, xmax + xstep, xstep), 
                np.arange(ymin, ymax + ystep, ystep)
            )

    # ensure x and y have the same shape
    assert x.shape == y.shape

    # fill z-values (i.e. the function values)
    xy = [list(zip(x_i, y_i)) for x_i, y_i in zip(x, y)]
    flatten_xy = [xy[i][j] for i in range(len(xy)) for j in range(len(xy[0]))]
    raw_z = np.array([f(x_i, y_i, Phi, Y) for x_i, y_i in flatten_xy])
    z = raw_z.reshape(x.shape)

    # define maxima
    maxima = np.array([path[0, -1], path[1, -1]]).reshape(-1, 1)

    # plotting methods
    fig, ax = plt.subplots(figsize=(10, 6))

    # the z-value lies between -70 and -25
    lvls = np.linspace(-70, -25, 100)
    ax.contour(x, y, z, levels=lvls)
    ax.quiver(path[0,:-1], path[1,:-1], 
            path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], 
            scale_units='xy', angles='xy', scale=1, color='k')
    ax.plot(*maxima, 'r*', markersize=15)

    ax.set_xlabel('$alpha$')
    ax.set_ylabel('$beta$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((xmin, xmax))

    plt.show()

if __name__ == '__main__':
    # obtain steps (i.e. path) by running the gradient descent algorithm
    path = gradient_descent(step=np.array([.35, .35]), step_size=.001)
    print('maxima a:', path[0,-1])
    print('maxima b:', path[1,-1])

    # plot contour (2d)
    plot_contour_2d(lml, path)
    

