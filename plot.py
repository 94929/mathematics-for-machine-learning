
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

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

def gradient_descent(f, x, step_size, nb_epochs):

    # declare empty lists for steps and function values
    x1_gd, x2_gd, fx_gd = [], [], []

    # use correct gradient function
    grad_f = grad_f2 if f == f2 else grad_f3

    # gradient descent algorithm 
    for i in range(nb_epochs):
        fx_gd.append(f(x))

        x1, x2 = x
        x1_gd.append(x1)
        x2_gd.append(x2)

        x = x - step_size * grad_f(x)

    # return steps and function values for plotting
    return x1_gd, x2_gd, fx_gd

def plot_contour_3d(f, x, y, z, x_gd, y_gd, z_gd):
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    surf = ax1.plot_surface(x, y, z, edgecolor='none', rstride=1,
                                    cstride=1, cmap='jet')

    # Plot target (the minimum of the function)
    min_point = np.array([0., 0.])
    min_point_ = min_point[:, np.newaxis]
    ax1.plot(*min_point_, f(np.array([*min_point_])), 'r*', markersize=10)

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_zlabel(r'$z$')

    # Create animation
    line, = ax1.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
    point, = ax1.plot([], [], [], 'bo')
    display_value = ax1.text(2., 2., 27.5, '', transform=ax1.transAxes)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        display_value.set_text('')

        return line, point, display_value

    def animate(i):
        # Animate line
        line.set_data(x_gd[:i], y_gd[:i])
        line.set_3d_properties(z_gd[:i])

        # Animate points
        point.set_data(x_gd[i], y_gd[i])
        point.set_3d_properties(z_gd[i])

        # Animate display value
        display_value.set_text('Min = ' + str(z_gd[i]))

        return line, point, display_value

    ax1.legend(loc = 1)

    anim = animation.FuncAnimation(fig1, animate, init_func=init,
        frames=len(x_gd), interval=120, repeat_delay=60, blit=True)

    plt.show()

if __name__ == '__main__':
    # decide which function to use
    f = f3

    # steps and corresponding function values after running gradient descent
    x_gd, y_gd, z_gd = gradient_descent(f, np.array([1, -1]), .1, 50)

    # plot contour for the output of the algorithm 
    a = np.arange(-7.5, 7.5, .1)
    b = np.arange(-7.5, 7.5, .1)
    x, y = np.meshgrid(a, b)
    z = f(np.array([x, y]))
    #plot_contour_2d(f, x, y, z, x_gd, y_gd, z_gd)
    plot_contour_3d(f, x, y, z, x_gd, y_gd, z_gd)

