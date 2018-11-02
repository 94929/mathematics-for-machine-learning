
import numpy as np
import matplotlib.pyplot as plt


def basis_func(x, degree, btype):
    if btype == 'poly':
        # Polynomial of degree K
        return np.array([x**i for i in range(degree+1)])
    elif btype == 'trigo':
        # Trigonometric of degree K with unit frequency
        return np.array(
                    [
                        np.sin(2*np.pi*i*x) if i%2==1 else np.cos(2*np.pi*i*x)
                        for i in range(2*degree+1)
                    ]
                )
    elif btype == 'gauss':
        # Gaussian with scale l and means mu_j
        # TODO know where to retrieve the scale l and means mu_j
        # TODO not sure how to define this
        return None
    else:
        raise ValueError('basis type is invalid')

def build_design_matrix(dataset, degree, btype):
    return np.asmatrix(np.array(
            [basis_func(x, degree, btype) for x in dataset]
        ).reshape(len(dataset), -1))

def find_optimal_parameters(dm, y):
    return np.linalg.inv(dm.T*dm) * dm.T * y

def plot_dataset(X, Y, btype, xs, degrees, opt_thetas):

    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    for i, degree in enumerate(degrees):
        plt.plot(xs, basis_func(xs, degree, btype).T*opt_thetas[i], 
                color=colors[i], label='K='+str(degree))

    # PLT & UI
    plt.scatter(X, Y)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('A Regression Dataset')

    axes = plt.gca()
    axes.set_xlim([-1, 1.2])
    axes.set_ylim([-1.2, 20])

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Generate dataset
    N = 25
    X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
    Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

    # configure values
    degs = [1, 11]
    btype = 'trigo'

    # design matrices and optimal thetas
    dms = [build_design_matrix(X, deg, btype) for deg in degs]
    ots = [find_optimal_parameters(dm, Y) for dm in dms]

    # Plot
    xrange = np.linspace(-1, 1.2, 200)
    plot_dataset(X, Y, btype=btype, xs=xrange, degrees=degs, opt_thetas=ots)

