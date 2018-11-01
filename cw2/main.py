
import numpy as np
import matplotlib.pyplot as plt


def basis_func(x, degree, btype):
    if btype == 'poly':
        # Polynomial of degree K
        return np.array([x**i for i in range(degree+1)])
    elif btype == 'trigo':
        # Trigonometric of degree K with unit frequency
        res = []
        for i in range(2*degree+1):
            if i % 2 == 1:
                v = np.sin(2*np.pi*i*x)
            else:
                v = np.cos(2*np.pi*i*x)
            res.append(v)
        return res
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

def plot_dataset(X, Y, degree, btype, xs, opt_theta):

    # main plotting algorithm
    plt.plot(xs, basis_func(xs, degree, btype).T*opt_theta)

    # PLT & UI
    plt.scatter(X, Y)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('A Regression Dataset')

    axes = plt.gca()
    axes.set_xlim([-0.3, 1.3])
    plt.show()

if __name__ == '__main__':
    # Generate dataset
    N = 25
    X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
    Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

    # Configure values
    degree = 4
    basis_type = 'poly'

    # Build a design matrix then find optimal parameters
    design_matrix = build_design_matrix(X, degree=degree, btype=basis_type)
    opt_theta = find_optimal_parameters(design_matrix, Y)

    # Plot
    xrange = np.linspace(-0.3, 1.3, 300)
    plot_dataset(X, Y, degree=degree, btype=basis_type, xs=xrange, opt_theta=opt_theta)

