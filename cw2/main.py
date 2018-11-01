
import numpy as np
import matplotlib.pyplot as plt

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

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

def build_design_matrix(dataset, basis, degree, btype):
    return np.asmatrix(np.array(
            [basis(x, degree, btype) for x in dataset]
        ).reshape(len(dataset), -1))

def find_optimal_parameters(dm, y):
    return np.linalg.inv(dm.T*dm) * dm.T * y

def plot_dataset(X, Y, degree, btype, xs, opt_theta):

    # main plotting algorithm
    plt.plot(xs, basis_func(xs, degree, btype).astype(np.int).T*opt_theta)

    # PLT & UI
    plt.scatter(X, Y)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('A Regression Dataset')

    axes = plt.gca()
    axes.set_xlim([-0.3, 1.3])
    plt.show()

if __name__ == '__main__':
    # Build a design matrix 
    design_matrix = build_design_matrix(X, basis=basis_func, degree=2, btype='poly')
    opt_theta = find_optimal_parameters(design_matrix, Y)

    x_range = np.linspace(-0.3, 1.3, 500)
    plot_dataset(X, Y, degree=2, btype='poly', xs=x_range, opt_theta=opt_theta)

