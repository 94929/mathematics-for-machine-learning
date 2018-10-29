
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

def basis_func(x, degree, type):
    if type == 'polynomial':
        # Polynomial of degree K
        return np.array([x**i for i in range(degree+1)])
    elif type == 'trigonometric':
        # Trigonometric of degree K with unit frequency
        return 2
    elif type == 'gaussian':
        # Gaussian with scale l and means mu_j
        return 3
    else:
        raise ValueError('basis type is invalid')

def build_design_matrix(dataset, basis, degree, type):
    return np.asmatrix(np.array(
            [basis_func(x, degree, type) for x in dataset]
        ).reshape(len(dataset), -1))

def find_optimal_parameters(dm, y):
    return np.linalg.inv(dm.T*dm) * dm.T * y

if __name__ == '__main__':
    design_matrix = build_design_matrix(X, basis_func, degree=2, type='polynomial')
    opt_theta = find_optimal_parameters(design_matrix, Y)
    print(opt_theta)

    #plot_dataset(X, Y)

