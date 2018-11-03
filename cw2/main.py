#!/Users/jha/miniconda3/bin/python3.6

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut


def basis_func(x, degree, btype):
    if btype == 'poly':
        # Polynomial of degree K
        return np.array([x**i for i in range(degree+1)])
    elif btype == 'trigo':
        # Trigonometric of degree K with unit frequency
        res = (np.array([1]) + 
                [np.sin(np.pi*(i+1)*x) if i%2==1 else np.cos(np.pi*i*x) 
                    for i in range(1, 2*degree+1)]
            )
        return np.array(res, dtype=float)
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

    # TODO create a list of infinite number of colors (i.e. generator)
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
    axes.set_ylim([-1.2, 2])

    plt.legend()
    plt.show()

def calculate_squared_error_loss(Y, design_matrix, optimal_theta):
    return (Y-design_matrix*optimal_theta).T * (Y-design_matrix*optimal_theta)

if __name__ == '__main__':
    # Generate dataset
    N = 25
    X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
    Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

    # configure values
    deg = 2
    btype = 'trigo'

    # design matrices and optimal thetas
    dm = build_design_matrix(X, deg, btype)
    ot = find_optimal_parameters(dm, Y) 

    # Q1C, when degree is 0
    print(calculate_squared_error_loss(Y, dm, ot))
    
    loo = LeaveOneOut()
    nb_splits = loo.get_n_splits(X)
    print('nb_splits:', nb_splits)

    for train_index, test_index in loo.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        print('X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(
                X_train, X_test, y_train, y_test)
            )
        break

