
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
    try:
        return np.linalg.inv(dm.T*dm) * dm.T * y
    except np.linalg.linalg.LinAlgError:
        # make the singular matrix (dm.T*dm) non-singular
        return np.linalg.inv((dm.T*dm) + (np.eye(dm.shape[1])*1e-10)) * dm.T * y

def calculate_squared_loss(Y, design_matrix, optimal_theta):
    return (Y-design_matrix*optimal_theta).T * (Y-design_matrix*optimal_theta)

def find_squared_losses(N, X, Y, degree, btype):
    """ returns squared losses for both train and test when loo is performed """

    # design matrix and optimal parameters
    dm = build_design_matrix(X, degree, btype)
    ot = find_optimal_parameters(dm, Y) 

    # init leave-one-out validator
    loo = LeaveOneOut()
    nb_splits = loo.get_n_splits(X)

    # run leave-one-out validation to calculate the losses
    squared_losses_train = []
    squared_losses_test = []
    for train_index, test_index in loo.split(X):
        # prepare train and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # find squared loss for train set
        dm_train = build_design_matrix(X_train, degree, btype)
        ot_train = find_optimal_parameters(dm_train, y_train)
        squared_loss_train = calculate_squared_loss(y_train, dm_train, ot_train)
        squared_loss_train_scalar = np.asscalar(squared_loss_train)
        squared_losses_train.append(squared_loss_train_scalar)

        # find squared loss for test set
        dm_test = build_design_matrix(X_test, degree, btype)
        ot_test = find_optimal_parameters(dm_test, y_test)
        squared_loss_test = calculate_squared_loss(y_test, dm_test, ot_test)
        squared_loss_test_scalar = np.asscalar(squared_loss_test)
        squared_losses_test.append(squared_loss_test_scalar)

    # calculate mean values
    mean_squared_loss_train = sum(squared_losses_train) / nb_splits
    mean_squared_loss_test = sum(squared_losses_test) / nb_splits

    return mean_squared_loss_train, mean_squared_loss_test

def get_errors(N, X, Y, degrees):

    train_errors = []
    test_errors = []
    for degree in degrees:

        # find mean squared loss for train and test set for a given degree K
        mean_squared_loss_train, mean_squared_loss_test = (
                    find_squared_losses(N, X, Y, degree=degree, btype='trigo')
                )

        # store losses for plotting
        train_errors.append(mean_squared_loss_train)
        test_errors.append(mean_squared_loss_test)

    return train_errors, test_errors

def plot_errors(degrees, e_train, e_test):
    """ plot the result for q1c """
    plt.plot(degrees, e_train, color='red')
    plt.plot(degrees, e_test, color='blue')

    plt.legend(['Training error', 'Test error'], loc='upper left')
    plt.xlabel('Degree of trigonometric basis')
    plt.ylabel('Mean squared error')
    plt.title('Training and test error')

    plt.show()

if __name__ == '__main__':
    # generate dataset
    N = 25
    X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
    Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

    # configure degrees to iterate
    degrees = list(range(11))

    # obtain errors after running leave one out validation
    e_train, e_test = get_errors(N, X, Y, degrees)

    # plot the result
    plot_errors(degrees, e_train, e_test)

