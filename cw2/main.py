
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
    squared_losses_test = []
    for train_index, test_index in loo.split(X):
        # prepare train and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # find squared loss for test set
        dm_test = build_design_matrix(X_test, degree, btype)
        ot_test = find_optimal_parameters(dm_test, y_test)
        squared_loss_test = calculate_squared_loss(y_test, dm_test, ot_test)
        squared_loss_test_scalar = np.asscalar(squared_loss_test)
        squared_losses_test.append(squared_loss_test_scalar)

    # calculate mean values
    mean_squared_loss_test = sum(squared_losses_test) / nb_splits

    # find maximum likelihood for variance for the whole dataset
    dm = build_design_matrix(X, degree, btype)
    ot = find_optimal_parameters(dm, Y)
    squared_loss_mle_var = calculate_squared_loss(Y, dm, ot)
    squared_loss_mle_var_scalar = np.asscalar(squared_loss_mle_var)

    return mean_squared_loss_test, squared_loss_mle_var_scalar

def get_errors(N, X, Y, degrees):

    test_errors = []
    mle_vars = []
    for degree in degrees:
        
        # find mean squared loss for train and test set for a given degree K
        mean_squared_loss_test, mle_var = (
                    find_squared_losses(N, X, Y, degree=degree, btype='trigo')
                )

        # store losses for plotting
        test_errors.append(mean_squared_loss_test)
        mle_vars.append(mle_var)

    return test_errors, mle_vars

def plot_errors(degrees, e_test, mle_var):
    """ plot the result for q1c """
    plt.plot(degrees, e_test, color='blue')
    plt.plot(degrees, mle_var, color='red')

    plt.legend(['Test error', 'ML for variance'], loc='upper left')
    plt.xlabel('Degree of trigonometric basis')
    plt.ylabel('Mean squared error')
    plt.title('Test error')

    axes = plt.gca()
    axes.set_ylim([0, 20])

    plt.show()

if __name__ == '__main__':
    # generate dataset
    N = 25
    X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
    Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

    # configure degrees to iterate
    degrees = list(range(11))

    # obtain errors after running leave one out validation
    e_test, mle_var = get_errors(N, X, Y, degrees)

    # plot the result
    plot_errors(degrees, e_test, mle_var)

