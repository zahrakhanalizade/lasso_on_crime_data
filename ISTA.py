from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    n, d = X.shape

    # Calculate predictions and errors
    predictions = X @ weight + bias # Runtime - invalid value encountered in matmul
    errors = predictions - y
    
    # Update bias
    bias_prime = bias - 2 * eta * np.sum(errors)

    # Check for NaN or infinite values
    # if np.isnan(errors).any() or np.isinf(errors).any() or np.isnan(X.T).any() or np.isinf(X.T).any():
    #     return weight, bias

    # Update weights
    weight_prime = weight - 2 * eta * (X.T @ errors) # Overflow
    weight_prime = np.where(weight_prime < -2*_lambda*eta, weight_prime + 2*eta*_lambda, 
                        np.where(weight_prime > 2*_lambda*eta, weight_prime - 2*eta*_lambda, 0))
    
    # for k in range (d):
    #     weight[k]= weight[k] - 2 *eta* np.sum(X[:, k] * ((X @ weight)-y + bias))
    #     if weight[k] < -2*_lambda*eta:
    #         weight[k] = weight[k] + 2*eta*_lambda
    #     elif weight[k] > 2*_lambda*eta:
    #         weight[k] = weight[k] - 2*eta*_lambda
    #     else:
    #         weight[k] = 0
    # weight_prime = weight

    # print(weight_prime)

    return weight_prime, bias_prime





@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    loss = np.sum((((X @ weight)-y + bias))**2) + _lambda * np.sum(abs(weight))

    return loss


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    if start_bias is None:
        start_bias = 0
   
    weight = np.copy(start_weight)
    bias = start_bias
    i = 0
    converged = False
    while not converged:
        # update weights
        old_w = np.copy(weight)
        old_b = np.copy(bias)
        weight, bias = step(X, y, weight, bias, _lambda, eta)
        # check for convergence
        if old_w is not None and convergence_criterion(weight, old_w, bias, old_b, convergence_delta):
            converged = True
        i += 1
        # print(f"{i}: w: {weight}, b: {bias}")
    return weight, bias

    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    # print(f"conv cri: weight:{weight}, bias: {bias}")
    max_abs_change = np.nanmax(np.abs(weight - old_w))
    bias_abs_change = np.abs(bias - old_b)
    if max_abs_change <= convergence_delta and bias_abs_change <= convergence_delta:
        return True
    else:
        return False
    
@problem.tag("hw2-A")
def plot_regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas,
    true_w,
    eta: float = 0.001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None,
    plot_num_nonzeros: bool = True,
):
    """
    Plots the regularization path for Lasso regression on the given data.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        lambdas (np.ndarray): An (k, n) array
        eta (float, optional): Step size to use in gradient descent. Defaults to 0.001.
        convergence_delta (float, optional): The convergence threshold for the gradient descent algorithm.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): The initial weights to use. Defaults to None.
        start_bias (float, optional): The initial bias to use. Defaults to None.
        plot_num_nonzeros (bool, optional): Whether to plot the number of non-zeros as a function of lambda.
            If False, plots the MSE as a function of lambda. Defaults to True.
    """
    d = X.shape[1]

    # Compute lambda_max if not specified
    lambda_max = np.max(lambdas)
    lambda_min = np.min(lambdas)

    # Initialize weights and bias
    if start_weight is None:
        w = np.zeros(X.shape[1])
    else:
        w = start_weight
    if start_bias is None:
        b = 0
    else:
        b = start_bias

    # Initialize lambda values
    num_lambdas = len(lambdas)
    # lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), num_lambdas, endpoint=True)

    # Initialize arrays to store results
    mse_vals = np.zeros(num_lambdas)
    num_nonzeros = np.zeros(num_lambdas)

    k = np.count_nonzero(true_w)
    fdr_list = []
    tpr_list = []

    w_s = np.zeros((d, num_lambdas))

    # Train model for each lambda value
    for i, lam in enumerate(lambdas):
        w, b = train(X, y, lam, eta, convergence_delta, w, b)
        w_s[:, i] = w.flatten()
        mse_vals[i] = loss(X, y, w, b, lam)
        num_nonzeros[i] = np.count_nonzero(w)
        # print(f"Loop {i}th done, mse = {mse_vals[i]}")
        fdr, tpr = calculate_fdr_tpr(w, true_w, k)
        fdr_list.append(fdr)
        tpr_list.append(tpr)


    # Regularization Path
    plt.figure()
    for i in range(20):
        plt.plot(np.reciprocal(lambdas), w_s[-20+i, :], label=f"Feature {d-20+i}")
    plt.xlabel("1/Lambda")
    plt.ylabel("Weight")
    plt.xscale("log")
    plt.title("Regularization Path")    
    plt.legend()
    plt.show()
    
    # # Plot number of non-zero coefficients as a function of lambda
    if plot_num_nonzeros:    
        nonzero_counts = [np.count_nonzero(w) for w in w_s.T]
        plt.plot(np.reciprocal(lambdas), nonzero_counts)
        plt.xlabel('1/Lambda')
        plt.ylabel('Number of Nonzero Weights')
        plt.xscale('log')
        plt.show()

    # # Plot MSE as a function of lambda
    plt.plot(np.reciprocal(lambdas), mse_vals)
    plt.xscale('log')
    plt.xlabel('1/Lambda')
    plt.ylabel('MSE')
    plt.title('Regularization Path')
    plt.show()

    plt.plot(fdr_list, tpr_list)
    plt.xlabel("False Discovery Rate")
    plt.ylabel("True Positive Rate")
    plt.title("FDR vs TPR Curve")
    plt.show()




@problem.tag("hw2-A")
def calculate_fdr_tpr(w_hat, w, k):
    """Calculates the false discovery rate (FDR) and true positive rate (TPR) of the weight vector w_hat, given the true weight vector w.

    Args:
        w_hat (np.ndarray): The weight vector estimated by the Lasso regression.
        w (np.ndarray): The true weight vector.
        k (int): The number of non-zero elements in the true weight vector.

    Returns:
        Tuple[float, float]: A tuple of FDR and TPR values.
    """
    # Find indices of non-zero elements in w_hat and w
    w_hat_nonzero = np.nonzero(w_hat)[0]
    w_nonzero = np.nonzero(w)[0]

    # Calculate false discovery rate (FDR)
    if len(w_hat_nonzero) == 0:
        fdr = 0
    else:
        incorrect_nonzeros = len(np.setdiff1d(w_hat_nonzero, w_nonzero))
        fdr = incorrect_nonzeros / len(w_hat_nonzero)

    # Calculate true positive rate (TPR)
    correct_nonzeros = len(np.intersect1d(w_hat_nonzero, w_nonzero))
    tpr = correct_nonzeros / k

    return fdr, tpr

@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    # raise NotImplementedError("Your Code Goes Here")

    # Parameters:

    n = 500
    d = 1000
    k = 100
    sigma = 1

    # Data Generating Process:

    w = np.zeros(d)
    w[:k] = np.arange(1, k+1) / k
    # print(f"true w: {w}")
    # Generate the feature matrix
    X = np.random.normal(loc=0, scale=1, size=(n, d))
    # Generate the noise
    eps = np.random.normal(loc=0, scale=1, size=(n,))

    # Generate the target vector
    y = X.dot(w) + eps

    # Standardize the feature matrix (Actually no need for this since I've drawn X from standard normal but for the sake of genericness!)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Calculate lambda_max
    lambda_max = np.max(2 * np.abs(X.T @ (y - np.mean(y))))

    # Set up a list of lambdas to evaluate
    lambdas = [lambda_max / (2 ** i) for i in range(20)]

    # Generate and plot the regularization path
    plot_regularization_path(
        X=X,
        y=y,
        lambdas=lambdas,
        true_w = w,
        plot_num_nonzeros=True,
    )



if __name__ == "__main__":
    main()
