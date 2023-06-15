if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train


import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    # Split data into features and targets
    y_train = df_train.iloc[:, 0].values  # target
    X_train = df_train.iloc[:, 1:].values  # features

    y_test = df_test.iloc[:, 0].values  # target
    X_test = df_test.iloc[:, 1:].values  # feature
    
    # Standardize data
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    # Calculate lambda_max
    lambda_max = np.max(2 * np.abs(X_train.T @ (y_train - np.mean(y_train))))


    # Initialize variables
    lambdas = []
    w_s = []
    num_nonzeros = []


    # Initialize weights and bias
    w = np.zeros(X_train.shape[1])
    b = 0

    # Train model for each lambda value
    
    while lambda_max > 0.02:
        print(f"lambda: {lambda_max}")
        lambdas.append(lambda_max)
        w, b = train(X_train, y_train, lambda_max, start_weight=w, start_bias=b)
        w_s.append(w.flatten())
        num_nonzeros.append(np.count_nonzero(w))
        lambda_max /= 2
    np.savez('crime_results.npz', w_s=w_s, num_nonzeros=num_nonzeros, lambdas=lambdas)

    ###
    # load results from file
    data = np.load('crime_results.npz')
    w_s = data['w_s']
    num_nonzeros = data['num_nonzeros']
    lambdas = data['lambdas']
    
    # Plot lambda vs. number of nonzero weights
    plt.plot(np.reciprocal(lambdas), w_s, '-o')
    plt.xlabel('1/Lambda')
    plt.ylabel('Number of nonzero weights')
    plt.title('Lambda vs. Number of Nonzero Weights')
    plt.xscale('log')
    plt.show()

    # # Evaluate final model on test data
    # weights, _ = train(X_train, y_train, weights, lambda_max)
    # y_pred = X_test @ weights
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Test MSE: {mse}")






if __name__ == "__main__":
    main()
