# Lasso

## ISTA
In this project, I implement the iterative shrinkage thresholding algorithm (ISTA). The functions implemented are  `step`, `loss`, or `convergence_criterion`in the file [ISTA.py](./ISTA.py) and `train` and then `main` functions

The `main` function repeatedly calls `train` and records various quantities around non-zero entries in returned weight vector. Also, it generates two plots: In plot 1, the number of non-zeros as a function of $\lambda$ on the x-axis is plotted. In plot 2, TPR is plotted against FPR.


## Crime Data Lasso
In this project, the ISTA (Iterative Shrinkage thresholding algorithm) is implemented and applied on real-world social data.
