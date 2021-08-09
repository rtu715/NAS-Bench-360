import numpy as np


def multinomial_KL_divergence(P, Q):
    """compute the KL-divergence for two metrics of identical shape

    Parameters
    ----------
    P : numpy.array
        n by 4 array, reference prob.
    Q : numpy.array
        target prob.

    Returns
    -------
    float : distance measured by KL-divergence
    """
    assert P.shape == Q.shape
    d = 0
    for j in range(P.shape[0]):
        idx = np.where(P[j, :] != 0)[0]
        d += np.sum(P[j, idx] * (np.log(P[j, idx]) - np.log(Q[j, idx])))
    return d


def bias_var_decomp(f, f_hat):
    """compute the mse for an estimator.
    .. math::

        mse(\hat{f}) = Bias(\hat{f})^2 + Var(\hat{f})

    Parameters
    ----------
    f : numpy.array
        observed/truth
    f_hat : numpy.array
        predicted values

    Returns
    --------
        float: mse

    """
    M = 10.
    bias2 = (f - np.mean(f_hat)) ** 2
    var = np.var(f_hat)
    if var == 0:
        score = M
    else:
        score = bias2 + var
    return score
