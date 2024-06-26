import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import percentileofscore

#This code is copypasted and taken from https://github.com/Netflix/vmaf/ and Source: https://github.com/qbarthelemy/PyPermut/blob/main/examples/compute_auroc_pvalue.py
#All rights belong to the original authors.

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)



def permutation_metric(y_true, y_score, func, *, n=10000, side='right'):
    """Permutation test for machine learning metric.

    This function performs a permutation test on any metric based on the
    predictions of a model. It permutes labels and predictions to obtain a
    p-value for any machine learning metrics:

    * the Area Under the Receiver Operating Characteristic (AUROC) curve,
    * the Area Under the Precision-Recall (AUPR) curve,
    * the negative log-likelihood (log-loss),
    * etc.

    Parameters
    ----------
    y_true : array_like, shape (n_samples, n_classes)
        True binary labels, with first dimension representing the sample
        dimension and with second dimension representing the different classes.

    y_score : array_like, shape (n_samples, n_classes)
        Scores of prediction, same dimensions as y_true. Scores can be
        probabilities or labels.

    func : callable
        Function to compute the metric, with signature `func(y_true, y_score)`.

    n : int (default 10000)
        Number of permutations for the permutation test.

    side : string (default 'right')
        Side of the test:

        * 'left' for a left-sided test,
        * 'two' or 'double' for a two-sided test,
        * 'right' for a right-sided test.

    Returns
    -------
    m : float
        The value of the metric.

    pval : float
        The p-value associated to the metric.

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(
            'Inputs y_true and y_score do not have compatible dimensions: '
            'y_true is of dimension {} while y_score is {}.'
            .format(y_true.shape, y_score.shape))
    n_samples = y_true.shape[0]

    # under the null hypothesis, sample the metric distribution
    null_dist = np.empty(n, dtype=float)
    for p in range(n):
        permuted_indices = np.random.permutation(n_samples)
        null_dist[p] = func(y_true[permuted_indices], y_score)

    # compute the real metric
    m = func(y_true, y_score)
    perc = percentileofscore(null_dist, m, kind='strict')
    pval = perc_to_pval(perc, side)

    return m, pval


def perc_to_pval(perc, side):
    """Transform percentile into p-value, depending on the side of the test.

    Parameters
    ----------
    perc : float
        Percentile of the observed statistic, in [0, 100].

    side : string
        Side of the test:

        * 'left', 'lower' or 'less', for a left-sided test;
        * 'two', 'double' or 'two-sided', for a two-sided test;
        * 'right', 'upper' or 'greater', for a right-sided test.

    Returns
    -------
    pval : float
        The p-value associated to the stat.
    """
    if not 0 <= perc <= 100:
        raise ValueError('Input percentile="{}" must be included in [0, 100].'
                         .format(perc))

    if side in ['left', 'lower', 'less']:
        pval = perc / 100
    elif side in ['two', 'double', 'two-sided']:
        pval = 2 * min(perc / 100, (100 - perc) / 100)
    elif side in ['right', 'upper', 'greater']:
        pval = (100 - perc) / 100
    else:
        raise ValueError('Invalid value for side="{}".'.format(side))

    return pval
