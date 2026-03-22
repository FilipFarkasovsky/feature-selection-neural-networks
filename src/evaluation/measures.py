import numpy as np
from scipy.spatial.distance import jaccard, hamming, dice
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import combinations
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from util.features import is_discrete
from .util import partial_rank, ranked_to_permutation_list




def one_vs_all_roc_auc(y, y_pred):
    classes = np.unique(y)
    bin_true = label_binarize(y = y, classes = classes).transpose()
    bin_pred = label_binarize(y=y_pred, classes=classes).transpose()

    scores = [
        roc_auc_score(tr, pr)
        for tr, pr
        in zip(bin_true, bin_pred)
    ]

    return np.mean(scores)


def jaccard_score(s1, s2):
    is_list = isinstance(s1, list) and isinstance(s2, list) or isinstance(s1, np.ndarray) and isinstance(s1, np.ndarray)

    if is_list and len(s1) == len(s2):
        return 1 - jaccard(s1, s2)
    elif isinstance(s1, set) and isinstance(s2, set):
        intersection = s1.intersection(s2)
        union = s1.union(s2)
        return len(intersection) / len(union)
    else:
        raise TypeError("Only a pair of `sets`, `lists` or `numpy arrays` is allowed.")


def normalized_hamming_distance(a, b):
    try:
        return hamming(a, b)
    except Exception:
        raise TypeError("Only pair of `lists` or `numpy arrays` of the same size is allowed.")


def set_normalized_hamming_distance(a, b):
    if isinstance(a, set) and isinstance(b, set):
        return len(a.symmetric_difference(b)) / len(a.union(b))
    else:
        raise TypeError("A and B must be pair of `sets`")


def dice_coefficient(a, b):
    is_list = isinstance(a, list) and isinstance(b, list) or isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    if is_list and len(a) == len(b):
        return 1 - dice(a, b)
    elif isinstance(a, set) and isinstance(b, set):
        return 2 * len(a.intersection(b)) / (len(a) + len(b))
    else:
        raise TypeError("Only a pair of `sets`, `lists` or `numpy arrays` is allowed.")


def ochiai_index(a, b):
    if isinstance(a, set) and isinstance(b, set):
        intersection = a.intersection(b)
        return len(intersection) / np.sqrt(len(a) * len(b))
    else:
        raise TypeError("Only a pair of `sets` is allowed.")


def kuncheva_index(a, b, m):
    '''
    m = total number of features
    k = length of features - assumes equal length of a and b
    r = number of common elements in both signatures
    kuncheva_index = (r*m - k^2)/k*(m-k)
    '''

    feat_size = len(a)
    same_size = feat_size == len(b)
    if m == feat_size:
        return 1.0

    if same_size and isinstance(a, set) and isinstance(b, set):
        r = a.intersection(b)
        k = feat_size
        return (len(r) * m - np.power(k, 2)) / (k * (m - k))
    else:
        raise TypeError("Only a pair of `sets` of the same size is allowed.")


def percentage_of_overlapping_features(a, b):
    if isinstance(a, set) and isinstance(b, set):
        intersection = a.intersection(b)
        return len(intersection) / len(a)
    else:
        raise TypeError("Only a pair of `sets` is allowed.")


def mutual_information(X, y):
    mutual_info = mutual_info_classif if is_discrete(y) else mutual_info_regression

    if len(X.shape) == 1:
        X = np.array([X]).T

    return mutual_info(X, y, discrete_features=is_discrete(X))


def _single_feature_kruskal_wallis(X, y):
    return kruskal(*[X[y == c] for c in np.unique(y)])


def kruskal_wallis(X, y):
    if len(X.shape) == 1:
        return _single_feature_kruskal_wallis(X, y)

    results = np.array([_single_feature_kruskal_wallis(x, y) for x in X.T])
    scores, pvalues = results.T
    return scores, pvalues


def no_ties_spearmans_correlation(a, b):
    _a = np.array(a) + 1
    _b = np.array(b) + 1
    m = len(_a)
    return 1 - 6 * np.power(_a - _b, 2).sum() / (m * (np.power(m, 2) - 1))


def spearmans_correlation(a, b):
    return pearsons_correlation(a, b)


def spearmans_correlation_ranked_list(a, b):
    _a = ranked_to_permutation_list(a)
    _b = ranked_to_permutation_list(b)
    return spearmans_correlation(_a, _b)


def spearmans_correlation_partial_ranked_list(a, b):
    _a, _b = partial_rank(a, b)
    return spearmans_correlation(_a, _b)


def pearsons_correlation(a, b):
    _a = np.array(a) + 1
    _b = np.array(b) + 1
    a_mean = _a.mean()
    b_mean = _b.mean()

    div = ((_a - a_mean) * (_b - b_mean)).sum()
    quo = np.sqrt(np.power(_a - a_mean, 2).sum() * np.power(_b - b_mean, 2).sum())

    return div / quo


def pearsons_correlation_no_zeros(a, b):
    a, b = np.array([(a, b) for a, b in zip(a, b) if not (a == 0 and b == 0)]).T
    return pearsons_correlation(a, b)


def canberra_distance(a, b):
    if list(a) == list(b):
        return 0

    _a, _b = np.array([(a, b) for a, b in zip(a, b) if not (a == b)]).T
    _a = np.array(_a)
    _b = np.array(_b)
    return (np.abs(_a - _b) / (np.abs(_a) + np.abs(_b))).sum()


def canberra_distance_ranked_list(a, b):
    _a = ranked_to_permutation_list(a)
    _b = ranked_to_permutation_list(b)
    return canberra_distance(_a, _b)


def canberra_distance_partial_ranked_list(a, b):
    _a, _b = partial_rank(a, b)
    return canberra_distance(_a, _b)


def kendalls_tau_coefficient(a, b):
    if list(a) == list(b):
        return 1.0

    _a = np.array(a) + 1
    _b = np.array(b) + 1

    _pairs = combinations(zip(_a, _b), 2)

    pair_comp_results = [x[0][0] < x[1][0] and x[0][1] < x[1][1] for x in _pairs]

    num_concordant = np.count_nonzero(pair_comp_results)
    num_total = len(pair_comp_results)
    num_discordant = num_total - num_concordant

    return (num_concordant - num_discordant) / num_total


def kendalls_tau_ranked_list(a, b):
    _a = ranked_to_permutation_list(a)
    _b = ranked_to_permutation_list(b)
    return kendalls_tau_coefficient(_a, _b)


def kendalls_tau_partial_ranked_list(a, b):
    _a, _b = partial_rank(a, b)
    return kendalls_tau_coefficient(_a, _b)
