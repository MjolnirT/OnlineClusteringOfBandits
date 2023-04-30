import numpy as np
import sys


def isInvertible(S):
    return np.linalg.cond(S) < 1 / sys.float_info.epsilon


def edge_probability(n):
    return 3 * np.log(n) / n


def is_power2(n):
    return n > 0 and ((n & (n - 1)) == 0)


