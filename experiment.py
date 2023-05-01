import numpy as np


def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d - 1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis=1),
                                              np.ones(np.shape(x)[1]))) /
                        np.sqrt(2),
                        np.ones((num_items, 1)) / np.sqrt(2)), axis=1)
    return x


# generating user features by repeating the cluster features, thetam, by num_users / m times
def get_theta(thetam, num_users, m):
    # generating theta vector
    k = int(num_users / m)
    theta = {i: thetam[0] for i in range(k)}
    for j in range(1, m):
        theta.update({i: thetam[j] for i in range(k * j, k * (j + 1))})
    return theta


# generating pmf vector for the "half" setting
def get_half_frequency_vector(num_users, m):
    p0 = list(np.random.dirichlet(np.ones(m)))
    p = np.ones(num_users)
    k = int(num_users / m)
    for j in range(m):
        for i in range(k * j, k * (j + 1)):
            p[i] = p0[j] / k
    p = list(p)
    return p
