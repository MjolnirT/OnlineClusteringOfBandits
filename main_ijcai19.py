import numpy as np
import time
import random
from scipy.stats import ortho_group
from CLUB import CLUB
from BASE import LinUCB, LinUCB_IND, LinUCB_Cluster
from SCLUB import SCLUB
from ENVIRONMENT import Environment
from utlis import edge_probability
from experiment import generate_items, get_theta, get_half_frequency_vector


def main(num_stages, num_users, d, m, L, pj, filename=''):
    seed = int(time.time() * 100) % 399
    print("Seed = %d" % seed)
    np.random.seed(seed)
    random.seed(seed)

    if filename == '':
        thetam = generate_items(num_items=m, d=d)
        # thetam = np.concatenate((np.dot(np.concatenate((np.eye(m), np.zeros((m,d-m-1))), axis=1), ortho_group.rvs(d-1))/np.sqrt(2), np.ones((m,1))/np.sqrt(2)), axis=1)
        print("(Env)filename is null, generating thetam")
        print(thetam, '\n', [[np.linalg.norm(thetam[i, :] - thetam[j, :]) for j in range(i)] for i in range(1, m)])
        theta = get_theta(thetam, num_users, m)
        # print([np.linalg.norm(theta[0]-theta[i]) for i in range(num_users)])
    else:
        theta = np.load(filename)

    # set up frequency vector
    user_dist = ['uniform', 'half', 'arbitrary']

    # "uniform" represents that all users following a uniform distribution.
    # "half" represents that users in the same cluster follow a uniform distribution
    # but the distribution among clusters is arbitrary.
    # "arbitrary" represents that the distribution of all users is arbitrary

    uniform = list(np.ones(num_users) / num_users)
    half = get_half_frequency_vector(num_users=num_users, m=m)
    arbitrary = list(np.random.dirichlet(np.ones(num_users)))
    user_pmf = [uniform, half, arbitrary]
    # in user_pmf, user_pmf[0] is the pdf under "uniform" setting,
    # user_pmf[1] is the pdf under "half" setting and user_pmf[2] is the pdf
    # under "arbitrary" setting

    path = 'dataset/'
    model_names = ['club', 'linucb', 'ind', 'sclub']
    models = [CLUB, LinUCB, LinUCB_IND, SCLUB]
    # iterate over three environments
    for dist_idx in np.array(pj):
        print("(Env)" + user_dist[dist_idx])
        p = user_pmf[dist_idx]
        envir = Environment(L=L, d=d, m=m, num_users=num_users, p=p, theta=theta)

        for model_idx, model_name in enumerate(model_names):
            print("(model) Running " + model_name)
            start_time = time.time()

            model = models[model_idx](nu=num_users, d=d, T=2 ** num_stages - 1, edge_probability=edge_probability(num_users))
            model.run(envir)
            out_filename = path + model_name + '_' + user_dist[dist_idx] + '-' + filename[0:2] + '-' + 'kmeans'
            run_time = time.time() - start_time
            np.savez(out_filename, seed, model.rewards, model.best_rewards, run_time, model.num_clusters)


if __name__ == "__main__":
    # synthetic experiment with user number is 10**3 and m=10 clusters
    main(num_stages=15, num_users=1000, d=20, m=10, L=20, pj=[0])

    # Using SVD
    # main(num_stages=15, num_users=1000, d=20, m=10, L=20, pj=[0, 2], filename='ml_1000user_d20.npy')
    # main(num_stages=15, num_users=1000, d=20, m=10, L=20, pj=[0, 2], filename='yelp_1000user_d20.npy')

    # Using kmeans
    # main(num_stages = 15, num_users = 1000, d = 20, m = 10, L = 20, pj = [0,2], filename='ml_1000user_d20_m10.npy')
    # main(num_stages = 15, num_users = 1000, d = 20, m = 10, L = 20, pj = [0,2], filename='yelp_1000user_d20_m10.npy')
