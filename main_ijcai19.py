import numpy as np
import time
import random
from scipy.stats import ortho_group

from CLUB import CLUB
from BASE import LinUCB, LinUCB_IND, LinUCB_Cluster
from SCLUB import SCLUB
from ENVIRONMENT import Environment

from utlis import generate_items, edge_probability

def main(num_stages, num_users, d, m, L, pj, filename=''):
    seed = int(time.time() * 100) % 399
    print("Seed = %d" % seed)
    np.random.seed(seed)
    random.seed(seed)

    # set up theta vector
    def _get_theta(thetam, num_users, m):
        k = int(num_users / m)
        theta = {i:thetam[0] for i in range(k)}
        for j in range(1, m):
            theta.update({i:thetam[j] for i in range(k * j, k * (j + 1))})
        return theta

    if filename == '':
        thetam = generate_items(num_items=m, d=d)
        # thetam = np.concatenate((np.dot(np.concatenate((np.eye(m), np.zeros((m,d-m-1))), axis=1), ortho_group.rvs(d-1))/np.sqrt(2), np.ones((m,1))/np.sqrt(2)), axis=1)
        print("(Env)filename is null, generating thetam")
        print(thetam, '\n', [[np.linalg.norm(thetam[i,:]-thetam[j,:]) for j in range(i)] for i in range(1,m)])
        theta = _get_theta(thetam, num_users, m)
        # print([np.linalg.norm(theta[0]-theta[i]) for i in range(num_users)])
    else:
        theta = np.load(filename)
        
    # set up frequency vector
    uniforms = ['uniform', 'half', 'arbitrary'] 
    # "uniform" represents that all users following a uniform distribution
    # "half" represents that users in the same cluster follow a uniform distribution but the distribution among clusters is arbitrary
    # "arbitrary" represents that the distribution of all users is arbitrary
    def _get_half_frequency_vector(num_users, m):
        p0 = list(np.random.dirichlet(np.ones(m)))
        p = np.ones(num_users)
        k = int(num_users / m)
        for j in range(m):
            for i in range(k*j, k*(j+1)):
                p[i] = p0[j] / k
        p = list(p)
        return p
    ps = [list(np.ones(num_users) / num_users), _get_half_frequency_vector(num_users=num_users,m=m), list(np.random.dirichlet(np.ones(num_users)))]
    # in ps, ps[0] is the pdf under "uniform" setting, ps[1] is the pdf under "half" setting and ps[2] is the pdf under "arbitrary" setting
    
    # iterate over three envorinments
    for j in np.array(pj):
        print("(Env)" + uniforms[j])
        p = ps[j]
        envir = Environment(L = L, d = d, m = m, num_users = num_users, p = p, theta = theta)

        print("(model) Running CLUB")
        club = CLUB(nu = num_users, d = d, T = 2 ** num_stages - 1, edge_probability = edge_probability(num_users))
        start_time = time.time()
        club.run(envir)
        run_time = time.time() - start_time
        # np.savez('club_' + uniforms[j] + '_nu' + str(num_users) + 'd' + str(d) + 'm' + str(m) + 'L' + str(L) + '_'+str(seed), 
        #          seed, club.rewards, club.best_rewards, run_time, club.num_clusters)
        np.savez('club_' + uniforms[j] + '-' + filename[0:2], 
                 seed, club.rewards, club.best_rewards, run_time, club.num_clusters)

        print("(model) Running LinUCB")
        linucb = LinUCB(d = d, T = 2 ** num_stages - 1)
        start_time = time.time()
        linucb.run(envir)
        run_time = time.time() - start_time
        # np.savez('linucb_' + uniforms[j] + '_nu'+ str(num_users) + 'd' + str(d) + 'm' + str(m) + 'L' + str(L) + '_'+str(seed), 
        #          seed, linucb.rewards, linucb.best_rewards, run_time)
        np.savez('linucb_' + uniforms[j] + '-' + filename[0:2], 
                 seed, linucb.rewards, linucb.best_rewards, run_time)

        print("(model) Running LinUCB_IND")
        ind = LinUCB_IND(nu = num_users, d = d, T = 2 ** num_stages - 1)
        start_time = time.time()
        ind.run(envir)
        run_time = time.time() - start_time
        # np.savez('ind_' + uniforms[j] + '_nu'+ str(num_users) + 'd' + str(d) + 'm' + str(m) + 'L' + str(L) + '_'+str(seed), 
        #          seed, ind.rewards, ind.best_rewards, run_time)
        np.savez('ind_' + uniforms[j] + '-' + filename[0:2], 
                 seed, ind.rewards, ind.best_rewards, run_time)

        print("(model) Running SCLUB")
        sclub = SCLUB(nu = num_users, d = d, num_stages = num_stages)
        sstart_time = time.time()
        sclub.run(envir)
        run_time = time.time() - start_time
        # np.savez('sclub_' + uniforms[j] + '_nu'+ str(num_users) + 'd' + str(d) + 'm' + str(m) + 'L' + str(L) + '_'+str(seed), 
        #          seed, sclub.rewards, sclub.best_rewards, run_time, sclub.num_clusters)
        np.savez('sclub_' + uniforms[j] + '-' + filename[0:2], 
                 seed, sclub.rewards, sclub.best_rewards, run_time, sclub.num_clusters)

if __name__== "__main__":
    # synthetic experiment with user number is 10**3 and m=10 clusters
    main(num_stages = 20, num_users = 1000, d = 20, m = 10, L = 20, pj = [2])

    # main(num_stages = 20, num_users = 1000, d = 20, m = 10, L = 20, pj = [0,2], filename='ml_1000user_d20.npy')
    # main(num_stages = 20, num_users = 1000, d = 20, m = 10, L = 20, pj = [0,2], filename='yelp_1000user_d20.npy')
