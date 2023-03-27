The main entrance of this model is **main_ijcai19.py,** and it includes four models (CLUB, LinUCB, LinUCB-Ind and SCLUB (proposed by Jin T, et al)). Each model needs to take six hyperparameters as the input,

| parameters       | description                                             |
| ---------------- | ------------------------------------------------------- |
| ***num_stages*** | number of iterations (stages)                           |
| ***nu***         | number of users                                         |
| ***d***          | dimensionality of the feature vector for items          |
| ***m***          | number of weight vectors indicating underlying clusters |
| ***L***          | number of returned items in each round                  |
| ***pj***         | environment settings                                    |

There are three different settings. *Uniform* indicates that all users following a uniform distribution. *half*  represents that users in the same cluster follow a uniform distribution but the distribution among clusters is arbitrary. *arbitrary* represents that the distribution of all users is arbitrary.

1. pj = 0, only output result from *uniform* distribution,
2. pj = 1, output result from *uniform*  and *half* settings,
3. pj = 2, output result from *uniform*, *half* and *arbitrary* settings,