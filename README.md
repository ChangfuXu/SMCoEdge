This code is a implementation of our paper "Dynamic Parallel Multi-Server Selection and Allocation in Collaborative Edge Computing".
The function of our method mainly two stages:
1) Make the multi-ES selection decision by the proposed top-k DQN model in the first stage.
2) Make the workload allocation decision by the proposed Solving Equation (SE) or High Efficient Workload Allocation (HEWA) method in the second stage.

To run this code, please install packages: tensorflow 1.4.0., NumPy, and matplotlib.

This code consists of three files TopkDQN.py, environment.py, and main.py. In addtion, some default experiment results are stored in the results directory. However, user can run the main.py file to achieve these results again. Note that sometimes, the results may be some devivations. However, user can achieve the better results by running more times.

The main.py file is the main code. User should run this code to acheive the experimental results.

The environment.py inculdes the code for MEC environment. In this file, some environment parameters such as ESs' computing capacities, task size, and transmission rate can be ajusted by user.

The TopkDQN.py includes the code for top-k deep reinforcement learning model. 

Note that the parameters in current code is set by default. Hence, user can ajust the parameters such as deadline, task arrival probabiltiy, and the number of tasks to run the main.py to get more experimental results.

Paramters setting information: 1) Deadline setting. User can adjust the variable value of DEADLINE_TIME_SLOTS in main.py. For example, when DEADLINE_TIME_SLOTS is set to 6, 8, and 10, which reresents 0.6, 0.8, and 1.0 seconds. 2) Task arrival probabiltiy setting. User can adjust the variable value of TASK_ARRIVAL_PROB from 0.1 to 1 in main.py. 3) The Number Setting of Tasks. User can adjust the variable value of NUM_TASK from 10 to 100 in main.py. 4) Algorithm setting. In our method, we propose two algoritms SE and HEWA to solve the workload allocation decision. The SE algorithm achieves lower make-span than HEWA algorithm. The HEWA algorithm has higher efficiency than SE algorithm. User can adjust the variable value of ALGORITHM_TYPE with 'SE' or 'HEWA' to perform SE and HEWA algorithms in main.py, respectively. 4) ESs' computing capacities setting. If user want to achieve the results varying different ESs' computing capacityes, user can adjust the variable value of BS_capacities in environment.py and then run the main.py to achieve the experimental results.
