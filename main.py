################################################
# Dynamic Parallel Multi-Server Selection and Allocation in Collaborative Edge Computing, including:
# 1) Make the multi-ES selection decision by the proposed top-k DQN model
# 2) Make the workload allocation decision by the proposed SE or HEWA algorithm
# Author: Changfu Xu
#################################################
from environment import OffloadEnvironment
from TopkDQN import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt


def DRL_SMO_algorithm(DQN_list_, NUM_EPISODE_):
    action_step = 0
    training_step = 0
    for episode_ in range(NUM_EPISODE_):
        print('Episode: %d' % episode_)
        # print('Greedy probability: %.10f' % DQN_list_[0].epsilon)  # Print the epsilon-greedy value

        # BITRATE ARRIVAL
        arrival_tasks = np.random.uniform(env.min_bit, env.max_bit, size=[env.n_time, env.n_BSs, env.n_tasks])
        arrival_tasks = arrival_tasks * (
                    np.random.uniform(0, 1, size=[env.n_time, env.n_BSs, env.n_tasks]) < env.task_arrive_prob)

        # Initialize the system environment
        env.initialize_env(arrival_tasks)
        # Initialize an array to storage the system state at next time slot t
        state_bnt = np.zeros([env.n_BSs, env.n_tasks, env.n_features])  # State: [D_{n,t},Q_{1,t},..., Q_{B,t}]

        # ========================================= DRL ===================================================
        # -------- Train DQN model with reinforcement learning ----------------
        # for time_index in range(0, NUM_TIME_):  # Perform action in the set (T) of time slots
        for t in range(env.n_time - 1):
            for b_index in range(env.n_BSs):
                for n_index in range(env.n_tasks):
                    state_bnt[b_index][n_index] = np.hstack([env.arrival_bits[t][b_index][n_index], env.queue_len[t]])
                    if state_bnt[b_index][n_index][0] != 0:
                        # Generate the multi-ES selection
                        all_actions[b_index][n_index] = DQN_list_[b_index].choose_action(state_bnt[b_index][n_index])
                        # Perform task offloading and achieve its reward
                        reward[b_index][n_index] = env.perform_offloading(t, b_index, n_index, all_actions[b_index][n_index])

            env.update_queues(t)
            if t > 0:  # Since the initial queue length is set to 0, the samples at time slot are dropped
                for b_index in range(env.n_BSs):
                    for n_index in range(env.n_tasks):
                        if state_bnt[b_index][n_index][0] != 0:
                            if n_index == env.n_tasks - 1:
                                next_state_bnt = np.hstack([env.arrival_bits[t + 1][b_index][0], env.queue_len[t + 1]])

                            else:
                                next_state_bnt = np.hstack([env.arrival_bits[t][b_index][n_index + 1], env.queue_len[t]])
                            # Store transition tuple
                            DQN_list_[b_index].store_transition(state_bnt[b_index][n_index], all_actions[b_index][n_index],
                                                                reward[b_index][n_index], next_state_bnt)
            # ADD STEP (one step does not mean one store)
            action_step += 1
            # Set learning start time as bigger than 200 and frequency with each 10 steps
            if (action_step > 200) and (action_step % 10 == 0):

                for b_index in range(env.n_BSs):
                    DQN_list_[b_index].learn()  # Perform training
                # training_step += 1
                # print('Training step: %d' % training_step)

        make_spans = env.make_spans.flatten()
        aver_make_spans[episode_] = np.mean(make_spans[make_spans > 0])
        all_num_tasks = np.size(make_spans[make_spans > 0])
        aver_failure_rates[episode_] = np.sum(env.is_fail_tasks) / all_num_tasks
        #  ======================================== DRL END=================================================


if __name__ == "__main__":
    # Default parameter setting
    NUM_BSs = 5  # The number of base stations （BSs）
    NUM_TASK = 50  # The max number of task in each BS
    NUM_K = 3  # The number of multi-ES selection, which is set by user
    NUM_EPISODE = 200  # The number of episode
    NUM_TIME_SLOTS = 101  # The total number of time slots for each episode
    DEADLINE_TIME_SLOTS = 10  # The time slots of deadline
    TASK_ARRIVAL_PROB = 0.3  # The task arrival probability
    ALGORITHM_TYPE = 'SE'  # Set the algorithm type (i.e., SE or HEWA) of workload allocation. SE presents the Solving Equation (SE) method and HEWA for High Efficient Workload Allocation (HEWA) method

    # Initial variables of actions, rewards, and delays for experimental testing
    # Record the actions of all tasks of a BS at a time slot
    all_actions = np.zeros([NUM_BSs, NUM_TASK, NUM_K])
    aver_make_spans = np.zeros([NUM_EPISODE])
    aver_failure_rates = np.zeros([NUM_EPISODE])
    reward = np.zeros([NUM_BSs, NUM_TASK])

    # Generate offloading environment
    env = OffloadEnvironment(NUM_TASK, NUM_BSs, NUM_K, NUM_TIME_SLOTS, DEADLINE_TIME_SLOTS, TASK_ARRIVAL_PROB, ALGORITHM_TYPE)

    # Distributed DQN: For each BS to generate agent's class for deep reinforcement learning
    DQN_list = list()
    for b in range(NUM_BSs):
        DQN_list.append(DeepQNetwork(env.n_actions, env.action_dim, env.n_features, env.n_time,  # GENERATE ENVIRONMENT
                                     learning_rate=0.001,
                                     reward_decay=0.9,  # discount factor
                                     e_greedy=0.99,
                                     e_greedy_increment=0.001,
                                     replace_target_iter=200,  # each 100 steps, update target net
                                     memory_size=1024,  # maximum of memory
                                     batch_size=32,  # batch size of samples
                                     N_L1=20)  # Number of neuron
                        )

    # Execute the online DRL_SMO algorithm
    DRL_SMO_algorithm(DQN_list, NUM_EPISODE)

    print('============ Training finished ==========')

    # Plot the average make-span varying episodes
    np.savetxt('results/Makespan_' + ALGORITHM_TYPE + '_k' + str(NUM_K) +'_f10_200.csv', aver_make_spans, delimiter=',', fmt='%.4f')
    plt.figure(1)
    plt.plot(aver_make_spans)
    plt.ylabel('Average make-span')
    plt.xlabel('Episode')
    plt.savefig('results/Makespan_' + ALGORITHM_TYPE + '_k' + str(NUM_K) +'_f10_200.png')

    # Plot the average failure rate varying time slot
    np.savetxt('results/Failure_' + str(ALGORITHM_TYPE) + '_k' + str(NUM_K) +'_f10_200.csv', aver_failure_rates, delimiter=',', fmt='%.4f')

