import numpy as np


class OffloadEnvironment:
    def __init__(self, num_tasks, num_BSs, dim_action, num_time, max_time, arrival_prob, alg_type):
        # INPUT DATA
        self.n_tasks = num_tasks  # The number of mobile devices
        self.n_BSs = num_BSs  # The number of base station
        self.action_dim = dim_action  # The number k of multi-ES selection
        self.n_actions = self.n_BSs  # The action space
        self.n_time = num_time  # The number of time slot set
        self.duration = 0.1  # unit: second

        # Environment parameter setting
        self.BS_capacities = np.array([10, 20, 30, 40, 50])  # GHz or Gigacycles/s
        # self.BS_capacities = np.array([30, 35, 40, 45, 50])  # GHz or Gigacycles/s        #
        self.tran_rates = np.array([500, 425, 450, 400, 475])  ## Transmission rate ranges from 400 t0 500 Mbits/s.
        self.comp_density = 30 / 1000000 * np.random.uniform(100, 300, size=[self.n_tasks])  # Gigacycles/Mbit
        self.deadline_delay = max_time * self.duration  # in second

        # Task size setting
        self.max_bit = 40  # Mbits
        self.min_bit = 10  # Mbits
        self.task_arrive_prob = arrival_prob  # The arrival probability of task
        self.arrival_bits = np.zeros([self.n_time, self.n_BSs, self.n_tasks])  # store task for each time slot and BS

        # The number of input features
        self.n_features = 1 + self.n_BSs  # STATE: [D_{n,t}, [Q_{1,t},..., Q_{B,t}]]
        # Initial an array to store the service delay for each task
        self.make_spans = np.zeros([self.n_time, self.n_BSs, self.n_tasks])
        # Initial the workload length of all ESs
        self.queue_len = np.zeros([self.n_time, self.n_BSs])
        # Initialize the workload lengths of processing queue before current arrival task in all ESs
        self.proc_queue_bef = np.zeros([self.n_time, self.n_BSs])
        # Initialize the transmission delay before the current arrival task in all ESs
        self.tran_delay_bef = np.zeros([self.n_time, self.n_BSs])
        # Initial the task offloading results. 0: succeed. 1: failure.
        self.is_fail_tasks = np.zeros([self.n_time, self.n_BSs, self.n_tasks])  # failure indicator
        # The control parameter of using Solving Equation (SE) or High Efficient Workload Allocation procedure
        self.algorithm_type = alg_type

    # Initialize the system environment
    def initialize_env(self, arrival_bits_):
        self.arrival_bits = arrival_bits_
        # Initial an array to store the service delay for each task
        self.make_spans = np.zeros([self.n_time, self.n_BSs, self.n_tasks])
        # Initial the workload length of processing queues in all ESs
        self.queue_len = np.zeros([self.n_time, self.n_BSs])
        # Initialize the workload lengths of processing queue before the current arrival task in all ESs
        self.proc_queue_bef = np.zeros([self.n_time, self.n_BSs])
        # Initialize the transmission delay before the current arrival task in all ESs
        self.tran_delay_bef = np.zeros([self.n_time, self.n_BSs])
        # Initial the task offloading results. 0: succeed. 1: failure.
        self.is_fail_tasks = np.zeros([self.n_time, self.n_BSs, self.n_tasks])  # failure indicator

    # Perform task offloading to achieve:
    # (1) Service delays;
    # (2) Fail results;
    # (3) Next state;
    # (4) Next queue workload.
    def perform_offloading(self, t, b, n, a_set):
        allocation_fractions = np.zeros(self.n_BSs)
        tran_comp_delays = np.zeros(self.n_BSs)
        wait_delays = np.zeros(self.n_BSs)
        n_bit = self.arrival_bits[t][b][n]  # Get the n_index's bit data
        n_a_set = a_set.astype(int)  # get the action of task n and convert it into integer
        if self.action_dim == 1:
            wait_delays[n_a_set] = np.min([self.tran_delay_bef[t][b] + (self.queue_len[t][n_a_set] + self.proc_queue_bef[t][n_a_set]) / self.BS_capacities[n_a_set], self.deadline_delay])
            self.tran_delay_bef[t][b] = np.min([self.tran_delay_bef[t][b] + n_bit / self.tran_rates[n_a_set], self.deadline_delay/2])
            if n_a_set == b:
                tran_comp_delays[n_a_set] = n_bit * self.comp_density[n] / self.BS_capacities[n_a_set]
            else:
                tran_comp_delays[n_a_set] = n_bit / self.tran_rates[n_a_set] + n_bit * self.comp_density[n] / self.BS_capacities[n_a_set]
            allocation_fractions[n_a_set] = 1
        else:
            # Calculate the workload allocation fractions of the selected ESs
            for i in range(self.action_dim):
                # wait_delays[n_a_set[i]] = np.min([self.tran_delay_bef[t][b] + (self.queue_len[t][n_a_set[i]] + self.proc_queue_bef[t][
                #         n_a_set[i]]) / self.BS_capacities[n_a_set[i]], self.deadline_delay])
                if n_a_set[i] == b:
                    if self.action_dim == 2:
                        wait_delays[n_a_set[i]] = np.min([self.tran_delay_bef[t][b] + (self.queue_len[t][n_a_set[i]] + self.proc_queue_bef[t][
                            n_a_set[i]]) / self.BS_capacities[n_a_set[i]], self.deadline_delay])
                        tran_comp_delays[n_a_set[i]] = n_bit / self.tran_rates[n_a_set[i]] + n_bit * self.comp_density[n] / self.BS_capacities[n_a_set[i]]
                    else:
                        wait_delays[n_a_set[i]] = np.min([(self.queue_len[t][n_a_set[i]] + self.proc_queue_bef[t][
                            n_a_set[i]]) / self.BS_capacities[n_a_set[i]], self.deadline_delay])
                        tran_comp_delays[n_a_set[i]] = n_bit * self.comp_density[n] / self.BS_capacities[n_a_set[i]]
                else:
                    wait_delays[n_a_set[i]] = np.min([self.tran_delay_bef[t][b] + (self.queue_len[t][n_a_set[i]] + self.proc_queue_bef[t][
                            n_a_set[i]]) / self.BS_capacities[n_a_set[i]], self.deadline_delay])
                    tran_comp_delays[n_a_set[i]] = n_bit / self.tran_rates[n_a_set[i]] + n_bit * self.comp_density[n] / self.BS_capacities[n_a_set[i]]

            # Achieve the workload allocation fractions of selected ESs by solving equation (SE) method
            # or high efficient workload allocation (HEWA) method
            if self.algorithm_type == 'SE':  # Using SE method
                eq_A = np.zeros([self.action_dim, self.action_dim])
                eq_b = np.ones([self.action_dim])
                for i in range(self.action_dim - 1):
                    eq_A[i][i] = tran_comp_delays[n_a_set[i]]
                    eq_A[i][i + 1] = -tran_comp_delays[n_a_set[i + 1]]
                    eq_b[i] = wait_delays[n_a_set[i + 1]] - wait_delays[n_a_set[i]]

                eq_A[self.action_dim - 1] = np.ones([self.action_dim])
                # Achieve the allocation fractions by Equations Solving. The time complexity is O(n^3)
                allocation_fractions[n_a_set] = np.linalg.solve(eq_A, eq_b)
                # Here, we should ensure that all the allocation fractions are greater than 0 by the following operation
                allocation_fractions[allocation_fractions < 0] = 0
                # Normalized the allocation fractions
                allocation_fractions[n_a_set] = allocation_fractions[n_a_set] / np.sum(allocation_fractions[n_a_set])
            else:  # Using HEWA method
                for i in range(self.action_dim):
                    except_i_indexes = np.delete(n_a_set, i)  # Get the ES indexes except the current ES index
                    allocation_fractions[n_a_set[i]] = np.prod(tran_comp_delays[except_i_indexes])
                allocation_fractions[n_a_set] = allocation_fractions[n_a_set] / np.sum(allocation_fractions[n_a_set])

            # Update the transmission delay at the BS b before current arrival task
            # Note that when BS b is selected to process the task, the transmission delay does not need to update
            # since the local transmission rate is very quick in real system.
            for j in range(self.action_dim):
                if n_a_set[j] != b:
                    self.tran_delay_bef[t][b] = self.tran_delay_bef[t][b] + allocation_fractions[n_a_set[j]] * n_bit / self.tran_rates[n_a_set[j]]
            # self.tran_delay_bef[t][b] = np.min([self.tran_delay_bef[t][b], self.duration])

        print("Output allocation decision x_" + str(b) + "," + str(n) + "," + str(t) + "=", allocation_fractions)

        # Update the processing queue workload length at the selected ES before current arrival task
        self.proc_queue_bef[t][n_a_set] = self.proc_queue_bef[t][n_a_set] + allocation_fractions[n_a_set] * n_bit * self.comp_density[n]

        # Calculate the service delay of task n, which equals to the longest service delay of subtasks
        self.make_spans[t][b][n] = np.max(allocation_fractions[n_a_set] * tran_comp_delays[n_a_set] + wait_delays[n_a_set])

        # Record the fail task and Calculate the reward of the action of task n offloading
        if self.make_spans[t][b][n] > self.deadline_delay:
            self.is_fail_tasks[t][b][n] = 1  # Mark the failure task
            self.make_spans[t][b][n] = self.deadline_delay  # Reset make-span with deadline
            reward = -2 * self.deadline_delay  # Set the punishment
        else:
            reward = -self.make_spans[t][b][n]  # Set the reward
        return reward

    # Update the processing queue length of all ESs at the beginning of next time slot.
    def update_queues(self, t):
        for b_ in range(self.n_BSs):
            self.queue_len[t + 1][b_] = np.max(
                [self.queue_len[t][b_] + self.proc_queue_bef[t][b_] - self.BS_capacities[b_] * self.duration, 0])
