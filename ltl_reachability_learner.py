import copy
from typing import Set, Tuple, List, Optional, Any
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from mdp import MDP
from mdp_simulator import MDPSimulator
from tqdm import tqdm
import json
import os
import time


class LTLReachabilityLearner:
    """
    Main learner class that orchestrates PAC learning of LTL reachability.
    """
    
    def __init__(self, mdp_simulator: MDPSimulator, min_num_iterations=5, max_num_iterations=50, convergence_threshold=0.001, num_policy_accuracy_sims=1000):
        """
        Initialize the learner.
        
        Args:
            error_tolerance: Acceptable error tolerance (PAC parameter delta).
            p_min: Minimum transition probability bound (required by algorithm).
        """
        self.discovered_mdp = MDP()  # Partial MDP
        self.mdp_sim = mdp_simulator  # Nests the true MDP
        self.latest_collapsed_mdp = MDP()  # Last created collapsed MDP after MEC detection - the learned policy can be extracted from this MDP
        self.learning_history = []  # [(k, num_samples, error), ...]
        self.states_set_history = []  # [(k, num_samples, seen_states_set), ...]
        self.transitions_seen_history = []  # [(k, num_samples, transitions_seen), ...]
        self.policy_accuracy_history = []  # [(k, num_samples, policy_accuracy), ..x.]

        self.min_num_iterations = min_num_iterations
        self.max_num_iterations = max_num_iterations
        self.convergence_threshold = convergence_threshold
        self.num_policy_accuracy_sims = num_policy_accuracy_sims
    
    def _is_staying_state_action_pair(self, curr_mdp, state_set, state, action):
        """
        Check if a given state-action pair remains within a given set of states

        Args:
            discovered_mdp (MDP): The discovered Markov Decision Process.
            state_set (set): The set of states to check.
            state: The state of the state-action pair.
            action: The action of the state-action pair.
        
        Returns:
            bool: True if all observed transitions stay in state_set, False otherwise.
        """
        if (state, action) not in curr_mdp.sample_counts:
            return True  # No samples yet, assume staying, but this should not happen because we are inputting actions from a state's set of actions.
        if (state not in state_set or action not in curr_mdp.get_actions(state)):
            return False  # If the state itself is not in the set, it can't be staying

        for next_state, count in curr_mdp.sample_counts[(state, action)].items():
            if count > 0 and next_state not in state_set:
                return False
        return True

    def _get_staying_state_action_pairs(self, curr_mdp, state_set):
        """
        Identify state-action pairs that remain within a given set of states

        Args:
            discovered_mdp (MDP): The discovered Markov Decision Process.
            state_set (set): The set of states to check.
        
        Returns:
            set: A set of staying state-action pairs (tuples).
        """
        staying_state_action_pairs = set()
        for state in state_set:
            actions = curr_mdp.get_actions(state)
            for action in actions:
                if (self._is_staying_state_action_pair(curr_mdp, state_set, state, action)):
                    staying_state_action_pairs.add((state, action))

        return staying_state_action_pairs

    def _get_staying_mdp(self, curr_mdp, state_set):
        """
        Create a sub-MDP containing only transitions that stay within state_set

        Args:
            curr_mdp (MDP): The current Markov Decision Process.
            state_set (set): The set of states to consider.

        Returns:
            MDP: A new MDP object restricted to state_set.
        """
        staying_mdp = MDP()
        staying_mdp.states = set(state_set)
        staying_state_action_pairs = self._get_staying_state_action_pairs(curr_mdp, state_set)
        
        for state in state_set:
            staying_mdp.topology[state] = set()
        
        for (state, action) in staying_state_action_pairs:
            staying_mdp.add_action_to_state(state, action)
            staying_mdp.state_action_pairs.add((state, action))
            
            # Copy transition probabilities for staying transitions
            for next_state, count in curr_mdp.sample_counts[(state, action)].items():
                assert next_state in state_set, "staying_state_action_pairs should have only state-action pairs where all possible transitions end up in the state_set."
                staying_mdp.update_transition_count(state, action, next_state, count)
        
        return staying_mdp

    def _find_all_possible_end_components(self, curr_mdp, X):
        """
        Find all possible end components (ECs) within a given set of states

        Args:
            discovered_mdp (MDP): The discovered Markov Decision Process.
            X (set): The set of states to check for possible ECs.
        
        Returns:
            list: A list of sets, where each set represents a possible end component.
        """
        staying_mdp = self._get_staying_mdp(curr_mdp, X)
        sccs = self._find_strongly_connected_components(staying_mdp, X)
        
        # Stopping Condition 1: if there are no SCCs in X
        if len(sccs) == 0:
            return []
        
        # Stopping condition 2: if the current set is one SCC
        if len(sccs) == 1 and sccs[0] == X:
            return sccs
        
        end_component_candidates = []
        for scc in sccs:
            end_component_candidates.extend(self._find_all_possible_end_components(curr_mdp, scc))
        
        return end_component_candidates
    
    def _find_strongly_connected_components(self, mdp, state_set=None):
        """
        Find all strongly connected components (SCCs) in an MDP using Tarjan's algorithm
        
        Args:
            mdp (MDP): MDP object with topology and transition_probabilities.
            state_set (set): Optional subset of states to search within.
            
        Returns:
            List of SCCs (sets), where each SCC is a set of states.
        """
        if state_set is None:
            state_set = mdp.states
                
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []
        
        def get_successors(state):
            """Get all successor states from a given state (within state_set)."""
            successors = set()
            actions = mdp.topology.get(state, set())
            
            for action in actions:
                for next_state, count in mdp.sample_counts[(state, action)].items():
                    if next_state in state_set and count > 0:
                        successors.add(next_state)
            
            return successors
        
        def strongconnect(node):
            """Tarjan's recursive DFS function."""
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            on_stack[node] = True
            stack.append(node)
            
            for successor in get_successors(node):
                if successor not in index:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif on_stack.get(successor, False):
                    lowlinks[node] = min(lowlinks[node], index[successor])
            
            if lowlinks[node] == index[node]:
                scc = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.add(w)
                    if w == node:
                        break
                
                # Only include non-trivial SCCs
                if len(scc) > 1:
                    sccs.append(scc)
                elif len(scc) == 1:
                    # Single node SCC only if it has a self-loop
                    single_node = list(scc)[0]
                    if single_node in get_successors(single_node):
                        sccs.append(scc)
                    
        
        for state in state_set:
            if state not in index:
                strongconnect(state)
        
        return sccs
    
    def _end_component_detection_num_required_samples(self, transition_error_tolerance, p_min):
        """
        Calculate the number of required samples to detect an end component (EC)
        within a given transition probability error tolerance for a minimum 
        transition probability.

        Args:
            transition_error_tolerance (float): The acceptable error tolerance level for transition probabilities (between 0 and 1).
            p_min (float): The minimum transition probability (between 0 and 1).
        
        Returns:
            int: The calculated number of required samples to detect an EC.
        """
        assert 0 < transition_error_tolerance < 1, "Transition error tolerance must be between 0 and 1."
        assert 0 < p_min < 1, "Minimum transition probability (p_min) must be between 0 and 1."

        num_required_samples = int(np.log(transition_error_tolerance) / (np.log(1 - p_min + 1e-100) + 1e-100)) + 1
        return num_required_samples
    
    def _confidently_detect_end_component(self, curr_mdp, ec_candidate, transition_error_tolerance, p_min):
        """
        Detect whether a given candidate set of states forms an end component (EC)

        Args:
            curr_mdp (MDP): The discovered Markov Decision Process.
            ec_candidate (set): The candidate set of states to check for EC.
            transition_error_tolerance (float): The acceptable error tolerance (between 0 and 1).
            p_min (float): The minimum transition probability (between 0 and 1).

        Returns:
            bool: True if the candidate set forms an EC, False otherwise.
        """
        assert isinstance(ec_candidate, set), "EC candidate must be a set."
        assert 0 < transition_error_tolerance < 1, "Transition error tolerance must be between 0 and 1."
        assert 0 < p_min < 1, "Minimum transition probability must be between 0 and 1."

        min_num_samples = self._end_component_detection_num_required_samples(transition_error_tolerance, p_min)
        staying_state_action_pairs = self._get_staying_state_action_pairs(curr_mdp, ec_candidate)
        
        # Check if all staying pairs have enough samples. If not, gather more samples.
        for (state, action) in tqdm(list(staying_state_action_pairs),
                        desc="Gathering enough samples for EC detection",
                        unit="sa-pair"):
            total_samples = curr_mdp.get_sample_count(state, action)
            if total_samples < min_num_samples:
                num_needed_samples = int(min_num_samples - total_samples + 1)
                for _ in range(num_needed_samples):
                    next_state, reward = self.mdp_sim.step(state, action)
                    if (next_state not in self.discovered_mdp.states):
                        self.add_gt_state_and_actions_to_mdp(self.discovered_mdp, next_state, is_goal=(reward == 1))
                    curr_mdp.record_sample(state, action, next_state)
        
        new_staying_state_action_pairs = self._get_staying_state_action_pairs(curr_mdp, ec_candidate)
        states_with_staying_actions = set()
        for (state, action) in new_staying_state_action_pairs:
            states_with_staying_actions.add(state)
        if states_with_staying_actions != ec_candidate:
            return False

        return True
    
    def is_looping(self, current_state, transition_error_tolerance, p_min):
        """
        Detect whether a loop can be formed in the sequence of visited states

        Args:
            current_state: The current state to check.
            transition_error_tolerance (float): The acceptable error tolerance.
            p_min (float): The minimum transition probability.
        
        Returns:
            bool: True if a loop is detected, False otherwise.
        """
        visited_states = self.discovered_mdp.states
        if current_state not in visited_states:
            return False
        
        end_component_candidates = self._find_all_possible_end_components(self.discovered_mdp, visited_states)
        
        for ec in end_component_candidates:
            if (current_state in ec and 
                self._confidently_detect_end_component(self.discovered_mdp, ec, transition_error_tolerance, p_min)):
                return True
        
        return False

    def is_looping_state_set_change(self, state_set_size_history):
        """
        Detect if the simulation is stuck in a loop.
        If the state set of the discovered MDP has not changed
        for the past |S|^2 steps, we are in a loop.

        Args:
            state_set_size_history (list): List of previously seen state set sizes.
        
        Returns:
            bool: True if a loop is detected, False otherwise.
        """
        if (len(state_set_size_history) == 0 or len(state_set_size_history) <= state_set_size_history[-1]**2):
            return False
        curr_num_states = state_set_size_history[-1]
        # We only need to check if the current state size equals that of the |S|^2 previous iteration. We only add states (never remove).
        is_looping = state_set_size_history[-(curr_num_states**2)] == curr_num_states
        return is_looping

    def is_looping_seen_state(self, visited_states_counter):
        """
        Detect if the simulation is stuck in a loop.
        If the state set of the discovered MDP has not changed
        for the past |S|^2 steps, we are in a loop.

        Args:
            visited_states_counter (Counter): Counter of previously seen states.
        
        Returns:
            bool: True if a loop is detected, False otherwise.
        """
        curr_num_states = len(visited_states_counter.keys())
        is_looping = visited_states_counter[curr_num_states] > curr_num_states
        return is_looping

    def find_all_MECs(self, transition_error_tolerance, p_min):
        """
        Detect whether a loop can be formed in the sequence of visited states

        Args:
            transition_error_tolerance (float): The acceptable error tolerance.
            p_min (float): The minimum transition probability.
        
        Returns:
            bool: True if a loop is detected, False otherwise.
        """
        
        mec_candidates = self._find_all_possible_end_components(self.discovered_mdp, self.discovered_mdp.states)

        mecs = []
        for mec in mec_candidates:
            if (self._confidently_detect_end_component(self.discovered_mdp, mec, transition_error_tolerance, p_min)):
                mecs.append((mec, self._get_staying_state_action_pairs(self.discovered_mdp, mec)))
        
        return mecs
    
    def _calculate_transition_probability_error_tolerance(self, total_error_tolerance, p_min, num_seen_sa_pairs):
        """
        Calculate the acceptable error tolerance for individual transition probabilities.

        Args:
            total_error_tolerance (float): The overall acceptable error tolerance level (between 0 and 1).
            p_min (float): The minimum transition probability (between 0 and 1).
            num_seen_sa_pairs (int): The number of all seen state-action pairs in the MDP.
        
        Returns:
            float: The calculated acceptable error tolerance for individual transition probabilities.
        """
        assert 0 < total_error_tolerance < 1, "Total error tolerance must be between 0 and 1."
        assert 0 < p_min < 1, "Minimum transition probability (p_min) must be between 0 and 1."
        assert 0 <= num_seen_sa_pairs, "Number of state-action pairs must be a positive integer."

        if (num_seen_sa_pairs == 0):
            error_tolerance = total_error_tolerance * p_min
        else:
            error_tolerance = (total_error_tolerance * p_min) / num_seen_sa_pairs
        return error_tolerance
    
    def calculate_policy_accuracy(self, mdp, max_steps=100):
        """
        Calculate the policy accuracy of the discovered MDP.

        Returns:
            float: The calculated policy accuracy.
        """
        n = self.num_policy_accuracy_sims
        successful_runs = 0
        for _ in tqdm(range(0, n), desc="Policy accuracy sims", unit="sim"):
        # for _ in range(0, n):
            # start from the ground-truth initial state
            curr_state = self.mdp_sim.gt_mdp.initial_state

            # for _ in tqdm(range(0, max_steps), desc="Policy accuracy sim steps", unit="step"):
            for _ in range(0, max_steps):
                if curr_state in self.mdp_sim.gt_mdp.goal_states:
                    successful_runs += 1
                    break
                
                in_super_state = (curr_state in mdp.state_MEC)
                if (in_super_state):
                    policy_state = mdp.state_MEC[curr_state]
                else:
                    policy_state = curr_state

                if (policy_state not in mdp.states):
                    break

                action = mdp.learned_policy.get(policy_state, None)
                if action is None:
                    break

                # If action comes from an MEC it will be in the form (s, a)
                if in_super_state:
                    mec_s, mec_a = action
                    if curr_state == mec_s:
                        chosen_action = mec_a
                    else:
                        # choose a random action available in the ground-truth MDP for the current state
                        # available = list(self.mdp_sim.gt_mdp.topology.get(curr_state, []))
                        available = [a for a in list(self.mdp_sim.gt_mdp.topology.get(curr_state, [])) if (curr_state, a) in mdp.MEC_state_action_pairs[policy_state]]
                        if not available:
                            break
                        chosen_action = np.random.choice(available)
                else:
                    chosen_action = action

                next_state, _ = self.mdp_sim.step(curr_state, chosen_action)
                curr_state = next_state

        return successful_runs / n
    
    def add_gt_state_and_actions_to_mdp(self, mdp, s, is_goal):
        assert s in self.mdp_sim.gt_mdp.states, f"The state must be in the MDP simulator's states. Got state: {s}. \n\nMDP Simulator's states: {self.mdp_sim.gt_mdp.states}"
        mdp.add_state(s, is_goal=is_goal)
        for a in self.mdp_sim.gt_mdp.topology[s]:
            mdp.add_action_to_state(s, a)

    def simulate(self, initial_state, confidence_error_k, p_k):
        """
        Simulating the environment to collect transition counts
        to explore black-box MDP to run BVI. Uses and updates 
        the global discovered_mdp object.

        Args:
            initial_state: Beginning state to start the simulation.
            confidence_error_k (float): Confidence error for the current iteration.
            p_k (float): Minimum transition probability for the current iteration.
        
        """
        for a in self.mdp_sim.gt_mdp.topology[initial_state]:
            self.discovered_mdp.add_action_to_state(initial_state, a)

        curr_state = initial_state     
        
        def reached_goal(state):
            if (state in self.mdp_sim.gt_mdp.goal_states):
                self.discovered_mdp.add_state(state, is_goal=True)
                return True
            else:
                return False

        transition_error_tolerance = self._calculate_transition_probability_error_tolerance(confidence_error_k, p_k, len(self.discovered_mdp.state_action_pairs))  # Used for delta_t sure EC detection
        state_set_size_history = []
        visited_states = set()
        visited_states.add(curr_state)
        visited_states_counter = Counter()  # Used for seen state looping detection

        # while (not reached_goal(curr_state) and not self.is_looping(curr_state, transition_error_tolerance, p_k)):
        while (not reached_goal(curr_state) and not self.is_looping_state_set_change(state_set_size_history)):
        # while (not reached_goal(curr_state) and not self.is_looping_seen_state(visited_states_counter)):
            if (curr_state not in self.discovered_mdp.states):
                self.add_gt_state_and_actions_to_mdp(self.discovered_mdp, curr_state, is_goal=(curr_state in self.mdp_sim.gt_mdp.goal_states))
            curr_action = self.discovered_mdp.sample_best_action_from_state(curr_state)
            self.discovered_mdp.add_action_to_state(curr_state, curr_action) # NOTE: should have already been added
            curr_next_state, reward = self.mdp_sim.step(curr_state, curr_action)            
            
            # NOTE: we assume we have an oracle that will tell us all the valid actions from a state
            if (curr_next_state not in self.discovered_mdp.states):
                self.add_gt_state_and_actions_to_mdp(self.discovered_mdp, curr_next_state, is_goal=(reward == 1))

            self.discovered_mdp.record_sample(curr_state, curr_action, curr_next_state)
            curr_state = curr_next_state
            visited_states.add(curr_state)
            state_set_size_history.append(len(visited_states))
            # visited_states_counter[curr_state] += 1
    
    def learn(self, analysis_dir=""):
        """
        Main learning loop for PAC learning of LTL reachability.
        Iteratively simulates, updates, and runs BVI on the partial
        discovered MDP until convergence. Confidence error and minimum
        transition probability are halved each iteration.

        Args:
            analysis_dir (str): Directory to save analysis plots and data.
        """

        self.learning_history = []
        prev_collapsed_mdp_MEC_states = dict()  # {super_state: set of states in MEC}
        error = 1
        confidence_error = 1 ### TODO: will update this later in the final algorithm.
        p_min = 1  # NOTE: Optimization? Update p_min by either dividing by 2 or updating it to the lowest seen transition probability?
        k = 0
        self.discovered_mdp.confidence_error = confidence_error
        self.discovered_mdp.p_min = p_min

        def calculate_N_k(k):
            # NOTE: This is a heuristic for N_k. It is not does not provide the same guarantees as the N_k in the paper.
            # return 1000  # constant - experimentation: plot learning curve to figure out the best way to calculate N_k
            return (len(self.discovered_mdp.states) + 1) * (k**2) * 10

        while True:
            
            # Accept control c interrupt and safely exit while saving progress
            try:
                # Update Hyperparameters
                k += 1
                print("Iteration:", k)
                error_k = error / (2.0 ** k)  # This value is not actually used
                confidence_error_k = confidence_error / (2.0 ** k)
                p_k = p_min / (2.0 ** k)

                self.discovered_mdp.confidence_error = confidence_error_k
                self.discovered_mdp.p_min = p_k
                self.discovered_mdp.set_initial_state(self.mdp_sim.gt_mdp.initial_state)

                N_k = calculate_N_k(k)

                ### Simulation Phase
                for _ in tqdm(range(0, N_k), desc=f"Sim k={k}", unit="sim"):
                    self.simulate(self.discovered_mdp.initial_state, confidence_error_k, p_k)
                
                total_sample_counts = sum(
                        sum(next_counts.values()) for next_counts in self.discovered_mdp.sample_counts.values()
                    )

                if (analysis_dir != ""):
                    print("Num Sample Counts:", total_sample_counts)
                    print("Explored states:", len(self.discovered_mdp.states), "/", len(self.mdp_sim.gt_mdp.states))

                    self.states_set_history.append((k, total_sample_counts, len(self.discovered_mdp.states)))
                    self.transitions_seen_history.append((k, total_sample_counts, sum(len(sat_counts.values()) for sa, sat_counts in self.discovered_mdp.sample_counts.items())))


                ### BVI Update Phase
                delta_c = self._calculate_transition_probability_error_tolerance(confidence_error_k, p_k, len(self.discovered_mdp.state_action_pairs))
                collapsed_discovered_mdp = copy.deepcopy(self.discovered_mdp)
                collapsed_discovered_mdp.collapse(self.find_all_MECs(delta_c, p_k))
                collapsed_discovered_mdp.initialize_mdp_value_bounds()

                bvi_history = []
                print("Explored Collapsed states:", len(collapsed_discovered_mdp.states))

                collapsed_discovered_mdp.update_transition_probabilities()
                for _ in tqdm(range(self.num_bvi_iterations(len(collapsed_discovered_mdp.states), k)),
                              desc=f"BVI k={k}", unit="it"):
                    collapsed_discovered_mdp.bvi_update()
                    bvi_history.append([k, confidence_error_k, p_k, collapsed_discovered_mdp.get_mdp_error()])
                self.plot_bvi_history(bvi_history, analysis_dir)

                collapsed_discovered_mdp.update_learned_policy()

                # The learned policy can be extracted from the latest collapsed MDP
                self.latest_collapsed_mdp = collapsed_discovered_mdp

                curr_error = collapsed_discovered_mdp.get_mdp_error()
                lower_bound = collapsed_discovered_mdp.s_value_bounds[collapsed_discovered_mdp.initial_state][0]
                upper_bound = collapsed_discovered_mdp.s_value_bounds[collapsed_discovered_mdp.initial_state][1]
                self.learning_history.append([k, total_sample_counts, confidence_error_k, p_k, curr_error, lower_bound, upper_bound])

                if (analysis_dir != ""):
                    # self.policy_accuracy_history.append((k, total_sample_counts, confidence_error_k, p_k, self.calculate_policy_accuracy(collapsed_discovered_mdp, max_steps=int(len(self.mdp_sim.gt_mdp.states)**2 / p_k))))  # TODO: use the number of gt MDP states or discovered MDP states?
                    self.policy_accuracy_history.append(
                        (k, 
                         total_sample_counts, 
                         confidence_error_k, 
                         p_k, 
                         self.calculate_policy_accuracy(
                             collapsed_discovered_mdp, 
                             max_steps=int(len(self.discovered_mdp.states)**2 / p_k)
                            )
                        )
                    )
                
                if (analysis_dir != ""):
                        self.run_analysis(analysis_dir)

                print("Error:", self.learning_history[-1])
                
                if (k > 1 and self.has_converged(collapsed_discovered_mdp, self.learning_history[-2], prev_collapsed_mdp_MEC_states)):  # NOTE: -2 because we want the one before the current iteration (current iteration is -1 index).
                    print("Algorithm has converged, exiting...")
                    # if (analysis_dir != ""):
                    #     self.run_analysis(analysis_dir)
                    break

                prev_collapsed_mdp_MEC_states = collapsed_discovered_mdp.MEC_states.copy()
            except KeyboardInterrupt:
                # if (analysis_dir != ""):
                #     self.run_analysis(analysis_dir)
                break
            

    def plot_error_history(self, analysis_dir, log_scale=False):
        # plots the learning error history for each iteration and saves it to error_history_plot_path
        error_vs_k_plot_path = os.path.join(analysis_dir, "error_vs_k.png")
        os.makedirs(analysis_dir, exist_ok=True)
        plt.figure()
        if log_scale:
            plt.semilogy(list(np.array(self.learning_history)[:, -3]), marker='o')
            plt.ylim(top=2)
        else:
            plt.plot(list(np.array(self.learning_history)[:, -3]), marker='o')
            plt.ylim(-0.1, 1.1)
        plt.xlabel("Iteration")
        plt.ylabel("Error (U - L)")
        plt.title("Learning Error History")
        plt.grid()
        plt.savefig(error_vs_k_plot_path)
        plt.close()

        error_vs_samples_plot_path = os.path.join(analysis_dir, "error_vs_samples.png")
        plt.figure()
        if log_scale:
            plt.semilogy(list(np.array(self.learning_history)[:, 1]), list(np.array(self.learning_history)[:, -3]))
            plt.ylim(top=2)
        else:
            plt.plot(list(np.array(self.learning_history)[:, 1]), list(np.array(self.learning_history)[:, -3]))
            plt.ylim(-0.1, 1.1)
        plt.xlabel("Number of Samples")
        plt.ylabel("Error (U - L)")
        plt.title("Learning Error vs Number of Samples")
        plt.grid()
        plt.savefig(error_vs_samples_plot_path)
        plt.close()

    def plot_states_set_history(self, analysis_dir, log_scale=False):
        # plots the learning error history for each iteration and saves it to error_history_plot_path
        num_states_seen_vs_k_plot_path = os.path.join(analysis_dir, "num_states_seen_vs_k.png")
        max_states = len(self.mdp_sim.gt_mdp.states)
        os.makedirs(analysis_dir, exist_ok=True)
        plt.figure()
        if log_scale:
            plt.semilogy(list(np.array(self.states_set_history)[:, -1]), marker='o')
            plt.ylim(top=max_states * 2)
        else:
            plt.plot(list(np.array(self.states_set_history)[:, -1]), marker='o')
            plt.ylim(-0.1 * max_states, max_states * 1.1)
        plt.xlabel("Iteration")
        plt.ylabel("Num Seen States")
        plt.title("Num Seen States History")
        plt.grid()
        plt.savefig(num_states_seen_vs_k_plot_path)
        plt.close()

        num_states_seen_vs_samples_plot_path = os.path.join(analysis_dir, "num_states_seen_vs_samples.png")
        plt.figure()
        if log_scale:
            plt.semilogy(list(np.array(self.states_set_history)[:, 1]), list(np.array(self.states_set_history)[:, -1]))
            plt.ylim(top=max_states * 2)
        else:
            plt.plot(list(np.array(self.states_set_history)[:, 1]), list(np.array(self.states_set_history)[:, -1]))
            plt.ylim(-0.1 * max_states, max_states * 1.1)
        plt.xlabel("Number of Samples")
        plt.ylabel("Num Seen States")
        plt.title("Num Seen States vs Number of Samples")
        plt.grid()
        plt.savefig(num_states_seen_vs_samples_plot_path)
        plt.close()

    def plot_transitions_seen_history(self, analysis_dir, log_scale=False):
        # plots the learning error history for each iteration and saves it to error_history_plot_path
        num_transitions_seen_vs_k_plot_path = os.path.join(analysis_dir, "num_transitions_seen_vs_k.png")
        max_transitions = sum(len(sat_counts) for sat_counts in self.mdp_sim.gt_mdp.transition_probabilities.values())
        os.makedirs(analysis_dir, exist_ok=True)
        plt.figure()
        if log_scale:
            plt.semilogy(list(np.array(self.transitions_seen_history)[:, -1]), marker='o')
            plt.ylim(top=max_transitions * 2)
        else:
            plt.plot(list(np.array(self.transitions_seen_history)[:, -1]), marker='o')
            plt.ylim(-0.1 * max_transitions, max_transitions * 1.1)
        plt.xlabel("Iteration")
        plt.ylabel("Num Seen Transitions")
        plt.title("Num Seen Transitions History")
        plt.grid()
        plt.savefig(num_transitions_seen_vs_k_plot_path)
        plt.close()

        num_transitions_seen_vs_samples_plot_path = os.path.join(analysis_dir, "num_transitions_seen_vs_samples.png")
        plt.figure()
        if log_scale:
            plt.semilogy(list(np.array(self.transitions_seen_history)[:, 1]), list(np.array(self.transitions_seen_history)[:, -1]))
            plt.ylim(top=max_transitions * 2)
        else:
            plt.plot(list(np.array(self.transitions_seen_history)[:, 1]), list(np.array(self.transitions_seen_history)[:, -1]))
            plt.ylim(-0.1 * max_transitions, max_transitions * 1.1)
        plt.xlabel("Number of Samples")
        plt.ylabel("Num Seen Transitions")
        plt.title("Num Seen Transitions vs Number of Samples")
        plt.grid()
        plt.savefig(num_transitions_seen_vs_samples_plot_path)
        plt.close()
    
    def plot_policy_accuracy_history(self, analysis_dir, log_scale=False):
        # plots the learning error history for each iteration and saves it to error_history_plot_path
        policy_accuracy_vs_k_plot_path = os.path.join(analysis_dir, "policy_accuracy_vs_k.png")
        os.makedirs(analysis_dir, exist_ok=True)
        plt.figure()
        if log_scale:
            plt.semilogx(list(np.array(self.policy_accuracy_history)[:, 0]), list(np.array(self.policy_accuracy_history)[:, -1]), marker='o')
            plt.xlabel("Iteration (log scale)")
            plt.ylim(top=2)
        else:
            plt.plot(list(np.array(self.policy_accuracy_history)[:, 0]), list(np.array(self.policy_accuracy_history)[:, -1]), marker='o')
            plt.xlabel("Iteration")
            plt.ylim(-0.1, 1.1)
        plt.ylabel("Policy Accuracy")
        plt.title("Policy Accuracy History")
        plt.grid()
        plt.savefig(policy_accuracy_vs_k_plot_path)
        plt.close()

        policy_accuracy_vs_samples_plot_path = os.path.join(analysis_dir, "policy_accuracy_vs_samples.png")
        plt.figure()
        if log_scale:
            plt.semilogy(list(np.array(self.policy_accuracy_history)[:, 1]), list(np.array(self.policy_accuracy_history)[:, -1]))
            plt.xlabel("Number of Samples (log scale)")
            plt.ylim(top=2)  
        else:
            plt.plot(list(np.array(self.policy_accuracy_history)[:, 1]), list(np.array(self.policy_accuracy_history)[:, -1]))
            plt.xlabel("Number of Samples")
            plt.ylim(-0.1, 1.1)
        plt.ylabel("Policy Accuracy")
        plt.title("Policy Accuracy vs Number of Samples")
        plt.grid()
        plt.savefig(policy_accuracy_vs_samples_plot_path)
        plt.close()

    def plot_bvi_history(self, bvi_history, analysis_dir, log_scale=False):
        # plots the learning error history for each iteration and saves it to error_history_plot_path
        error_history_plot_path = os.path.join(analysis_dir, "bvi_error_history.png")
        os.makedirs(analysis_dir, exist_ok=True)
        plt.figure()
        if log_scale:
            plt.semilogy(list(np.array(bvi_history)[:, -1]), marker='o')
            plt.ylim(top=2)
        else:
            plt.plot(list(np.array(bvi_history)[:, -1]), marker='o')
            plt.ylim(-0.1, 1.1)
        plt.xlabel("Iteration")
        plt.ylabel("Error (U - L)")
        plt.title("BVI Error History")
        plt.grid()
        plt.savefig(error_history_plot_path)
        plt.close()
    
    def plot_value_bounds(self, analysis_dir):
        """Plot lower and upper bounds vs iteration k and vs number of samples."""
        bounds_vs_k_plot_path = os.path.join(analysis_dir, "value_bounds_vs_k.png")
        os.makedirs(analysis_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        k_values = list(np.array(self.learning_history)[:, 0])
        lower_bounds = list(np.array(self.learning_history)[:, -2])
        upper_bounds = list(np.array(self.learning_history)[:, -1])
        
        plt.plot(k_values, lower_bounds, marker='o', label='Lower Bound L(s0)', linewidth=2)
        plt.plot(k_values, upper_bounds, marker='s', label='Upper Bound U(s0)', linewidth=2)
        plt.fill_between(k_values, lower_bounds, upper_bounds, alpha=0.2)
        
        plt.xlabel("Iteration (k)")
        plt.ylabel("Value")
        plt.ylim(-0.1, 1.1)
        plt.title("Value Bounds for Initial State vs Iteration")
        plt.legend()
        plt.grid()
        plt.savefig(bounds_vs_k_plot_path)
        plt.close()

        bounds_vs_samples_plot_path = os.path.join(analysis_dir, "value_bounds_vs_samples.png")
        os.makedirs(analysis_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        sample_counts = list(np.array(self.learning_history)[:, 1])
        lower_bounds = list(np.array(self.learning_history)[:, -2])
        upper_bounds = list(np.array(self.learning_history)[:, -1])
        
        plt.plot(sample_counts, lower_bounds, marker='o', label='Lower Bound L(s0)', linewidth=2)
        plt.plot(sample_counts, upper_bounds, marker='s', label='Upper Bound U(s0)', linewidth=2)
        plt.fill_between(sample_counts, lower_bounds, upper_bounds, alpha=0.2)
        
        plt.xlabel("Number of Samples")
        plt.ylabel("Value")
        plt.ylim(-0.1, 1.1)
        plt.title("Value Bounds for Initial State vs Number of Samples")
        plt.legend()
        plt.grid()
        plt.savefig(bounds_vs_samples_plot_path)
        plt.close()

    def run_analysis(self, analysis_dir):
        """
        Run analysis and generate plots for the learning process.

        Args:
            analysis_dir (str): Directory to save analysis plots and data.
        """
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        self.plot_error_history(analysis_dir, log_scale=True)
        self.plot_states_set_history(analysis_dir)
        self.plot_transitions_seen_history(analysis_dir)
        self.plot_policy_accuracy_history(analysis_dir)
        self.plot_value_bounds(analysis_dir)

        # Save all histories to json file
        analysis_data = {
            "learning_history": self.learning_history,
            "states_set_history": self.states_set_history,
            "transitions_seen_history": self.transitions_seen_history,
            "policy_accuracy_history": self.policy_accuracy_history,
        }
        analysis_data_path = os.path.join(analysis_dir, "analysis_data.json")
        with open(analysis_data_path, "w") as f:
            json.dump(analysis_data, f, indent=4)
    
    def print_learner_results(self, k, confidence_error_k, error_k, p_k):
        print(f"\nLearning interrupted by user on iteration {k}. Saving progress and exiting safely...")
        print("")
        print("Algorithm Variables:")
        print("error_k: ", error_k)
        print("confidence_error_k: ", confidence_error_k)
        print("p_k: ", p_k)
        print(f"Explored states: {len(self.discovered_mdp.states)}/{len(self.mdp_sim.gt_mdp.states)}")
        print("")
        print("Results:")
        print("L(s0):", self.discovered_mdp.s_value_bounds[self.discovered_mdp.initial_state][0])
        print("U(s0):", self.discovered_mdp.s_value_bounds[self.discovered_mdp.initial_state][1])
        print("Error (U - L):", self.discovered_mdp.s_value_bounds[self.discovered_mdp.initial_state][1] - self.discovered_mdp.s_value_bounds[self.discovered_mdp.initial_state][0])

    
    def get_discovered_mdp(self) -> Optional[MDP]:
        """Get the discovered MDP model."""
        return self.discovered_mdp
    
    def print_summary(self, true_confidence_error, true_p_min, k=10, output_path="", analysis_dir=""):

        """Print a summary of the learned model or save it as JSON if output_path is provided."""
        print("Generating learned MDP summary...")
        if self.discovered_mdp is None:
            print("No MDP has been learned yet.")
            return

        collapsed_discovered_mdp = copy.deepcopy(self.discovered_mdp)
        collapsed_discovered_mdp.confidence_error = true_confidence_error
        collapsed_discovered_mdp.p_min = true_p_min
        collapsed_discovered_mdp.collapse(self.find_all_MECs(true_confidence_error, true_p_min))
        collapsed_discovered_mdp.initialize_mdp_value_bounds()

        bvi_history = []
        collapsed_discovered_mdp.update_transition_probabilities()
        for _ in tqdm(range(0, self.num_bvi_iterations(len(collapsed_discovered_mdp.states), k)),
                  desc="Final BVI", unit="it"):
            collapsed_discovered_mdp.bvi_update()
            bvi_history.append([_, true_confidence_error, true_p_min, collapsed_discovered_mdp.get_mdp_error()])
        self.plot_bvi_history(bvi_history, analysis_dir)
        collapsed_discovered_mdp.update_learned_policy()
        print("Error after final BVI:", collapsed_discovered_mdp.get_mdp_error())

        # Prepare data in JSON-serializable form
        def sa_key(state, action):
            return f"{state}||{action}"

        summary = {
            "states_count": len(self.discovered_mdp.states),
            "state_action_pairs_count": len(self.discovered_mdp.state_action_pairs),
            "initial_state": str(self.discovered_mdp.initial_state),
            "goal_states": list(self.discovered_mdp.goal_states),
            "collapsed": {
                "states": list(collapsed_discovered_mdp.states),
                "state_action_pairs": [list(pair) for pair in collapsed_discovered_mdp.state_action_pairs],
                "initial_state": str(collapsed_discovered_mdp.initial_state),
                "goal_states": list(collapsed_discovered_mdp.goal_states),
            },
            "state_value_bounds": {},
            "sa_value_bounds": {},
            "s_value_bounds": {},
            "error": None,
            "sample_counts": {},
            "transition_probabilities": {},
            "learned_policy": {},
            "learning_history": self.learning_history,
            "states_seen_history": self.states_set_history,
            "transitions_seen_history": self.transitions_seen_history,
            "policy_accuracy_history": self.policy_accuracy_history,
        }

        print("Compiling summary data...")

        # State value bounds
        for state in collapsed_discovered_mdp.states:
            try:
                bounds = collapsed_discovered_mdp.get_value_bounds(state)
            except Exception:
                bounds = collapsed_discovered_mdp.s_value_bounds.get(state, (None, None))
            summary["state_value_bounds"][str(state)] = [None if b is None else float(b) for b in bounds]

        # SA value bounds
        for (state, action) in collapsed_discovered_mdp.state_action_pairs:
            bounds = collapsed_discovered_mdp.sa_value_bounds.get((state, action), (None, None))
            summary["sa_value_bounds"][sa_key(state, action)] = [None if b is None else float(b) for b in bounds]

        # s_value_bounds (full MDP)
        for state, bounds in getattr(collapsed_discovered_mdp, "s_value_bounds", {}).items():
            summary["s_value_bounds"][str(state)] = [None if b is None else float(b) for b in bounds]

        # Overall error (U(s0) - L(s0))
        try:
            init = collapsed_discovered_mdp.initial_state
            error_val = (collapsed_discovered_mdp.s_value_bounds[init][1] - collapsed_discovered_mdp.s_value_bounds[init][0])
            summary["error"] = float(error_val)
        except Exception:
            summary["error"] = None

        # Sample counts
        for (state, action) in self.discovered_mdp.state_action_pairs:
            total = self.discovered_mdp.get_sample_count(state, action)
            summary["sample_counts"][sa_key(state, action)] = int(total)

        # Transition probabilities
        for (state, action) in collapsed_discovered_mdp.state_action_pairs:
            trans = collapsed_discovered_mdp.transition_probabilities.get((state, action), {})
            # convert keys to str and values to float
            summary["transition_probabilities"][sa_key(state, action)] = {str(ns): float(p) for ns, p in trans.items()}

        # Learned policy
        for state, best_action in getattr(collapsed_discovered_mdp, "learned_policy", {}).items():
            summary["learned_policy"][str(state)] = best_action

        print("Summary data compilation complete.")

        if output_path:
            # ensure directory exists
            dirpath = os.path.dirname(output_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        else:
            # Human readable printing (keeps previous format)
            print("\n" + "="*60)
            print("DISCOVERED MDP SUMMARY")
            print("="*60)
            print(f"States: {summary['states_count']}")
            print(f"State-Action Pairs: {summary['state_action_pairs_count']}")
            print(f"Initial State: {summary['initial_state']}")
            print(f"Goal States: {summary['goal_states']}")

            print("\nDiscovered MDP after collapsing MECs:")
            print(f"States: {summary['collapsed']['states']}")
            print(f"State-Action Pairs: {summary['collapsed']['state_action_pairs']}")
            print(f"Initial State: {summary['collapsed']['initial_state']}")
            print(f"Goal States: {summary['collapsed']['goal_states']}")

            print("\nState Value Bounds:")
            for state, bounds in summary["state_value_bounds"].items():
                lb = "None" if bounds[0] is None else f"{bounds[0]:.4f}"
                ub = "None" if bounds[1] is None else f"{bounds[1]:.4f}"
                print(f"  {state}: [{lb}, {ub}]")

            print("\nState Action Value Bounds:")
            for sa, bounds in summary["sa_value_bounds"].items():
                lb = "None" if bounds[0] is None else f"{bounds[0]:.4f}"
                ub = "None" if bounds[1] is None else f"{bounds[1]:.4f}"
                print(f"  {sa}: [{lb}, {ub}]")

            print("\nError (U(s0) - L(s0)):", summary["error"])

            print("\nSample Counts by State-Action:")
            for sa, cnt in summary["sample_counts"].items():
                print(f"  {sa}: {cnt} samples")

            print("\nTransition Probabilities:")
            for sa, probs in summary["transition_probabilities"].items():
                print(f"  {sa}: {probs}")

            print("\nLearned Policy:")
            for state, best_action in summary["learned_policy"].items():
                print(f"  State: {state} --> Action: {best_action}")
            
            print("="*60 + "\n")
            print("Analysis")
            print("="*60 + "\n")
            print("Analysis History (k, total_samples, error, num_seen_states, num_seen_transitions, policy_accuracy, error, lower_bound, upper_bound):")
            for i in range(0, len(self.learning_history)):
                record = self.learning_history[i]
                print(f"  {record[0]}, {record[1]}, {record[-3]}, {self.states_set_history[i][-1]}, {self.transitions_seen_history[i][-1]}, {self.policy_accuracy_history[i][-1]}, {record[-3]}, {record[-2]}, {record[-1]}")
        

    def has_converged(self, curr_mdp, prev_iter_history, prev_collapsed_mdp_MEC_states=None):
        """
        Check if the learning process has converged based on the change in MDP error.
        
        Args:
            curr_mdp (MDP): The current discovered MDP after the latest iteration.
            prev_iter_history (tuple): A tuple containing the previous iteration's parameters:
                (k, total_num_samples, delta, p_min, error).
            threshold (float): The threshold for convergence based on error change.
        """
        # Make sure MEC states have not changed
        if (prev_collapsed_mdp_MEC_states is not None):
            curr_mdp_MEC_states = list(curr_mdp.MEC_states.values())
            prev_mdp_MEC_states = list(prev_collapsed_mdp_MEC_states.values())
            if (len(curr_mdp_MEC_states) != len(prev_mdp_MEC_states)):
                return False
            for mec_states in curr_mdp_MEC_states:
                if mec_states not in prev_mdp_MEC_states:
                    return False
                prev_mdp_MEC_states.remove(mec_states)
            if (len(prev_mdp_MEC_states) != 0):
                return False
        min_iterations = self.min_num_iterations
        max_iterations = self.max_num_iterations
        threshold = self.convergence_threshold
        prev_k, prev_total_num_samples, prev_delta, prev_p_min, prev_error, prev_l, prev_u =  prev_iter_history
        if (max_iterations and prev_k >= max_iterations):
            return True  # Reached maximum number of iterations
        if (min_iterations and prev_k < min_iterations):
            return False  # Need at least 'min_iterations' iterations to check for convergence
        test_mdp = copy.deepcopy(curr_mdp)
        test_mdp.confidence_error = prev_delta
        test_mdp.p_min = prev_p_min

        test_mdp.update_transition_probabilities()
        for _ in range(0, self.num_bvi_iterations(len(test_mdp.states), prev_k)):
            # Update
            test_mdp.bvi_update()
        curr_error = test_mdp.get_mdp_error()
        # NOTE: the curr_error may be greater than the prev_error if a new MEC was found.
        # assert curr_error <= prev_error, f"Current error should be less than previous error if BVI is working correctly, got curr_error: {curr_error}, prev_error: {prev_error}"
        print(prev_error, curr_error, (prev_error - curr_error), threshold)
        return (prev_error - curr_error) < threshold

    def num_bvi_iterations(self, num_states, k):
        # TODO: use what they used in their paper? --> len(self.discovered_mdp.states) * (2 ** k))
        return int(num_states * k / 5) + 1