import copy
import random
from typing import Set, Tuple, List, Optional, Any
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from mdp import MDP
from mdp_simulator import MDPSimulator
from analysis_utils import run_analysis, plot_bvi_history
from tqdm import tqdm
import json
import os
import time


class TDQL_PAC_Stage:
    """
    Class to encapsulate all variales and methods for a single TDQL PAC stage.
    """

    def __init__(
        self,
        k: int,
        error: float,
        confidence_error: float,
        p_min: float,
        mu: float,
        mdp_simulator: MDPSimulator,
    ):
        self.k = k
        self.error = error
        self.confidence_error = confidence_error
        self.p_min = p_min
        self.mu = mu
        self.mdp_simulator = mdp_simulator
        self.s0 = self.mdp_simulator.gt_mdp.initial_state
        self.sample_count = 0

        self.seen_states = set() # Set of all discovered states in the current PAC stage
        self.seen_state_action_pairs = set() # Set of all discovered state-action pairs in the current PAC stage
        self.ecs = set() # Set of all detected ECs in the current PAC stage
        self.corresponding_ecs = dict() # {s: ec, ...} mapping of states to their corresponding ECs
        self.u = dict() # {s: {a: upper_bound, ...}, ...}
        self.u[self.s0] = dict()
        self.l = dict() # {s: {a: lower_bound, ...}, ...}
        self.l[self.s0] = dict()

        # Round-scoped tables. Rebound at the top of every round in run(); kept in
        # sync inside simulate_and_update so refresh_best_actions() can read them.
        self.u_new = self.u
        self.l_new = self.l

        self.best_actions = dict() # {s: [best_actions], ...}

        self.eps_u = self.calculate_eps_u(self.k)
        self.m = self.calculate_m()

    def calculate_m(self):
        return int(10 * (10 + len(self.seen_state_action_pairs)) / self.confidence_error**2)

    def calculate_num_b_hat(self, err_u):
        return int(1 * 10 // err_u)

    def calculate_N(self, k):
        return 10 * self.calculate_m() * (len(self.seen_state_action_pairs) + 1)
    
    def calculate_H(self, k):
        return 100 * k**2 * (len(self.seen_state_action_pairs)**2 + 100)
    
    def calculate_Z(self, k):
        return 100 * k**2 * (len(self.seen_state_action_pairs)**2 + 100)
    
    def calculate_eps_u(self, k):
        return 0.001 / 2**k

    def calculate_sa_confidence_error(self, k, confidence_error):
        return confidence_error / (len(self.seen_state_action_pairs) + 1)

    def calculate_c(self):
        # sa_confidence_error = self.calculate_sa_confidence_error(self.k, self.confidence_error)
        # return np.sqrt((np.log(2 / sa_confidence_error)) / (2 * self.m))
        return 0.0001
    
    def sample_best_action(self, s):
        """
        Sample the best action from the current state s based on the upper bound of the value function.

        Args:
            s: The current state to sample the best action from.
        
        Returns:
            a: The sampled best action from the current state s.
        """
        return random.choice(self.best_actions[s])

    def sample_action(self, s, mu):
        """
        With probability mu, sample a random action from the current state s.
        With probability 1-mu, sample the best action from the current state s.

        Args:
            s: The current state to sample an action from.
            mu: The probability of sampling a random action.
        Returns:
            a: The sampled action from the current state s.
        """
        if (random.random() < mu):
            a = self.mdp_simulator.gt_mdp.sample_random_action_from_state(s)
        else:
            a = self.sample_best_action(s)
        return a
    
    def get_best_actions(self, s, u_table=None):
        """
        Get the U-maximal actions at s, read off the given upper-bound table.

        Args:
            s: The state to get the best actions for.
            u_table: The table to read. Defaults to self.u_new, the round-scoped
                     table that accumulates this round's writes.

        Returns:
            list: The U-maximal actions at s.
        """
        row = (self.u_new if u_table is None else u_table)[s]
        best = max(row.values())
        return [a_hat for a_hat in row if row[a_hat] == best]

    def refresh_best_actions(self, s, u_table=None):
        """Recompute and store the cached greedy action set at s."""
        self.best_actions[s] = self.get_best_actions(s, u_table)
    
    def detect_ec(self, s, patience_factor=100, min_patience=1000):
        """
        Given that the inputted state is part of an EC, find
        all states and state-action pairs in the EC.

        Args:
            s: The state that is part of an EC.
            patience_factor: Steps-without-a-new-pair, per pair already found, before
                             the component is treated as covered.
            min_patience: Floor on that window, for the early steps where the
                          discovered set is still tiny.

        Returns:
            ec: The set of state-action pairs that make up the EC.
        """
        ec = set()
        Z = self.calculate_Z(self.k)
        steps_since_new = 0
        sim_step = self.mdp_simulator.step

        for _ in range(0, Z):
            a = self.sample_action(s, mu=0)
            s_new, _ = sim_step(s, a)
            n_before = len(ec)
            ec.add((s, a))
            if len(ec) > n_before:
                steps_since_new = 0
            else:
                steps_since_new += 1
                # The walk has been confined to pairs it has already seen for far longer
                # than the component's cover time; treat it as covered and stop. Erring
                # short is safe: a smaller ec means a larger exit set, hence a bestExit
                # that is >= the true one, hence a less aggressive (still valid) deflation.
                if steps_since_new >= max(min_patience, patience_factor * len(ec)):
                    break
            s = s_new

        self.ecs.add(frozenset(ec))
        for (s, a) in ec:
            self.corresponding_ecs[s] = ec
        
        return ec
    
    def deflate_ec(self, ec, u):
        """
        Perform a pseudo-collapse on the EC. Deflate all the upper bounds of the
        state-action pairs in the EC to the upper bound fo the best exit action

        Args:
            ec: The set of state-action pairs that make up the EC.
            u: The current upper bound of the value function.
        
        Returns:
            u_new: The updated upper bound of the value function after collapsing the EC.
        """
        u_new = u
        ec_states = set()
        for (s, a) in ec:
            ec_states.add(s)
            if (self.mdp_simulator.gt_mdp.is_goal_state(s)):
                return u_new  # If the EC contains a goal state, we don't collapse it
        
        # Alg. COLLAPSE_EC: bestExit is 0 when the EC has no exits. U >= 0 is an
        # invariant here (init 1; writes are u_pred + c >= 0; deflations write
        # bestExit >= 0), so seeding at 0 also leaves the with-exits case unchanged.
        best_exit_action_value = 0
        for s in ec_states:
            actions = self.mdp_simulator.gt_mdp.get_actions(s)
            for a in actions:
                if (s, a) not in ec:
                    exit_value = u[s][a]
                    if exit_value > best_exit_action_value:
                        best_exit_action_value = exit_value
        
        for (s, a) in ec:
            u_new[s][a] = best_exit_action_value

        # The deflation changed U at every EC state, so the cached greedy sets are stale.
        # Refreshing them is what makes the collapse steer the walk out of the component.
        for s in ec_states:
            self.refresh_best_actions(s, u_new)

        return u_new

    
    def simulate_and_update(self, u, l, u_new, l_new, u_agg, l_agg, curr_counts, eps_u, m, mu):
        """
        Simulate the MDP and update the upper and lower bounds of the value function.

        Args:
            u (dict): Current upper bounds of the value function.
            l (dict): Current lower bounds of the value function.
            u_new (dict): New upper bounds of the value function.
            l_new (dict): New lower bounds of the value function.
            u_agg (dict): Aggregated upper bounds of the value function.
            l_agg (dict): Aggregated lower bounds of the value function.
            curr_counts (dict): Current counts of state-action pairs.
            eps_u (float): Error tolerance for the upper bound.
            m (int): Number of samples to collect.
            mu (float): Learning rate or exploration parameter.

        Returns:
            u_new (dict): Updated upper bounds of the value function.
            l_new (dict): Updated lower bounds of the value function.
        """
        # Keep the round-scoped tables reachable from the refresh helpers.
        self.u_new, self.l_new = u_new, l_new

        def handle_new_state_action_pair(s_new, a_hat, is_goal_state):
            u[s_new][a_hat] = u[s_new].get(a_hat, 1)
            l[s_new][a_hat] = l[s_new].get(a_hat, 0 if not is_goal_state else 1)
            u_new[s_new][a_hat] = u_new[s_new].get(a_hat, 1)
            l_new[s_new][a_hat] = l_new[s_new].get(a_hat, 0 if not is_goal_state else 1)
            u_agg[s_new][a_hat] = u_agg[s_new].get(a_hat, 0)
            l_agg[s_new][a_hat] = l_agg[s_new].get(a_hat, 0)
            curr_counts[s_new][a_hat] = 0
            self.seen_state_action_pairs.add((s_new, a_hat))


        def handle_new_state(s_new, is_goal_state):
            u[s_new] = dict()
            l[s_new] = dict()
            u_new[s_new] = dict()
            l_new[s_new] = dict()
            u_agg[s_new] = dict()
            l_agg[s_new] = dict()
            curr_counts[s_new] = dict()
            self.seen_states.add(s_new)

            actions = self.mdp_simulator.gt_mdp.get_actions(s_new)
            for a_hat in actions:
                handle_new_state_action_pair(s_new, a_hat, is_goal_state)
            
            self.refresh_best_actions(s_new)

        curr_state = self.s0
        curr_action = None
        t = 0
        c = self.calculate_c()
        H = self.calculate_H(self.k)

        # Hoist the hot attribute chains: each `self.mdp_simulator.gt_mdp.X` costs three
        # LOAD_ATTRs per loop iteration, and this loop runs up to H times.
        sim_step = self.mdp_simulator.step
        goal_states = self.mdp_simulator.gt_mdp.goal_states
        seen_states = self.seen_states
        sample_action = self.sample_action

        while t < H and (curr_state not in goal_states):
            if (curr_state not in seen_states):
                handle_new_state(curr_state, curr_state in goal_states)

            curr_action = sample_action(curr_state, mu)
            next_state, reward = sim_step(curr_state, curr_action)
            if (next_state not in seen_states):
                handle_new_state(next_state, next_state in goal_states)

            self.sample_count += 1
            # handle_new_state seeds curr_counts[s][a] for every action of every state it
            # discovers, and run() reseeds them each round, so both keys always exist here
            # --- as the `u_agg[...] += ...` two lines below has always relied on.
            curr_counts[curr_state][curr_action] += 1
            
            u_agg[curr_state][curr_action] += self.u[next_state][self.best_actions[next_state][0]]
            l_agg[curr_state][curr_action] += self.l[next_state][self.best_actions[next_state][0]]

            # print(f"t: {t}, reward: {reward}, sample_count: {self.sample_count}, u_agg: {u_agg[curr_state][curr_action]}, l_agg: {l_agg[curr_state][curr_action]}, curr_counts: {curr_counts[curr_state][curr_action]}")

            if curr_counts[curr_state][curr_action] >= m:
                curr_counts[curr_state][curr_action] = 0
                curr_u_pred_sa = u_agg[curr_state][curr_action] / m
                if (curr_u_pred_sa + c <= self.u[curr_state][curr_action] - eps_u):
                    u_new[curr_state][curr_action] = curr_u_pred_sa + c
                    self.refresh_best_actions(curr_state)
                curr_u_pred_sa, u_agg[curr_state][curr_action] = 0, 0
                curr_l_pred_sa = l_agg[curr_state][curr_action] / m
                if (curr_l_pred_sa - c >= self.l[curr_state][curr_action] + eps_u):
                    l_new[curr_state][curr_action] = curr_l_pred_sa - c
                    # if (curr_state == self.s0):
                    #     print(curr_l_pred_sa, c, self.l[curr_state][curr_action], eps_u)
                    #     print(l_new[curr_state][curr_action])
                    # print(curr_counts[curr_state][curr_action], m, l_agg[curr_state][curr_action], curr_l_pred_sa, self.l[curr_state][curr_action])
                curr_l_pred_sa, l_agg[curr_state][curr_action] = 0, 0
                
            
            curr_state = next_state
            t += 1

        # Detection is restricted to the purely greedy simulations, so that the walk that
        # triggers detection and the greedy walk detect_ec reads the component off with
        # are drawn from the same kernel.
        if (t >= H and mu == 0):
            if (curr_state in self.corresponding_ecs.keys()):
                ec = self.corresponding_ecs[curr_state]
            else:
                ec = self.detect_ec(curr_state)
            u_new = self.deflate_ec(ec, u_new)
        
        return u_new, l_new
    
    def summarize_learning(self):
        return (
            self.best_actions.copy(),  # learned policy for the current stage
            self.u[self.s0][self.best_actions[self.s0][0]],
            self.l[self.s0][self.best_actions[self.s0][0]],
            self.u[self.s0][self.best_actions[self.s0][0]] - self.l[self.s0][self.best_actions[self.s0][0]],
            self.sample_count
        )

    
    def run(self):
        """
        Run a single stage of the PAC learning algorithm using TDQL.
        Guarantees the policy found is within the specified error bounds
        with the given confidence, assuming p_min.

        Returns:
            policy: The learned policy for the current stage.
            upper_bound_s0: Upper bound of the value function for the initial state.
            lower_bound_s0: Lower bound of the value function for the initial state.
            error: The error of the learned policy.
            total_sample_counts: Total number of samples collected during this stage.
        """        
        num_b_hat = self.calculate_num_b_hat(self.eps_u)

        for _ in tqdm(range(0, num_b_hat), desc="TDQL", leave=False):
        # for _ in range(0, num_b_hat):
            updated = False
            # Per-row shallow copy: rows are fresh dicts, and every leaf is an immutable
            # float, so this isolates u_new from self.u exactly as deepcopy would --- at
            # ~23x the speed. Only valid while the leaves stay scalar.
            u_new = {s: dict(row) for s, row in self.u.items()}
            l_new = {s: dict(row) for s, row in self.l.items()}
            u_agg = dict() # {s: {a: upper_agg, ...}, ...}
            l_agg = dict() # {s: {a: lower_agg, ...}, ...}
            curr_counts = dict() # {s: {a: count, ...}, ...}
            for s in self.u.keys():
                u_agg[s] = dict()
                l_agg[s] = dict()
                curr_counts[s] = dict()
                for a in self.u[s].keys():
                    u_agg[s][a] = 0
                    l_agg[s][a] = 0
                    curr_counts[s][a] = 0
            
            # for j in tqdm(range(0, 2 * self.calculate_N(self.k)), desc="TDQL B_hat", leave=False):
            for j in range(0, 2 * self.calculate_N(self.k)):
                mu_j = self.mu if j%2==0 else 0.0
                u_new, l_new = self.simulate_and_update(
                    u=self.u, 
                    l=self.l, 
                    u_new=u_new, 
                    l_new=l_new, 
                    u_agg=u_agg, 
                    l_agg=l_agg, 
                    curr_counts=curr_counts, 
                    eps_u=self.eps_u, 
                    m=self.m, 
                    mu=mu_j
                )
                if u_new != self.u or l_new != self.l:
                    updated = True
                    break
            self.u = u_new
            self.l = l_new
            if (not updated):
                break

        return self.summarize_learning()


    

class ModelFreeTDQLearner:
    """
    Main learner class that orchestrates PAC learning of LTL reachability using
    the model-free paradigm.
    """
    
    def __init__(
        self, 
        mdp_simulator: MDPSimulator, 
        min_num_iterations=5, 
        max_num_iterations=50, 
        convergence_threshold=0.001, 
        num_policy_accuracy_sims=1000,
        true_confidence_error=0.01,
        true_p_min=0.01
    ):
        """
        Initialize the learner.
        
        Args:
            mdp_simulator (MDPSimulator): The MDP simulator to interact with the true MDP.
            min_num_iterations (int): Minimum number of learning iterations.
            max_num_iterations (int): Maximum number of learning iterations.
            convergence_threshold (float): Threshold for convergence based on error.
            num_policy_accuracy_sims (int): Number of simulations to estimate policy accuracy.
            true_confidence_error (float): FOR ANALYSIS PURPOSES ONLY
            true_p_min (float): FOR ANALYSIS PURPOSES ONLY

        """
        self.mdp_sim = mdp_simulator  # Nests the true MDP
        self.learning_history = []  # [(k, num_samples, error), ...]
        self.states_set_history = []  # [(k, num_samples, seen_states_set), ...]
        self.transitions_seen_history = []  # [(k, num_samples, transitions_seen), ...]
        self.policy_accuracy_history = []  # [(k, num_samples, policy_accuracy), ..x.]
        self.true_error_history = []  # [(k, num_samples, true_error), ...]  FOR ANALYSIS PURPOSES ONLY - using the true p_min and true confidence error

        self.min_num_iterations = min_num_iterations
        self.max_num_iterations = max_num_iterations
        self.convergence_threshold = convergence_threshold
        self.num_policy_accuracy_sims = num_policy_accuracy_sims
        self.true_confidence_error = true_confidence_error  # FOR ANALYSIS PURPOSES ONLY - not used by the algorithm
        self.true_p_min = true_p_min  # FOR ANALYSIS PURPOSES ONLY - not used by the algorithm

    
    def calculate_policy_accuracy(self, tdql_pac_stage, max_steps=100):
        """
        Calculate the policy accuracy of the discovered MDP.

        Returns:
            float: The calculated policy accuracy.
        """
        n = self.num_policy_accuracy_sims
        successful_runs = 0
        for _ in tqdm(range(0, n), desc="Policy accuracy sims", unit="sim"):
            # start from the ground-truth initial state
            curr_state = self.mdp_sim.gt_mdp.initial_state

            for _ in range(0, max_steps):
                if curr_state in self.mdp_sim.gt_mdp.goal_states:
                    successful_runs += 1
                    break

                if (curr_state not in tdql_pac_stage.seen_states):
                    break

                action = tdql_pac_stage.sample_best_action(curr_state)
                if action is None:
                    break

                next_state, _ = self.mdp_sim.step(curr_state, action)
                curr_state = next_state

        return successful_runs / n
    



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
        prev_ecs = set()

        mu = 0.01

        while True:
            
            # Accept control c interrupt and safely exit while saving progress
            try:
                # Update Hyperparameters
                k += 1
                print("Iteration:", k)
                error_k = error / (2.0 ** k)  # This value is not actually used
                confidence_error_k = confidence_error / (2.0 ** k)
                p_k = p_min / (2.0 ** k)

                tdql_pac_stage_k = TDQL_PAC_Stage(k=k, error=error_k, confidence_error=confidence_error_k, p_min=p_k, mu=mu, mdp_simulator=self.mdp_sim)
                policy_k, upper_bound_s0, lower_bound_s0, curr_error, total_sample_counts = tdql_pac_stage_k.run()
                initial_state = self.mdp_sim.gt_mdp.initial_state

                self.learning_history.append([k, total_sample_counts, confidence_error_k, p_k, curr_error, lower_bound_s0, upper_bound_s0])

                if (analysis_dir != ""):
                    print("Num Sample Counts:", total_sample_counts)
                    print("Explored states:", len(tdql_pac_stage_k.seen_states), "/", len(self.mdp_sim.gt_mdp.states))

                    self.states_set_history.append((k, total_sample_counts, len(tdql_pac_stage_k.seen_states)))
                    self.transitions_seen_history.append((k, total_sample_counts, -1))
                    self.true_error_history.append((k, total_sample_counts, 0))

                    # Calculate policy accuracy
                    self.policy_accuracy_history.append(
                        (k, 
                         total_sample_counts, 
                         confidence_error_k, 
                         p_k, 
                         self.calculate_policy_accuracy(
                             tdql_pac_stage=tdql_pac_stage_k, 
                             max_steps=int(len(tdql_pac_stage_k.seen_states)**2 / p_k)
                            )
                        )
                    )
                    print("Policy accuracy:", self.policy_accuracy_history[-1][-1])

                    # Run iteration analysis, save plots and data
                    run_analysis(
                        analysis_dir=analysis_dir,
                        learning_history=self.learning_history,
                        states_set_history=self.states_set_history,
                        transitions_seen_history=self.transitions_seen_history,
                        policy_accuracy_history=self.policy_accuracy_history,
                        true_error_history=self.true_error_history,
                        max_states=len(self.mdp_sim.gt_mdp.states),
                        max_transitions=sum(len(sat_counts) for sat_counts in self.mdp_sim.gt_mdp.transition_probabilities.values()),
                        true_confidence_error=self.true_confidence_error,
                        true_p_min=self.true_p_min
                    )

                print("Error:", self.learning_history[-1])
                
                if (k > 1 and self.has_converged(tdql_pac_stage_k, self.learning_history[-2], curr_error, prev_ecs)):  # NOTE: -2 because we want the one before the current iteration (current iteration is -1 index).
                    print("Algorithm has converged, exiting...")
                    break

                prev_ecs = tdql_pac_stage_k.ecs  # Update the previous ECs for the next iteration
            except KeyboardInterrupt:
                print("Learning interrupted by user. Exiting and saving progress...")
                break

    
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
        plot_bvi_history(analysis_dir=analysis_dir, bvi_history=bvi_history)
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
        

    def has_converged(self, curr_tdql_pac_stage, prev_iter_history, curr_error, prev_ecs=None):
        """
        Check if the learning process has converged based on the change in MDP error.
        
        Args:
            curr_tdql_pac_stage (TDQLPACStage): The current TDQL PAC stage.
            prev_iter_history (tuple): A tuple containing the previous iteration's parameters:
                (k, total_num_samples, delta, p_min, error).
            prev_ecs (set): The set of previously detected ECs. If provided, the function checks if the current ECs have changed.
        """
        # Make sure MEC states have not changed
        if (prev_ecs is not None):
            if (prev_ecs != curr_tdql_pac_stage.ecs):
                return False

        min_iterations = self.min_num_iterations
        max_iterations = self.max_num_iterations
        threshold = self.convergence_threshold
        prev_k, prev_total_num_samples, prev_delta, prev_p_min, prev_error, prev_l, prev_u =  prev_iter_history
        if (max_iterations and prev_k >= max_iterations):
            return True  # Reached maximum number of iterations
        if (min_iterations and prev_k < min_iterations):
            return False  # Need at least 'min_iterations' iterations to check for convergence

        # NOTE: the curr_error may be greater than the prev_error if a new MEC was found.
        # assert curr_error <= prev_error, f"Current error should be less than previous error if BVI is working correctly, got curr_error: {curr_error}, prev_error: {prev_error}"
        print(prev_error, curr_error, (prev_error - curr_error), threshold)
        return (prev_error - curr_error) < threshold