"""
Main script for PAC Statistical Model Checking of LTL Reachability Properties
Provides end-to-end learning of reachability objectives from an unknown MDP.
"""

import numpy as np
from mdp import MDP
from mdp_simulator import MDPSimulator
from ltl_reachability_learner import LTLReachabilityLearner
from convert_jani_to_mdp import convert_jani_to_mdp

def sim_mdp_setup_1():
    mdp_sim = MDPSimulator()
    mdp_sim.add_state(1)
    mdp_sim.add_state(2)
    mdp_sim.add_state(3)
    mdp_sim.add_state(4)
    mdp_sim.add_state(5)
    mdp_sim.add_state(6)

    mdp_sim.set_goal_states(set([3, 6]))

    mdp_sim.add_action_to_state(1, 'b')
    mdp_sim.add_action_to_state(1, 'c')
    mdp_sim.add_action_to_state(1, 'd')
    mdp_sim.add_action_to_state(2, 'b')
    mdp_sim.add_action_to_state(3, 'a')
    mdp_sim.add_action_to_state(4, 'a')
    mdp_sim.add_action_to_state(4, 'd')
    mdp_sim.add_action_to_state(5, 'a')
    mdp_sim.add_action_to_state(6, 'a')

    mdp_sim.add_transition(1, 'b', 6, 0.1)
    mdp_sim.add_transition(1, 'b', 5, 0.9)
    mdp_sim.add_transition(1, 'c', 1, 0.4)
    mdp_sim.add_transition(1, 'c', 2, 0.05)
    mdp_sim.add_transition(1, 'c', 3, 0.1)
    mdp_sim.add_transition(1, 'c', 4, 0.05)
    mdp_sim.add_transition(1, 'c', 5, 0.2)
    mdp_sim.add_transition(1, 'c', 6, 0.2)
    mdp_sim.add_transition(1, 'd', 4, 1)
    mdp_sim.add_transition(2, 'b', 2, 0.5)
    mdp_sim.add_transition(2, 'b', 3, 0.5)
    mdp_sim.add_transition(3, 'a', 3, 0.5)
    mdp_sim.add_transition(3, 'a', 4, 0.25)
    mdp_sim.add_transition(3, 'a', 5, 0.25)
    mdp_sim.add_transition(4, 'a', 3, 0.5)
    mdp_sim.add_transition(4, 'a', 6, 0.1)
    mdp_sim.add_transition(4, 'a', 1, 0.4)
    mdp_sim.add_transition(4, 'd', 1, 0.9999)
    # mdp_sim.add_transition(4, 'd', 1, 1)
    mdp_sim.add_transition(4, 'd', 5, 0.0001)
    mdp_sim.add_transition(5, 'a', 5, 1)
    mdp_sim.add_transition(6, 'a', 6, 1)

    mdp_sim.set_initial_state(1)

    return mdp_sim

def example_simple_mdp():
    """
    Example 1: Simple 3-state MDP
    States: 0, 1, 2 (2 is goal)
    Actions: a, b
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple 3-State MDP")
    print("="*60)

    mdp_sim = sim_mdp_setup_1()

    print("===================== True MDP Schema =====================")
    print(mdp_sim.gt_mdp.initial_state)
    print(mdp_sim.gt_mdp.states)
    print(mdp_sim.gt_mdp.topology)
    print(mdp_sim.gt_mdp.state_action_pairs)
    print(mdp_sim.gt_mdp.transition_probabilities)
    
    # Create learner
    learner = LTLReachabilityLearner(mdp_simulator=mdp_sim)

    print("================= Starting Learning Process =================")
    analysis_path = "./results/example_analysis"
    
    learner.learn(analysis_path)
    learner.print_summary(true_confidence_error=0.0001, true_p_min=0.3, analysis_dir=analysis_path)

def ij3_jani_mdp():
    """
    Example 1: Simple 3-state MDP from JANI file
    States: 0, 1, 2 (2 is goal)
    Actions: a, b
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple 3-State MDP")
    print("="*60)

    mdp_sim = MDPSimulator(mdp=convert_jani_to_mdp("./mdp_models/ij.3.v1.jani"))

    # print("===================== True MDP Schema =====================")
    # print(mdp_sim.gt_mdp.initial_state)
    # print(mdp_sim.gt_mdp.states)
    # print(mdp_sim.gt_mdp.topology)
    # print(mdp_sim.gt_mdp.state_action_pairs)
    # print(mdp_sim.gt_mdp.transition_probabilities)
    
    # Create learner
    learner = LTLReachabilityLearner(mdp_simulator=mdp_sim)

    print("================= Starting Learning Process =================")
    analysis_path = "./results/ij3_analysis"
    learner.learn(analysis_path)
    
    learner.print_summary(true_confidence_error=0.01, true_p_min=0.5, output_path="./logs/ij3_learning_log.json", analysis_dir=analysis_path)

def ij10_jani_mdp():
    """
    Example 1: Simple 3-state MDP from JANI file
    States: 0, 1, 2 (2 is goal)
    Actions: a, b
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple 3-State MDP")
    print("="*60)

    mdp_sim = MDPSimulator(mdp=convert_jani_to_mdp("./mdp_models/ij.10.v1.jani"))

    # print("===================== True MDP Schema =====================")
    # print(mdp_sim.gt_mdp.initial_state)
    # print(mdp_sim.gt_mdp.states)
    # print(mdp_sim.gt_mdp.topology)
    # print(mdp_sim.gt_mdp.state_action_pairs)
    # print(mdp_sim.gt_mdp.transition_probabilities)
    
    # Create learner
    learner = LTLReachabilityLearner(mdp_simulator=mdp_sim)

    print("================= Starting Learning Process =================")
    analysis_path = "./results/ij10_analysis"
    learner.learn(analysis_path)
    
    learner.print_summary(true_confidence_error=0.01, true_p_min=0.5, output_path="./logs/ij10_learning_log.json", analysis_dir=analysis_path)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run examples
    example_simple_mdp()
    # ij10_jani_mdp()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
