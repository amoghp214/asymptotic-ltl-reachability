from collections import defaultdict
import numpy as np
from typing import Set, Tuple, List, Optional, Any
from mdp import MDP

class MDPSimulator:
    """
    Interface for simulating an MDP (unknown to the learning algorithm).
    Provides step() method that returns (next_state, reward) when given (state, action).
    """
    
    def __init__(self, mdp=MDP()):
        self.gt_mdp = mdp
    
    def step(self, state: Any, action: Any) -> Tuple[Any, float]:
        """
        Execute one step of the MDP.

        Args:
            state: Current state
            action: Action to take

        Returns: (next_state, reward)
        """
        assert state in self.gt_mdp.states, f"State {state} is not in the given MDP."
        assert action in self.gt_mdp.topology[state], f"Action {action} is not present for state {state} in the given MDP."
        if (state, action) not in self.gt_mdp.state_action_pairs:
            raise ValueError(f"Invalid state-action pair: ({state}, {action})")
        
        transitions = self.gt_mdp.get_transition_probabilities(state, action)
        # Given this transitions dictionary, sample next state according to the probabilities
        next_states = list(transitions.keys())
        probabilities = list(transitions.values())
        assert (abs(sum(probabilities) - 1.0) < 1e-8), f"Simulator should have true probabilities that sum up to 1 for ({state}, {action})."
        sampled_index = np.random.choice(len(next_states), p=probabilities)
        next_state = next_states[sampled_index]
        reward = (next_state in self.gt_mdp.goal_states)

        return next_state, reward
    
    def add_transition(self, state, action, next_state, probability):
        """
        Record a transition in the MDP.
        
        Args:
            state: Source state
            action: Action taken
            next_state: Destination state
            probability: True transition probability
        """
        assert (state in self.gt_mdp.states), f"The inputted state {state} must alrady exist in the MDP. Please add it before trying to add a transition."
        assert (action in self.gt_mdp.topology[state]), f"The inputted action {action} must alrady exist in the MDP for the input state. Please add it before trying to add a transition."
        
        if (state, action) not in self.gt_mdp.transition_probabilities:
            self.gt_mdp.transition_probabilities[(state, action)] = {}
        if (state, action) not in self.gt_mdp.sample_counts:
            self.gt_mdp.sample_counts[(state, action)] = defaultdict(int)
        
        if probability is not None:
            self.gt_mdp.transition_probabilities[(state, action)][next_state] = probability

    def add_state(self, state, is_goal=False):
        """Add a state to the MDP."""
        self.gt_mdp.add_state(state, is_goal=is_goal)
    
    def add_action_to_state(self, state, action):
        """Add an action available at a state."""
        self.gt_mdp.add_action_to_state(state, action)

    def set_goal_states(self, goal_states):
        """Set the goal states for reachability."""
        self.gt_mdp.set_goal_states(goal_states)
    
    def set_initial_state(self, state):
        """Set the initial/start state."""
        self.gt_mdp.set_initial_state(state)

    
