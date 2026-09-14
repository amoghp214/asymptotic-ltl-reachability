import bisect
import random
from collections import defaultdict
from itertools import accumulate
import numpy as np
from typing import Set, Tuple, List, Optional, Any
from mdp import MDP

class MDPSimulator:
    """
    Interface for simulating an MDP (unknown to the learning algorithm).
    Provides step() method that returns (next_state, reward) when given (state, action).
    """

    def __init__(self, mdp=None):
        # `mdp=MDP()` as a default would be evaluated once, at definition time, so
        # every default-constructed simulator would share one MDP object.
        self.gt_mdp = MDP() if mdp is None else mdp
        # Private: per-(state, action) sampling tables, built lazily on first use.
        # Not exposed by any accessor; step()'s return value is unchanged, so the
        # learner gains no information it could not already get from sampling.
        self._sampling_cache = dict()

    def _build_sampling_entry(self, state: Any, action: Any):
        """
        Validate (state, action) once and cache its cumulative distribution, so that
        step() does not repeat the validation and the CDF construction on every call.

        Returns: (next_states_tuple, cumulative_probabilities)
        """
        assert state in self.gt_mdp.states, f"State {state} is not in the given MDP."
        assert action in self.gt_mdp.topology[state], f"Action {action} is not present for state {state} in the given MDP."
        if (state, action) not in self.gt_mdp.state_action_pairs:
            raise ValueError(f"Invalid state-action pair: ({state}, {action})")

        transitions = self.gt_mdp.get_transition_probabilities(state, action)
        next_states = tuple(transitions.keys())
        probabilities = list(transitions.values())
        assert (abs(sum(probabilities) - 1.0) < 1e-8), f"Simulator should have true probabilities that sum up to 1 for ({state}, {action})."

        entry = (next_states, list(accumulate(probabilities)))
        self._sampling_cache[(state, action)] = entry
        return entry

    def step(self, state: Any, action: Any) -> Tuple[Any, float]:
        """
        Execute one step of the MDP.

        Args:
            state: Current state
            action: Action to take

        Returns: (next_state, reward)
        """
        entry = self._sampling_cache.get((state, action))
        if entry is None:
            entry = self._build_sampling_entry(state, action)
        next_states, cumulative = entry

        # Inverse-CDF sample: bisect_right finds which cumulative bucket the uniform
        # draw fell into, so bucket i is hit with probability cumulative[i] -
        # cumulative[i-1]. The clamp guards against float drift leaving the last
        # cumulative entry marginally below 1.0.
        i = bisect.bisect_right(cumulative, random.random())
        if i >= len(next_states):
            i = len(next_states) - 1
        next_state = next_states[i]
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

        # This is the only mutator that touches the distribution step() samples from,
        # so it is the only one that can stale a cached CDF. add_state /
        # add_action_to_state / set_goal_states / set_initial_state only touch states,
        # topology, goal_states or a brand-new empty row, and step() reads goal_states
        # live rather than from the cache.
        self._sampling_cache.pop((state, action), None)

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

    
