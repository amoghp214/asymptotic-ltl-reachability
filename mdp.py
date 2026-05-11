import numpy as np
from collections import defaultdict, Counter

class MDP:
    """
    Markov Decision Process class that stores both the discovered MDP structure
    and metadata for PAC learning of reachability objectives.
    """
    def __init__(self, confidence_error=1, p_min=0, initial_state=None):
        # NOTE: We only update sample counts in real time,
        #       values are calculated the moment they is needed (during BVI updates).

        ### FOR NORMAL STATES
        self.topology = dict()  # {state: set(action)} - available actions per state
        # Stores lower bound transition probabilities for each state-action pair
        self.transition_probabilities = dict()  # {(state, action): {next_state: lower_bound_probability}}
        self.states = set()  # All discovered states
        self.goal_states = set()  # Goal states for reachability
        self.state_action_pairs = set()  # All discovered (state, action)
        self.s_value_bounds = dict()  # {state: [lower_value_bound, upper_value_bound]}
        self.sa_value_bounds = dict()  # {(state, action): [lower_value_bound, upper_value_bound]}
        self.sample_counts = dict()  # {(state, action): {next_state: count}} - frequency of transitions
        self.learned_policy = dict()  # {state: action} - For a given state, what is the best possible action to take
        self.initial_state = initial_state  # Starting state for learning

        self.state_MEC = dict()  # {state: respective super_state} - many to 1 relation between states and MECs/super_states
        self.MEC_states = dict()  # {super_state: set(states)} - reverse mapping of state_MEC
        self.MEC_state_action_pairs = dict()  # {(super_state, (state, action)): True} - all discovered (super_state, (state, action)) pairs

        self.confidence_error = confidence_error  # delta
        self.p_min = p_min  # minimum transition probability


    def add_state(self, state, is_goal=False):
        """Add a state to the MDP."""
        if (state in self.states):
            self.topology[state] = self.topology.get(state, set())
            self.s_value_bounds[state] = self.s_value_bounds.get(state, [0, 1])
            if (is_goal):
                self.goal_states.add(state)
                self.s_value_bounds[state] = [1, 1]
            return
        
        assert state not in self.states
        assert state not in self.topology.keys()
        assert state not in self.goal_states
        assert state not in self.s_value_bounds

        self.states.add(state)
        if (is_goal):
            self.goal_states.add(state)
        self.topology[state] = set()
        self.s_value_bounds[state] = [1, 1] if is_goal else [0, 1]

    def add_action_to_state(self, state, action):
        """Add an action available at a state."""
        assert (state in self.states), f"The inputted state {state} must alrady exist in the MDP. Please add it before trying to add an action."
        
        if action in self.topology[state]:
            self.transition_probabilities[(state, action)] = self.transition_probabilities.get((state, action), dict())
            self.state_action_pairs.add((state, action))
            self.sa_value_bounds[(state, action)] = self.sa_value_bounds.get((state, action), [0, 1])
            self.sample_counts[(state, action)] = self.sample_counts.get((state, action), Counter())
            return
        
        assert action not in self.topology[state]
        assert (state, action) not in self.transition_probabilities.keys()
        assert (state, action) not in self.state_action_pairs
        assert (state, action) not in self.sa_value_bounds.keys()
        assert (state, action) not in self.sample_counts.keys()

        self.topology[state].add(action)
        self.transition_probabilities[(state, action)] = dict()
        self.state_action_pairs.add((state, action))
        self.sa_value_bounds[(state, action)] = [0, 1]
        self.sample_counts[(state, action)] = Counter()
    
    def remove_state(self, state):
        """Remove a state from the MDP."""
        assert (state in self.states), f"The inputted state {state} does not exist in the MDP. Cannot remove non-existent state."
        
        actions_to_remove = [action for action in self.topology[state]]

        self.states.remove(state)
        if state in self.goal_states:
            self.goal_states.remove(state)
        del self.topology[state]
        del self.s_value_bounds[state]
        
        for action in actions_to_remove:
            self.state_action_pairs.remove((state, action))
            del self.transition_probabilities[(state, action)]
            del self.sa_value_bounds[(state, action)]
            del self.sample_counts[(state, action)]
        
        if state in self.learned_policy:
            del self.learned_policy[state]

    def add_super_state(self, mec):
        """
        Collapse an MEC (a special set of states) into one super state.
        We will explore sim MDP, collapse ECs, and then run BVI.

        Args:
            mec (tuple): The set of states that form a MEC and the respective state-action pairs.
        """
        state_set, state_action_pairs = mec
        assert state_set.issubset(self.states), "state_set must be a subset of the current MDP states."
        new_super_state = f"mec_{len(self.MEC_states.keys())+1}"
        assert new_super_state not in self.MEC_states.keys()

        # NOTE: the MEC must be maximal (there should be no overlapping ECs)
        for s in state_set:
            assert s not in self.state_MEC.keys() or self.state_MEC[s] in ['', None], f"State {s} already present in state_MEC, a state cannot be part of multiple MECs."
        
        self.MEC_states[new_super_state] = state_set
        self.add_state(new_super_state, is_goal=any(s in self.goal_states for s in state_set))

        for s in state_set:
            self.state_MEC[s] = new_super_state
            for a in self.topology[s]:
                self.add_action_to_state(new_super_state, (s, a))
                self.transition_probabilities[(new_super_state, (s, a))] = self.transition_probabilities[(s, a)]
                self.sa_value_bounds[(new_super_state, (s, a))] = self.sa_value_bounds[(s, a)] if (s, a) not in state_action_pairs else [0, 0]
                self.sample_counts[(new_super_state, (s, a))] = self.sample_counts[(s, a)]
            self.s_value_bounds[new_super_state] = [0, 1] # TODO: is this correct?
            if (s == self.initial_state):
                self.initial_state = new_super_state

        self.MEC_state_action_pairs[new_super_state] = set()
        for s, a in state_action_pairs:
            self.MEC_state_action_pairs[new_super_state].add((s, a))

    def collapse_state_MEC_transitions(self, state):
        """
        Collapse sample counts values from a state to their respective MECs/super_states.

        Args:
            state: The state whose transitions need to be collapsed.
        """
        assert (state in self.states), f"The inputted state {state} must alrady exist in the MDP. Please add it before trying to collapse its MEC transitions."

        for a in self.topology[state]:
            mec_sample_counts = Counter() # do not need to update transition probabilities since that happens automaticaly in bvi_update.
            for t in self.sample_counts[(state, a)].keys():
                if t in self.state_MEC.keys():
                    mec = self.state_MEC[t]
                    mec_sample_counts[mec] = mec_sample_counts.get(mec, 0) + self.sample_counts[(state, a)][t]
                else:
                    assert t not in mec_sample_counts.keys(), "A state that is not part of an MEC cannot appear multiple times in the collapsed sample counts."
                    mec_sample_counts[t] = mec_sample_counts.get(t, 0) + self.sample_counts[(state, a)][t]
            self.sample_counts[(state, a)] = mec_sample_counts

    def collapse(self, mecs):
        """
        Collapse all discovered MECs in the MDP.

        Args:
            mecs (list): List of MECs to collapse; each MEC is in the format (R, B).
        """
        # Step 1: create a super state
        # Step 2: add state super_state mapping
        # Step 3: remove all original states in the super state 
        # Result: we should only be left with a MDP of super states and normal states with NO MECs
        for mec in mecs:
            self.add_super_state(mec)
            
        for s in self.states:
            self.collapse_state_MEC_transitions(s)

        for s in list(self.states):
            if s in self.state_MEC.keys():
                self.remove_state(s)

            

    def add_transition(self, state, action, next_state, probability=None):
        """
        Record a transition in the MDP.
        
        Args:
            state: Source state
            action: Action taken
            next_state: Destination state
            probability: Lower bound transition probability (optional)
        """
        assert (state in self.states), f"The inputted state {state} must alrady exist in the MDP. Please add it before trying to add a transition."
        assert (action in self.topology[state]), f"The inputted action {action} must alrady exist in the MDP for the input state. Please add it before trying to add a transition."
        
        if (state, action) not in self.transition_probabilities:
            self.transition_probabilities[(state, action)] = {}
        if (state, action) not in self.sample_counts:
            self.sample_counts[(state, action)] = Counter()
        
        if probability is not None:
            if (next_state not in self.transition_probabilities[(state, action)]):
                self.transition_probabilities[(state, action)][next_state] = 0
            # If the transition already exists, we update (add to) its probability
            self.transition_probabilities[(state, action)][next_state] += probability

    def update_transition_count(self, state, action, next_state, num_samples=1):
        """
        Record a transition in the MDP.
        
        Args:
            state: Source state
            action: Action taken
            next_state: Destination state
            probability: Lower bound transition probability (optional)
        """
        assert (state in self.states), f"The inputted state {state} must alrady exist in the MDP. Please add it before trying to add a transition."
        assert (action in self.topology[state]), f"The inputted action {action} must alrady exist in the MDP for the input state. Please add it before trying to add a transition."

        curr_state_action_sample_counts = self.sample_counts[(state, action)]
        curr_state_action_sample_counts[next_state] = curr_state_action_sample_counts.get(next_state, 0) + num_samples
    
    def update_transition_probabilities(self):
        for state, action in self.state_action_pairs:
            self._update_transition_probabilities(state, action)

    def _update_transition_probabilities(self, state, action):
        """
        Compute and update lower bound transition probabilities for a state-action
        pair based on observed sample counts.

        Args:
            state: Source state
            action: Action taken
        """
        transition_probability_confidence_error = self._calculate_transition_probability_confidence_error()
        c = self._calculate_transition_probability_confidence_width(state, action, transition_probability_confidence_error, two_sided_hoeffding=True)
        num_state_action_samples = self.get_sample_count(state, action)
        estimated_transition_probabilities = dict()
        for next_state, num_transitions in self.sample_counts[(state, action)].items():
            estimated_transition_probabilities[next_state] = max(0, num_transitions / num_state_action_samples - c)
        self.transition_probabilities[(state, action)] = estimated_transition_probabilities


    def _update_sa_value_bounds(self, state, action, inplace=False): 
        new_sa_value_bounds = [-1, -1]
        if (state in self.goal_states):
            new_sa_value_bounds = [1, 1]
        elif (self.transition_probabilities[(state, action)].keys() == set([state])): # Self loop (for a state or super_state)
            new_sa_value_bounds = [0, 0]
        else:
            new_sa_value_bounds = self.get_state_action_value_bounds(state, action)
        
        if (inplace):
            self.sa_value_bounds[(state, action)] = new_sa_value_bounds
        else:
            return new_sa_value_bounds
    
    def _update_s_value_bounds(self, state, inplace=False):
        new_s_value_bounds = [-1, -1]
        if (state in self.goal_states):
            new_s_value_bounds = [1, 1]
        
        new_s_value_bounds = [-1, -1]
        for a in self.topology[state]:
            if (self.sa_value_bounds[(state, a)][0] > new_s_value_bounds[0]):
                new_s_value_bounds[0] = self.sa_value_bounds[(state, a)][0]
            if (self.sa_value_bounds[(state, a)][1] > new_s_value_bounds[1]):
                new_s_value_bounds[1] = self.sa_value_bounds[(state, a)][1]
        
        assert new_s_value_bounds != [-1, -1], "new_s_value_bounds was not updated."

        if (inplace):
            self.s_value_bounds[state] = new_s_value_bounds
        else:
            return new_s_value_bounds
    
    def update_learned_policy(self):
        for s in self.states:
            if s in self.goal_states:
                continue
            best_action = None
            best_action_value_upper_bound = -1
            for a in self.topology[s]:
                lower_action_value_bound, upper_action_value_bound = self.sa_value_bounds[(s, a)]
                if (upper_action_value_bound > best_action_value_upper_bound):
                    best_action = a
                    best_action_value_upper_bound = upper_action_value_bound
                elif (upper_action_value_bound == best_action_value_upper_bound and
                      lower_action_value_bound > self.sa_value_bounds[(s, best_action)][0]):
                    # Tie-breaker: choose action with higher lower bound
                    best_action = a
            self.learned_policy[s] = best_action

    
    def bvi_update(self):
        # Transition probabilities should already be updated before running bvi_update.
        new_s_value_bounds = dict()
        for s in self.states:
            for a in self.topology[s]:
                # self._update_transition_probabilities(s, a)
                self._update_sa_value_bounds(s, a, inplace=True)
            new_s_value_bounds[s] = self._update_s_value_bounds(s, inplace=False)
        
        self.s_value_bounds = new_s_value_bounds


    def _calculate_transition_probability_confidence_error(self):
        """
        Calculate the acceptable error tolerance for individual transition probabilities

        Returns:
            float: The calculated acceptable error tolerance for individual transition probabilities.
        """
        error_tolerance = (self.confidence_error * self.p_min) / len(self.state_action_pairs)
        return error_tolerance
    
    def _calculate_transition_probability_confidence_width(self, state, action, confidence_error, two_sided_hoeffding=False):
        """
        Calculate the margin of error for an estimated transition probability
        using Chernoff bound (Algorithm 1: ErrorBound).

        Args:
            state: The current state
            action: The current action taken for the given state.
            confidence_error (float): The acceptable error tolerance level (between 0 and 1).
            num_samples (int): The number of samples taken for the state-action pair.
        
        Returns:
            float: The calculated margin of error for the transition probability estimate.
        """
        assert 0 < confidence_error <= 1, f"Confidence error tolerance must be between 0 and 1, got {confidence_error}."
        num_samples = self.get_sample_count(state, action)
        if (num_samples == 0):
            return 1
        if two_sided_hoeffding:
            c = np.sqrt(np.log(2.0 / confidence_error) / (2.0 * num_samples))
        else:
            c = np.sqrt(np.log(1.0 / confidence_error) / (2.0 * num_samples))
        return c
    
    def get_state_action_value_bounds(self, state, action):
        assert state in self.states, f"State {state} is not in the given MDP."
        assert state not in self.goal_states, f"State {state} is a goal state, cannot compute state-action value bounds for goal states. The algorithm should have already terminated."
        assert action in self.topology[state], f"Action {action} is not present for state {state} in the given MDP."
        transitions = self.get_transition_probabilities(state, action)
        lower_state_action_value_bound = sum([probability * self.get_value_bounds(next_state)[0] for next_state, probability in transitions.items()])
        upper_state_action_value_bound = sum([probability * self.get_value_bounds(next_state)[1] for next_state, probability in transitions.items()]) + (1 - sum(transitions.values()))
        assert lower_state_action_value_bound <= upper_state_action_value_bound
        assert upper_state_action_value_bound > 0, f"No transitions found for state-action pair ({state}, {action})."
        return [lower_state_action_value_bound, upper_state_action_value_bound]
    
    def sample_best_action_from_state(self, state):
        assert state in self.states, f"State {state} is not in the given MDP."
        assert len(self.topology[state]) > 0, f"State {state} has no actions."
        best_actions = []
        best_action_value_upper_bound = 0
        for action in self.topology[state]:
            action_value_bounds = self.sa_value_bounds[(state, action)]
            upper_action_value_bound = action_value_bounds[1]
            if (len(best_actions) == 0):
                best_actions = [action]
                _, best_action_value_upper_bound = self.sa_value_bounds[(state, action)]
            elif (upper_action_value_bound > best_action_value_upper_bound):
                best_actions = [action]
                _, best_action_value_upper_bound = self.sa_value_bounds[(state, action)]
            elif (upper_action_value_bound == best_action_value_upper_bound):
                best_actions.append(action)

        chosen_action = np.random.choice(best_actions)
        return chosen_action

    def record_sample(self, state, action, next_state):
        """Record an observed sample/transition."""
        self.update_transition_count(state, action, next_state, 1)
    
    def set_goal_states(self, goal_states):
        """
        Set the goal states for reachability.
        
        Args:
            goal_state (set): The set of all goal states for the MDP
        """
        for state in goal_states:
            self.add_state(state, is_goal=True)
    
    def set_initial_state(self, state):
        """Set the initial/start state."""
        self.add_state(state)
        self.initial_state = state
    
    def get_actions(self, state):
        """
        Get all actions available at a state.
        
        Args:
            state: The state to query.
        
        Returns:
            list: List of available actions at the state.
        """
        return self.topology.get(state, set())
    
    def get_transition_probabilities(self, state, action):
        """
        Get estimated transition probabilities for a state-action pair.
        
        Args:
            state: The source state.
            action: The action taken.
        
        Returns:
            dict: Mapping of next_state to estimated lower bound transition probability.
        """
        return self.transition_probabilities.get((state, action), dict())
    
    def get_sample_count(self, state, action, next_state=None):
        """
        Get sample counts for a transition.
        
        Args:
            state: The source state.
            action: The action taken.
            next_state: The destination state (optional).
        
        Returns:
            int: If next_state is provided, returns the count for that transition.
                 If next_state is None, returns total count for the (state, action) pair.
        """
        if (state not in self.states) or ((state, action) not in self.sample_counts):
            print(f"No samples recorded for state-action pair: ({state}, {action})")
            return 0
        if next_state is None:
            return sum(self.sample_counts[(state, action)].values())
        return self.sample_counts[(state, action)].get(next_state, 0)
    
    def initialize_state_value_bounds(self, state):
        """Initialize value bounds for a state."""
        if (state in self.goal_states):
            self.s_value_bounds[state] = [1, 1]
        else:
            self.s_value_bounds[state] = [0, 1]
    
    def initialize_state_action_value_bounds(self, sa_pair):
        """Initialize value bounds for a state-action pair."""
        self.sa_value_bounds[sa_pair] = [0, 1]
    
    def initialize_mdp_value_bounds(self):
        for state in self.states:
            self.initialize_state_value_bounds(state)
        
        for sa in self.state_action_pairs:
            self.initialize_state_action_value_bounds(sa)
    
    def get_value_bounds(self, state):
        """Get value bounds for a state."""
        # We check if it is already initialized so that we can do an O(1) merge operation.
        if state not in self.s_value_bounds:
            self.initialize_state_value_bounds(state)
        return self.s_value_bounds[state]
    
    def merge(self, other_mdp):
        """
        Merges the current MDP with the inputted MDP.
        """
        self.states = self.states.union(other_mdp.states)
        self.goal_states = self.goal_states.union(other_mdp.goal_states)
        self.state_action_pairs = self.state_action_pairs.union(other_mdp.state_action_pairs)

        for s, a in other_mdp.state_action_pairs:
            self.topology[s] = self.topology.get(s, set())
            self.topology[s].add(a)
            self.sample_counts[(s, a)] = self.sample_counts.get((s, a), Counter()) + other_mdp.sample_counts.get((s, a), Counter()) # dict(new_sa_counts)

    def get_mdp_error(self):
        return self.s_value_bounds[self.initial_state][1] - self.s_value_bounds[self.initial_state][0]

