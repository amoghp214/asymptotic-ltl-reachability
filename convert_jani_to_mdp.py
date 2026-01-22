import json
from typing import Dict, Set, Tuple, Any, List
from mdp import MDP
from tqdm import tqdm
import itertools

variables = None
constants = None
functions = None
transient_variables = None
location_transient_defs = None
system_syncs = None
automata_edges_by_action = None

def convert_jani_to_mdp(jani_file_path: str) -> MDP:
    """
    Convert a JANI file to an MDP object.
    
    Args:
        jani_file_path (str): Path to the JANI file
    
    Returns:
        MDP: An MDP object with all states, actions, and transition probabilities
    """
    # Read and parse the JANI file
    with open(jani_file_path, 'r') as f:
        jani_data = json.load(f)
    
    # Initialize MDP
    mdp = MDP()

    global variables, constants, functions, transient_variables, location_transient_defs, system_syncs, automata_edges_by_action
    variables_list = jani_data.get("variables", [])
    variables = {var["name"]: var for var in variables_list}
    constants = {c["name"]: c.get("value") for c in jani_data.get("constants", [])}
    functions_list = jani_data.get("functions", [])
    functions = {f["name"]: f for f in functions_list}
    
    # Identify transient variables
    transient_variables = {name for name, var in variables.items() if var.get("transient", False)}
    
    # Parse location-specific transient value definitions
    location_transient_defs = _parse_location_transient_defs(jani_data)
    
    # Parse system syncs
    system_syncs = _parse_system_syncs(jani_data)

    # Separate edges into categories based on their actions
    automata_edges_by_action = _separate_edges_by_action(jani_data)

    print("Extracted variables:", variables.keys())
    print("Extracted constants:", constants.keys())
    print("Extracted functions:", functions.keys())
    print("Transient variables:", transient_variables)
    print("System syncs:", system_syncs)
    
    # Extract initial state from the first automaton
    initial_state = _get_initial_state(jani_data)
    print("Initial state:", initial_state)
    if initial_state:
        mdp.set_initial_state(initial_state)
    
    # Extract goal condition from properties
    goal_condition = _extract_goal_condition(jani_data)

    print("Goal condition:", goal_condition)
    
    # Discover all reachable states by exploring from initial state
    all_states = _discover_reachable_states(jani_data, initial_state, goal_condition)
    print("Total reachable states discovered:", len(all_states))
    
    # Add all discovered states to the MDP
    for state in tqdm(all_states, desc="Adding states"):
        # print(state)
        mdp.add_state(state)
    
    # Extract goal states from properties if available
    goal_states = _extract_goal_states(jani_data, all_states, goal_condition)
    if goal_states:
        mdp.set_goal_states(goal_states)
    
    print("Goal states identified:", len(goal_states))

    
    # Extract edges (transitions) and populate the MDP
    _process_automata_edges(jani_data, mdp, all_states)
    
    return mdp


def _separate_edges_by_action(jani_data: Dict) -> Dict[str, List[Dict]]:
    """
    Separate edges from all automata based on their actions.

    Args:
        jani_data (Dict): Parsed JANI file data
    Returns:
        Dict[Dict[str, List[Dict]]]: Mapping of action names to lists of edges - {automaton: {action: [edges]}}
    """
    edges_by_action = {}
    automata = jani_data.get("automata", [])
    
    for automaton in automata:
        aut_name = automaton["name"]
        edges_by_action[aut_name] = {}
        
        for edge in automaton.get("edges", []):
            action = edge.get("action", "silent") # Edges that do not have an action are considered "silent"
            if action not in edges_by_action[aut_name]:
                edges_by_action[aut_name][action] = []
            edges_by_action[aut_name][action].append(edge)
    
    return edges_by_action

def _parse_location_transient_defs(jani_data: Dict) -> Dict[str, Dict[str, Any]]:
    """
    Parse transient value definitions from each location in each automaton.
    Returns a mapping: {automaton_name: {location_name: {var_name: expr}}}
    
    Args:
        jani_data (Dict): Parsed JANI file data
    
    Returns:
        Dict: Mapping of automaton and location to transient variable definitions
    """
    defs = {}
    automata = jani_data.get("automata", [])
    
    for automaton in automata:
        aut_name = automaton["name"]
        defs[aut_name] = {}
        
        for location in automaton.get("locations", []):
            loc_name = location.get("name")
            transient_vals = location.get("transient-values", [])
            
            loc_defs = {}
            for tv in transient_vals:
                var_name = tv.get("ref")
                value_expr = tv.get("value")
                loc_defs[var_name] = value_expr
            
            if loc_defs:
                defs[aut_name][loc_name] = loc_defs
    
    return defs


def _parse_system_syncs(jani_data: Dict) -> Dict[str, List[str]]:
    """
    Parse the system syncs to map actions to required automata.
    If no syncs are defined, return empty dict (spontaneous transitions).
    
    Args:
        jani_data (Dict): Parsed JANI file data
    
    Returns:
        Dict[str, List[str]]: Maps action name to list of automata that must participate
    """
    syncs = {}
    system = jani_data.get("system", {})
    sync_list = system.get("syncs", [])
    
    for sync in sync_list:
        action = sync.get("result")
        synchronise = sync.get("synchronise", [])
        # Map action to the automata that must synchronize (based on sync indices)
        # synchronise is a list matching the order of automata
        syncs[action] = synchronise
    
    return syncs

def _r_find_goal_condition_in_expr(expr: Any) -> Any:
    """
    Recursively search an expression for a sub-expression of the form:
        {"left": True, "op": "U", "right": <goal>}
    Return the first <goal> found or None if not present.
    """
    if isinstance(expr, dict):
        # direct match
        if expr.get("op") == "U" and expr.get("left") is True:
            return expr.get("right")
        # recurse into all dict values
        for v in expr.values():
            if isinstance(v, dict):
                found = _r_find_goal_condition_in_expr(v)
                if found is not None:
                    return found
    return None

def _extract_goal_condition(jani_data: Dict) -> Any:
    """
    Extract the goal condition from JANI properties.
    
    Args:
        jani_data (Dict): Parsed JANI file data
    
    Returns:
        Any: The goal condition expression, or None if not found
    """
    properties = jani_data.get("properties", [])
    goal_condition = None
    
    for prop in properties:
        expression = prop.get("expression", {})
        
        if expression.get("op") == "filter":
            values_exp = expression.get("values", {})
            if (isinstance(values_exp, dict)):# and
                # For "true U condition", return the right side which is the goal condition
                print(prop)
                extracted_goal_condition = _r_find_goal_condition_in_expr(values_exp)
                if goal_condition is None and extracted_goal_condition is not None:
                    goal_condition = extracted_goal_condition
                elif goal_condition is not None and extracted_goal_condition is not None:
                    goal_condition = {
                        "op": "∨",
                        "left": goal_condition,
                        "right": extracted_goal_condition
                    }

    return goal_condition


def _get_initial_state(jani_data: Dict) -> Tuple:
    """
    Extract the initial state from JANI data, including initial locations, 
    all variables (including transient), excluding only uninitialized transient variables.
    
    Args:
        jani_data (Dict): Parsed JANI file data
    
    Returns:
        Tuple: A tuple representing the initial state (all variables + locations)
    """
    # Get initial values of all variables
    initial_values = {}
    
    if "variables" in jani_data:
        for var in jani_data["variables"]:
            var_name = var["name"]
            assert "initial-value" in var, f"Variable {var_name} has no initial value."
            initial_values[var_name] = var["initial-value"]
    
    # Add initial locations from automata and compute transient values
    automata = jani_data.get("automata", [])
    for automaton in automata:
        automaton_name = automaton["name"]
        initial_locations = automaton["initial-locations"]
        assert len(initial_locations) == 1, f"Automaton {automaton_name} has multiple initial locations."
        location = initial_locations[0] # TODO: assert that there is only 1 initial location
        location_var_name = f"__location_{automaton_name}"
        initial_values[location_var_name] = location
        
        # Compute transient variables for this location
        if automaton_name in location_transient_defs and location in location_transient_defs[automaton_name]:
            loc_trans_defs = location_transient_defs[automaton_name][location]
            for var_name, expr in loc_trans_defs.items():
                initial_values[var_name] = expr
    
    # Evaluate all expressions
    for iv, val in initial_values.items():
        if isinstance(val, dict):
            initial_values[iv] = _evaluate_expression(val, initial_values)
        assert not isinstance(initial_values[iv], dict), f"Initial value of variable {iv} is still an expression: {initial_values[iv]}"
    
    # Convert to tuple for hashability
    assert len(initial_values) > 0, "No initial values found."
    return tuple(sorted(initial_values.items()))


def _find_synchronized_edges(state_dict: Dict, automata: List, automaton_names: List[str],
                                             action_name: str, sync_list: List) -> List[Tuple]:
    """
    For a given global action and its sync list, return all possible edge tuples
    (one entry per automaton, 'None' if the automaton does not participate).
    Each tuple index corresponds to the automaton at the same index in `automata`.

    Returns an empty list if the synchronization cannot occur in the given state.
    """
    edge_options = []

    for idx, automaton in enumerate(automata):
        # Determine required local action for this automaton from the sync list.
        req = None
        if idx < len(sync_list):
            req = sync_list[idx]

        # Treat non-string or special markers as "not participating"
        if not isinstance(req, str) or req in ("", "*", "-"): # TODO: double check this
            edge_options.append([None])
            continue

        # Find candidate edges that:
        #  - are enabled at the current location,
        #  - have the required local action,
        #  - whose guard is satisfied in the current state.
        aut_name = automaton_names[idx]
        current_location = state_dict.get(f"__location_{aut_name}")
        candidates = []

        for edge in automata_edges_by_action[aut_name].get(req, []):
            if (edge.get("location") == current_location and 
                _is_guard_satisfied(edge.get("guard"), state_dict)):
                candidates.append(edge)

        # If this automaton is required to participate but no candidate exists, the sync cannot fire.
        if not candidates:
            return []

        edge_options.append(candidates)

    # Cartesian product of candidate edges (one choice per automaton) -> list of tuples
    edge_tuples = [tuple(prod) for prod in itertools.product(*edge_options)]
    return edge_tuples

def _find_silent_edges(state_dict: Dict, automata: List, automaton_names: List[str]) -> List[Tuple]:
    """
    Return all possible edge tuples that correspond to silent/spontaneous transitions.

    Behavior:
      - A silent transition is a local edge whose action is missing or marked "silent".
      - Only one automaton participates in a silent transition; other automata are None.
      - For each enabled silent edge in any automaton we return a tuple where that index
        contains the edge and all other indices are None.
      - If no enabled silent edges exist in the global state, return an empty list.

    Args:
        state_dict (Dict): Current global state as a dict
        automata (List): List of automata (each as parsed from JANI)
        automaton_names (List[str]): Names of the automata

    Returns:
        List[Tuple]: List of edge-tuples (one entry per automaton, None for non-participating)
    """
    edge_tuples = []

    for idx, automaton in enumerate(automata):
        aut_name = automaton_names[idx]
        current_location = state_dict.get(f"__location_{aut_name}")

        for edge in automata_edges_by_action[aut_name].get("silent", []):
            if (edge.get("location") == current_location and
                _is_guard_satisfied(edge.get("guard"), state_dict)):
                # Build tuple with this edge at position idx and None elsewhere
                tuple_slots = [None] * len(automata)
                tuple_slots[idx] = edge
                edge_tuples.append(tuple(tuple_slots))

    return edge_tuples

def _find_destinations(edge_tuple: List[Tuple], sync_list: List[str], sync_action: bool) -> List[Tuple]:
    """
    For a given edge tuple and its sync list, return all possible destination tuples
    (one entry per automaton, 'None' if the automaton does not participate).
    Each tuple index corresponds to the automaton at the same index in `automata`.

    Returns an empty list if there are no edge tuples inputted.
    """
    # for each edge_tuple, create a list of possible destination lists and return destination tuples
    if not edge_tuple:
        return []

    dest_options = []
    for idx, edge in enumerate(edge_tuple):
        # If this automaton was required to participate but the provided edge is None, the sync cannot fire.
        if edge is None:
            if (sync_action and sync_list[idx]):
                return []
            elif (sync_action and not sync_list[idx]):
                dest_options.append([None])
                continue
            elif (not sync_action):
                dest_options.append([None])
                continue

        # Collect destinations for this edge
        destinations = edge.get("destinations", [])
        if not destinations:
            # No destinations -> cannot fire
            return []

        # If there's a single destination and no explicit probability, assume probability 1.0
        if len(destinations) == 1 and "probability" not in destinations[0]:
            destinations[0]['probability'] = {'exp': 1.0}

        dest_options.append(destinations)

    # Cartesian product of destination choices (one destination per automaton) -> list of tuples
    dest_tuples = [tuple(prod) for prod in itertools.product(*dest_options)]
    return dest_tuples

def _discover_reachable_states(jani_data: Dict, initial_state: Tuple, goal_condition: Any) -> Set[Tuple]:
    """
    Discover all reachable states by doing a breadth-first search from the initial state,
    following edges defined in the JANI automata with proper synchronization.
    Handles both synchronized edges (from syncs) and spontaneous edges.
    Goal states are not expanded.
    
    Args:
        jani_data (Dict): Parsed JANI file data
        initial_state (Tuple): The initial state
        goal_condition (Any): The goal condition expression
    
    Returns:
        Set[Tuple]: Set of all reachable states (including transient variables)
    """
    automata = jani_data.get("automata", [])
    automaton_names = [a["name"] for a in automata]
    discovered_states = set()
    assert initial_state is not None, "Initial state is None."
    queue = [initial_state]
    
    while queue:
        state = queue.pop(0)
        
        if state in discovered_states:
            continue
        
        discovered_states.add(state)
        state_dict = dict(state)

        # NOTE: idk if this is needed or not since some PRISM MDP need it and others do not
        # if (goal_condition is not None and _evaluate_expression(goal_condition, state_dict)):
        #     # Do not expand goal states
        #     continue
        
        # Process synchronized actions
        for action_name, sync_list in system_syncs.items():
            # Find all synchronized edge combinations for this action
            edge_tuples = _find_synchronized_edges(state_dict, automata, automaton_names, action_name, sync_list)
            for edge_tuple in edge_tuples:
                dest_tuples = _find_destinations(edge_tuple, sync_list, sync_action=True)
                for dest_tuple in dest_tuples:
                    next_state = _get_next_state(state_dict, dest_tuple, jani_data)
                    if next_state not in discovered_states:
                        queue.append(next_state)

        # Process spontaneous/silent edges: each automaton independently
        silent_edges = _find_silent_edges(state_dict, automata, automaton_names)
        for edge_tuple in silent_edges:
            dest_tuples = _find_destinations(edge_tuple, [], sync_action=False)
            for dest_tuple in dest_tuples:
                next_state = _get_next_state(state_dict, dest_tuple, jani_data)
                if next_state not in discovered_states:
                    queue.append(next_state)
                    
    return discovered_states
    
def _compute_transient_values(state_dict: Dict, assignments: List, automaton_names: List[str]) -> Dict:
    """
    Compute and populate transient variables in the provided state dictionary.

    Priority for each transient variable:
        1) If the current location of an automaton defines a transient value (via
            location_transient_defs), use that (and evaluate it).
        2) Else if the variable is getting assigned in the current transition, use that value.
        3) Else use the variable's "initial-value" from the variable definition (if any).

    Args:
        state_dict (Dict): Current state as a dictionary (may or may not contain transient vars)
        assignments (List): List of assignments in the current transition
        automaton_names (List[str]): Names of the automata (used to find current locations)

    Returns:
        Dict: The state_dict with transient variables set/evaluated.
    """
    global transient_variables, location_transient_defs, variables

    # Ensure we have a working dict to mutate
    if state_dict is None:
        state_dict = {}

    # For each transient variable, determine value by priority described above
    for tvar in transient_variables:
        applied = False

        # 1) Check for a location-specific transient definition
        if location_transient_defs:
            for aut_name in automaton_names:
                loc_var = f"__location_{aut_name}"
                loc = state_dict.get(loc_var)
                if loc is None:
                    continue
                aut_defs = location_transient_defs.get(aut_name, {})
                loc_defs = aut_defs.get(loc, {})
                if tvar in loc_defs:
                    expr = loc_defs[tvar]
                    if isinstance(expr, (dict, str)):
                        state_dict[tvar] = _evaluate_expression(expr, state_dict)
                    else:
                        state_dict[tvar] = expr
                    applied = True
                    break

        if applied:
            continue

        # 2) If the variable is assigned in the current transition, use that value
        for assignment in assignments:
            var_ref = assignment.get("ref")
            if var_ref == tvar:
                value = assignment.get("value")
                if isinstance(value, (dict, str)):
                    state_dict[tvar] = _evaluate_expression(value, state_dict)
                else:
                    state_dict[tvar] = value
                applied = True
                break
        if applied:
            continue
        
        # 3) Fall back to the variable's initial-value (if provided)
        tdef = variables.get(tvar, {})
        if "initial-value" in tdef:
            init_expr = tdef["initial-value"]
            if isinstance(init_expr, (dict, str)):
                state_dict[tvar] = _evaluate_expression(init_expr, state_dict)
            else:
                state_dict[tvar] = init_expr
        else:
            # No definition available; set to None to be explicit
            state_dict[tvar] = None

    return state_dict


def _apply_edge_tuple_destinations(state_dict: Dict, automata: List, automaton_names: List[str],
                                   edge_tuple: Tuple, discovered_states: Set, queue: List) -> None:
    """
    Helper function to apply destinations from an edge tuple and add to queue.
    
    Args:
        state_dict (Dict): Current state dictionary
        automata (List): List of automata
        automaton_names (List[str]): Names of automata
        edge_tuple (Tuple): Tuple of edges (one per automaton or None)
        discovered_states (Set): Set of discovered states
        queue (List): Queue of states to process
    """
    for automaton_idx, edge in enumerate(edge_tuple):
        if edge is None:
            continue
        
        automaton = automata[automaton_idx]
        automaton_name = automaton_names[automaton_idx]
        location_var_name = f"__location_{automaton_name}"
        
        destinations = edge.get("destinations", [])
        
        for dest in destinations:
            next_state_dict = state_dict.copy()
            
            assert "location" in dest.keys(), f"Destination in automaton {automaton_name} missing location."
            dest_location = dest["location"]
            next_state_dict[location_var_name] = dest_location
            
            assignments = dest.get("assignments", [])
            
            for assignment in assignments:
                var_ref = assignment.get("ref")
                if var_ref in transient_variables:
                    continue
                value = assignment.get("value")
                next_state_dict[var_ref] = value
                if isinstance(value, dict) or isinstance(value, str):
                    next_state_dict[var_ref] = _evaluate_expression(value, state_dict)
            
            # Recompute transient values for new location
            next_state_dict = _compute_transient_values(next_state_dict, assignments, automaton_names)
            next_state = tuple(sorted(next_state_dict.items()))
            
            if next_state not in discovered_states:
                queue.append(next_state)

def _get_next_state(state_dict: Dict, dest_tuple: Tuple, jani_data: Dict) -> None:
    """
    Create the next state from the current state and a tuple of destinations,
    and add it to the queue if not already discovered.
    
    Args:
        state_dict (Dict): Current state dictionary
        dest_tuple (Tuple): Tuple of destinations (one per automaton)
        discovered_states (Set): Set of discovered states
        queue (List): Queue of states to process
    """
    next_state_dict = state_dict.copy()
    automata = jani_data.get("automata", [])
    automaton_names = [a["name"] for a in automata]
    
    for automaton_idx, dest in enumerate(dest_tuple):
        if dest is None:
            continue
        
        automaton_name = automaton_names[automaton_idx]
        location_var_name = f"__location_{automaton_name}"
        
        assert "location" in dest.keys(), f"Destination in automaton {automaton_name} missing location."
        dest_location = dest["location"]
        next_state_dict[location_var_name] = dest_location
        
        assignments = dest.get("assignments", [])
        
        for assignment in assignments:
            var_ref = assignment.get("ref")
            if var_ref in transient_variables:
                continue
            value = assignment.get("value")
            next_state_dict[var_ref] = value
            if isinstance(value, dict) or isinstance(value, str):
                next_state_dict[var_ref] = _evaluate_expression(value, state_dict)
    
    # Recompute transient values for new locations
    next_state_dict = _compute_transient_values(next_state_dict, assignments, automaton_names)
    for k, v in next_state_dict.items():
        assert not isinstance(v, dict), f"Variable {k} in next state is still an expression: {v}"
    next_state = tuple(sorted(next_state_dict.items()))
    return next_state


def _extract_goal_states(jani_data: Dict, all_states: Set[Tuple], goal_condition: Any) -> Set[Tuple]:
    """
    Extract goal states from the JANI properties.
    
    For reachability properties of the form "F (condition)", the goal states are those
    where the condition evaluates to true.
    
    Args:
        jani_data (Dict): Parsed JANI file data
        all_states (Set[Tuple]): Set of all reachable states
        goal_condition (Any): The goal condition expression
    
    Returns:
        Set[Tuple]: Set of goal states
    """
    goal_states = set()
    
    # If no goal condition was found, return empty set
    if goal_condition is None:
        return goal_states
    
    # Evaluate the goal condition against all reachable states
    for state in tqdm(all_states, desc="Evaluating goal states", leave=False, total=len(all_states)):
        state_dict = dict(state)
        is_goal = _evaluate_expression(goal_condition, state_dict)
        assert isinstance(is_goal, bool), f"Goal condition did not evaluate to a boolean: {is_goal}"
        if is_goal:
            goal_states.add(state)
    
    return goal_states


def _process_automata_edges(jani_data: Dict, mdp: MDP, all_states: Set[Tuple]) -> None:
    """
    Process all automata edges and populate the MDP with actions and transitions.
    Handles both synchronized transitions and spontaneous transitions.
    
    Args:
        jani_data (Dict): Parsed JANI file data
        mdp (MDP): MDP object to populate
        all_states (Set[Tuple]): Set of all reachable states
    """
    automata = jani_data.get("automata", [])
    automaton_names = [a["name"] for a in automata]
    has_syncs = bool(system_syncs)
    
    # For each state, find all possible actions
    for state in tqdm(all_states, desc="Processing states", total=len(all_states)):
        state_dict = dict(state)
        
        # Process synchronized actions
        for action_name, sync_list in system_syncs.items():
            edge_tuples = _find_synchronized_edges(state_dict, automata, automaton_names, action_name, sync_list)
            
            # For each valid edge tuple, generate transitions
            for edge_tuple_idx, edge_tuple in enumerate(edge_tuples):
                action = f"{action_name}_{edge_tuple_idx}" if len(edge_tuples) > 1 else action_name
                
                mdp.add_action_to_state(state, action)
                
                # Collect all destinations from all edges in the tuple
                # Use _find_destinations to enumerate concrete destination tuples for this synchronized edge tuple,
                # then for each destination-tuple compute the resulting next state via _get_next_state and add the
                # transition to the MDP (product of per-automaton probabilities).
                dest_tuples = _find_destinations(edge_tuple, sync_list, sync_action=True)
                assert dest_tuples, f"No destinations found for edge tuple: {edge_tuple}"

                total_prob = 0.0
                for dest_tuple in dest_tuples:
                    # compute overall probability as product of per-automaton destination probabilities
                    prob = 1.0
                    for dest in dest_tuple:
                        if dest is None:
                            continue
                        assert "probability" in dest, f"Destination missing probability: {dest}"
                        assert "exp" in dest["probability"], f"Destination probability missing 'exp': {dest}"
                        pexp = dest["probability"]["exp"]
                        prob *= _evaluate_expression(pexp, state_dict)

                    # compute the concrete next state for this destination tuple
                    next_state = _get_next_state(state_dict, dest_tuple, jani_data)
                    assert next_state is not None, "Next state is None."
                    assert next_state in all_states, "Next state not in all_states."
                    mdp.add_transition(state, action, next_state, prob)
                    total_prob += prob

                assert total_prob == 1.0, f"Total probability for action {action} from state {state} does not sum to 1 (sum={total_prob})"

        # Process spontaneous edges: each automaton independently
        silent_edges = _find_silent_edges(state_dict, automata, automaton_names)
        for edge_tuple_idx, edge_tuple in enumerate(silent_edges):
            action = f"silent_{edge_tuple_idx}" if len(silent_edges) > 1 else 'silent'
                
            mdp.add_action_to_state(state, action)
            dest_tuples = _find_destinations(edge_tuple, [], sync_action=False)
            
            assert dest_tuples, f"No destinations found for edge tuple: {edge_tuple}"

            total_prob = 0.0
            for dest_tuple in dest_tuples:
                # compute overall probability as product of per-automaton destination probabilities
                prob = 1.0
                for dest in dest_tuple:
                    if dest is None:
                        continue
                    assert "probability" in dest, f"Destination missing probability: {dest}"
                    assert "exp" in dest["probability"], f"Destination probability missing 'exp': {dest}"
                    pexp = dest["probability"]["exp"]
                    prob *= _evaluate_expression(pexp, state_dict)

                # compute the concrete next state for this destination tuple
                next_state = _get_next_state(state_dict, dest_tuple, jani_data)
                assert next_state is not None, "Next state is None."
                assert next_state in all_states, "Next state not in all_states."
                mdp.add_transition(state, action, next_state, prob)
                total_prob += prob

            assert abs(total_prob - 1.0) < 1e-8, f"Total probability for action {action} from state {state} does not sum to 1 (sum={total_prob})"


def _is_guard_satisfied(guard: Dict, state: Dict) -> bool:
    """
    Check if a guard condition is satisfied in a given state.
    
    Args:
        guard (Dict): Guard condition expression
        state (Dict): Current state as a dictionary (with transient vars already computed)
    
    Returns:
        bool: True if the guard is satisfied, False otherwise
    """
    if guard is None or guard == {}:
        return True
    
    exp = guard.get("exp")
    
    if exp is None:
        return True
    
    if isinstance(exp, bool):
        return exp
    
    evaluated_exp = _evaluate_expression(exp, state)
    assert isinstance(evaluated_exp, bool), f"Guard expression did not evaluate to a boolean: {evaluated_exp}"
    return evaluated_exp


def _evaluate_expression(expr: Any, state: Dict) -> Any:
    """
    Evaluate an expression in the context of a state.
    Transient variables are already stored in the state dictionary.
    
    Args:
        expr (Any): Expression to evaluate
        state (Dict): Current state as a dictionary (with transient vars)
    
    Returns:
        Any: The evaluated value
    """
    if isinstance(expr, (int, float, bool)):
        return expr
    
    if isinstance(expr, str):
        # First check if it's a state variable (including transient if already computed)
        if expr in state:
            return state[expr]
        
        # Then check provided constants
        if constants is not None and expr in constants.keys():
            return constants[expr]
        
        # fallback
        assert False, f"Unknown variable or constant: {expr}"
    
    if isinstance(expr, dict):
        op = expr.get("op")
        
        # Handle unary operators first
        if op == "¬":
            operand = _evaluate_expression(expr.get("exp"), state)
            return not operand
        
        if op == "call":
            func_name = expr["function"]
            func = functions[func_name]
            # NOTE: For simplicity, assume no parameters for now. Current MDPs do not pass in any func args (pacman.v2).
            func_body = func.get("body")
            return _evaluate_expression(func_body, state)
        
        if op == "ite":
            condition = _evaluate_expression(expr['if'], state)
            assert isinstance(condition, bool)
            if (condition):
                return _evaluate_expression(expr['then'], state)
            else:
                return _evaluate_expression(expr['else'], state)
        
        left = expr.get("left")
        right = expr.get("right")
        
        # Handle binary operators
        left_val = _evaluate_expression(left, state)
        right_val = _evaluate_expression(right, state)

        if op == "+":
            return left_val + right_val
        elif op == "-":
            return left_val - right_val
        elif op == "*":
            return left_val * right_val
        elif op == "/":
            return left_val / right_val if right_val != 0 else 0
        elif op == "=":
            return left_val == right_val
        elif op == "!=":
            return left_val != right_val
        elif op == "<":
            return left_val < right_val
        elif op == "<=":
            return left_val <= right_val
        elif op == "≤":
            return left_val <= right_val
        elif op == ">":
            return left_val > right_val
        elif op == ">=":
            return left_val >= right_val
        elif op == "≥":
            return left_val >= right_val
        elif op == "≠":
            return left_val != right_val
        elif op == "∧":
            return left_val and right_val
        elif op == "∨":
            return left_val or right_val
        elif op == "min":
            return min(left_val, right_val)
        elif op == "max":
            return max(left_val, right_val)
    
    assert False, f"Unsupported expression: {expr}"
    return 0


if __name__ == "__main__":
    # Example usage
    jani_file = "mdp_models/ij.10.v1.jani"
    # jani_file = "mdp_models/philosophers-mdp.3.v1.jani"
    # jani_file = "mdp_models/csma.2-2.v1.jani"
    # jani_file = "mdp_models/zeroconf.v1.jani"
    # jani_file = "mdp_models/consensus.2.v1.jani"
    # jani_file = "mdp_models/firewire_abst.v1.jani"
    # jani_file = "mdp_models/pacman.v2.jani"
    # jani_file = "mdp_models/rabin.3.v1.jani"

    mdp = convert_jani_to_mdp(jani_file)
    
    print(f"Number of states: {len(mdp.states)}")
    print(f"Initial state: {mdp.initial_state}")
    print(f"Number of goal states: {len(mdp.goal_states)}")
    print(f"Number of state-action pairs: {len(mdp.state_action_pairs)}")
    print(f"Number of transitions: {sum(len(dests) for dests in mdp.transition_probabilities.values())}")
    
    # Print some example states and transitions
    if mdp.states:
        sample_state = list(mdp.states)[0]
        print(f"\nSample state: {sample_state}")
        if sample_state in mdp.topology:
            print(f"Actions available: {mdp.topology[sample_state]}")
            for action in mdp.topology[sample_state]:
                transitions = mdp.get_transition_probabilities(sample_state, action)
                print(f"  Action {action}: {transitions}")
    
    smallest_transition = 1
    for (s, a), dests in mdp.transition_probabilities.items():
        for (next_s, prob) in dests.items():
            if prob < smallest_transition and prob > 0:
                smallest_transition = prob
    print("Smallest non-zero transition probability:", smallest_transition)
