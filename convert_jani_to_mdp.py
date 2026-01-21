import json
from typing import Dict, Set, Tuple, Any
from mdp import MDP
from tqdm import tqdm


variables = None
constants = None
functions = None


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

    global variables, constants, functions
    variables_list = jani_data.get("variables", [])
    variables = {var["name"]: var for var in variables_list}
    constants = {c["name"]: c.get("value") for c in jani_data.get("constants", [])}
    functions_list = jani_data.get("functions", [])
    functions = {f["name"]: f for f in functions_list}
    
    # Extract initial state from the first automaton
    initial_state = _get_initial_state(jani_data)
    print("Initial state:", initial_state)
    if initial_state:
        mdp.set_initial_state(initial_state)
    
    # Extract all states and build a mapping from state tuple to state representation
    all_states = _enumerate_all_states(jani_data)
    
    # Add all states to the MDP
    for state in tqdm(all_states, desc="Adding states"):
        mdp.add_state(state)
    
    # Extract goal states from properties if available
    goal_states = _extract_goal_states(jani_data, all_states)
    if goal_states:
        mdp.set_goal_states(goal_states)
    
    # Extract edges (transitions) and populate the MDP
    _process_automata_edges(jani_data, mdp, all_states)
    
    return mdp


def _get_initial_state(jani_data: Dict) -> Tuple:
    """
    Extract the initial state from JANI data.
    
    Args:
        jani_data (Dict): Parsed JANI file data
    
    Returns:
        Tuple: A tuple representing the initial state (variable_values)
    """
    # Get initial values of all variables
    initial_values = {}
    
    if "variables" in jani_data:
        for var in jani_data["variables"]:
            var_name = var["name"]
            if "initial-value" in var:
                initial_values[var_name] = var["initial-value"]
    
    for iv, val in initial_values.items():
        if (isinstance(val, dict)):
            initial_values[iv] = _evaluate_expression(val, initial_values)
        assert not isinstance(initial_values[iv], dict), f"Initial value of variable {iv} is still an expression: {initial_values[iv]}"
    
    # Convert to tuple for hashability
    if initial_values:
        return tuple(sorted(initial_values.items()))
    
    return None


def _enumerate_all_states(jani_data: Dict) -> Set[Tuple]:
    """
    Enumerate all possible states by computing the Cartesian product of variable domains.
    
    Args:
        jani_data (Dict): Parsed JANI file data
    
    Returns:
        Set[Tuple]: Set of all possible states as tuples of (variable_name, value) pairs
    """
    # variables = jani_data.get("variables", [])
    # constants = {c["name"]: c.get("value") for c in jani_data.get("constants", [])}
    # print("Variables:", variables)
    # print("Constants:", constants)
    
    if not variables:
        return {()}
    
    # Extract the domain of each variable
    var_domains = {}
    for var_name, var in tqdm(variables.items(), desc="Enumerating variables"):
        var_type = var.get("type", {})
        
        # Handle string type (e.g., "bool")
        if isinstance(var_type, str):
            if var_type == "bool":
                var_domains[var_name] = [True, False]
            else:
                # For other string types, use initial value if available
                if "initial-value" in var:
                    var_domains[var_name] = [var["initial-value"]]
                if var_type == "real":
                    var_domains[var_name] = var_domains.get(var_name, []) + [1] # This is a hack to handle the 'steps' variable in consensus.2.v1.jani
                # else:
                #     var_domains[var_name] = [0]
        # Handle dictionary type
        elif isinstance(var_type, dict):
            if var_type.get("kind") == "bounded":
                lower = var_type.get("lower-bound", 0)
                upper = var_type.get("upper-bound", 0)
                
                # Handle case where bounds might be constant references
                if isinstance(lower, str):
                    lower = constants[lower]
                if isinstance(upper, str):
                    upper = constants[upper]

                if isinstance(lower, dict):
                    lower = _evaluate_expression(lower, {})
                if isinstance(upper, dict):
                    upper = _evaluate_expression(upper, {})
                
                assert not isinstance(lower, dict) and not isinstance(upper, dict), f"Bounds for variable {var_name} must not be dicts."
                
                print(f"Variable {var_name} has bounded type with range [{lower}, {upper}]")
                var_domains[var_name] = list(range(lower, upper + 1))
            else:
                # Default: use initial value
                if "initial-value" in var:
                    var_domains[var_name] = [var["initial-value"]]
                else:
                    var_domains[var_name] = [0]
        else:
            # Fallback: use initial value
            if "initial-value" in var:
                var_domains[var_name] = [var["initial-value"]]
            # else:
            #     var_domains[var_name] = [0]
    
    # Generate all combinations (Cartesian product)
    all_states = set()
    _generate_state_combinations(var_domains, list(var_domains.keys()), 0, {}, all_states)
    
    return all_states


def _generate_state_combinations(
    var_domains: Dict[str, list],
    var_names: list,
    index: int,
    current_state: Dict,
    all_states: Set[Tuple]
) -> None:
    """
    Recursively generate all possible state combinations.
    
    Args:
        var_domains (Dict): Mapping of variable names to their possible values
        var_names (list): List of variable names
        index (int): Current index in var_names
        current_state (Dict): Current partial state being built
        all_states (Set): Set to accumulate all states
    """
    if index == len(var_names):
        # Convert to tuple and add to all_states
        state_tuple = tuple(sorted(current_state.items()))
        all_states.add(state_tuple)
        return
    
    var_name = var_names[index]
    for value in var_domains[var_name]:
        current_state[var_name] = value
        _generate_state_combinations(var_domains, var_names, index + 1, current_state, all_states)
    
    del current_state[var_name]


def _extract_goal_states(jani_data: Dict, all_states: Set[Tuple]) -> Set[Tuple]:
    """
    Extract goal states from the JANI properties.
    
    For reachability properties of the form "F (condition)", the goal states are those
    where the condition evaluates to true.
    
    Args:
        jani_data (Dict): Parsed JANI file data
        all_states (Set[Tuple]): Set of all possible states
    
    Returns:
        Set[Tuple]: Set of goal states
    """
    goal_states = set()
    
    properties = jani_data.get("properties", [])
    
    for i, prop in enumerate(tqdm(properties, desc="Processing properties")):
        expression = prop.get("expression", {})
        
        # Extract the goal condition from the property expression
        # For properties with a "filter" operation, the actual condition is in "values"
        goal_condition = None
        
        if expression.get("op") == "filter":
            # The goal condition is the right side of the "U" (Until) operator
            # print("Expression:", expression)
            values_exp = expression.get("values", {}).get("exp")
            # print("Values expression:", values_exp)
            if isinstance(values_exp, dict) and values_exp.get("op") == "U":
                # For "true U condition", we want the right side which is the goal condition
                goal_condition = values_exp.get("right")
        
        # If we found a goal condition, evaluate it against all states
        if goal_condition:
            for state in tqdm(all_states, desc=f"Evaluating goal states (prop {i})", leave=False, total=len(all_states)):
                state_dict = dict(state)
                is_goal = _evaluate_expression(goal_condition, state_dict)
                assert isinstance(is_goal, bool), f"Goal condition did not evaluate to a boolean: {is_goal}"
                if is_goal:
                    goal_states.add(state)
    
    return goal_states


def _process_automata_edges(jani_data: Dict, mdp: MDP, all_states: Set[Tuple]) -> None:
    """
    Process all automata edges and populate the MDP with actions and transitions.
    Uses the MDP class helper methods to properly populate all fields.
    
    Args:
        jani_data (Dict): Parsed JANI file data
        mdp (MDP): MDP object to populate
        all_states (Set[Tuple]): Set of all possible states
    """
    automata = jani_data.get("automata", [])
    # variables = {var["name"]: var for var in jani_data.get("variables", [])}
    
    # For each state, find all possible edges (transitions)
    for state in tqdm(all_states, desc="Processing states", total=len(all_states)):
        state_dict = dict(state)
        
        # Check which automata can make transitions from this state
        for automaton in automata:
            edges = automaton.get("edges", [])
            
            for edge_idx, edge in enumerate(edges):
                # Check if the guard condition is satisfied in this state
                if _is_guard_satisfied(edge.get("guard"), state_dict, jani_data.get("constants", [])):
                    action = f"{automaton['name']}_edge_{edge_idx}"
                    
                    # Add action to the state using MDP helper method
                    mdp.add_action_to_state(state, action)
                    
                    # Process all destinations (probabilistic transitions)
                    destinations = edge.get("destinations", [])
                    
                    if destinations:
                        # In case there is only one destination and the probability is not specified, we assume probability 1
                        if (len(destinations) == 1):
                            destinations[0]['probability'] = {'exp': 1.0}
                        # total_probability = sum(d.get("probability", {}).get("exp", 0) for d in destinations)
                        total_probability = sum(_evaluate_expression(d.get("probability", {}).get("exp", 0), {}) for d in destinations)
                        assert total_probability == 1, f"Total probability of destinations must sum to 1. Got {total_probability} for edge {edge_idx} in automaton {automaton['name']}"
                        
                        for dest in destinations:
                            # Apply assignments to get the next state
                            next_state_dict = state_dict.copy()
                            assignments = dest.get("assignments", [])
                            
                            for assignment in assignments:
                                var_ref = assignment.get("ref")
                                value = assignment.get("value")
                                next_state_dict[var_ref] = value
                                for iv, val in next_state_dict.items():
                                    if (isinstance(val, dict)):
                                        next_state_dict[iv] = _evaluate_expression(val, state_dict)
                                    assert not isinstance(next_state_dict[iv], dict), f"Variable {iv} in next state is still an expression: {next_state_dict[iv]}"
                            
                            # Convert next_state_dict to tuple
                            next_state = tuple(sorted(next_state_dict.items()))
                            
                            # Get probability
                            prob_exp = dest.get("probability", {}).get("exp", 0)
                            if isinstance(prob_exp, (int, float)):
                                probability = prob_exp
                            else:
                                print("SHOULD NOT BE USED")
                                probability = 1.0 / len(destinations) if destinations else 0
                            
                            # Normalize probability
                            if total_probability > 0:
                                probability = prob_exp / total_probability
                            
                            # Add transition using MDP helper method
                            mdp.add_transition(state, action, next_state, probability)
                            # if ((state, action) == ((('num_tokens_var', 0), ('q1', 1), ('q2', 1), ('q3', 1)), 'process2_edge_0')):
                            #     for d in destinations:
                            #         print("Prob:", d.get("probability", {}))
                            #     print("State:", state)
                            #     print("Action:", action)
                            #     print("Next State:", next_state)
                            #     print("Transition:", mdp.transition_probabilities[(state, action)])


def _is_guard_satisfied(guard: Dict, state: Dict, constants: list = []) -> bool:
    """
    Check if a guard condition is satisfied in a given state.
    
    Args:
        guard (Dict): Guard condition expression
        state (Dict): Current state as a dictionary
        constants (list): List of constants available for evaluation
    
    Returns:
        bool: True if the guard is satisfied, False otherwise
    """
    if guard is None:
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
    
    Args:
        expr (Any): Expression to evaluate
        state (Dict): Current state as a dictionary
    
    Returns:
        Any: The evaluated value
    """
    # if (isinstance(expr, dict) and 'exp' in expr):
    #     print(expr.keys(), expr['op'])
    #     return _evaluate_expression(expr['exp'], state)
    
    if isinstance(expr, (int, float, bool)):
        return expr
    
    if isinstance(expr, str):
        # Could be a state variable or a named constant.
        # First prefer the state variable.
        if expr in state:
            return state[expr]

        # Then check provided constants. Support either:
        # - a dict mapping name -> value or name -> {name,type,value}
        # - a list/tuple of {name, type, value} dicts
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
    # jani_file = "mdp_models/ij.10.v1.jani"
    jani_file = "mdp_models/philosophers-mdp.3.v1.jani"
    # jani_file = "mdp_models/firewire_abst.v1.jani"
    # jani_file = "mdp_models/consensus.2.v1.jani"

    mdp = convert_jani_to_mdp(jani_file)
    
    print(f"Number of states: {len(mdp.states)}")
    print(f"Initial state: {mdp.initial_state}")
    print(f"Number of goal states: {len(mdp.goal_states)}")
    print(f"Number of state-action pairs: {len(mdp.state_action_pairs)}")

    # for g in mdp.goal_states:
    #     print("Goal state:", g)
    # print("HELLO")
    # for s in mdp.states:
    #     if ('steps', 1.0) in s or ('steps', 1) in s:
    #         print("State with steps=1:", s)
    
    # Print some example states and transitions
    if mdp.states:
        sample_state = list(mdp.states)[0]
        print(f"\nSample state: {sample_state}")
        if sample_state in mdp.topology:
            print(f"Actions available: {mdp.topology[sample_state]}")
            for action in mdp.topology[sample_state]:
                transitions = mdp.get_transition_probabilities(sample_state, action)
                print(f"  Action {action}: {transitions}")
