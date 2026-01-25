import numpy as np
import matplotlib.pyplot as plt
import json
import os


def plot_error_history(analysis_dir, learning_history, log_scale=False):
    # plots the learning error history for each iteration and saves it to error_history_plot_path
    error_vs_k_plot_path = os.path.join(analysis_dir, "error_vs_k.png")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure()
    if log_scale:
        plt.semilogy(list(np.array(learning_history)[:, -3]), marker='o')
        plt.ylim(top=2)
    else:
        plt.plot(list(np.array(learning_history)[:, -3]), marker='o')
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
        plt.semilogy(list(np.array(learning_history)[:, 1]), list(np.array(learning_history)[:, -3]))
        plt.ylim(top=2)
    else:
        plt.plot(list(np.array(learning_history)[:, 1]), list(np.array(learning_history)[:, -3]))
        plt.ylim(-0.1, 1.1)
    plt.xlabel("Number of Samples")
    plt.ylabel("Error (U - L)")
    plt.title("Learning Error vs Number of Samples")
    plt.grid()
    plt.savefig(error_vs_samples_plot_path)
    plt.close()

def plot_states_set_history(analysis_dir, states_set_history, max_states, log_scale=False):
    # plots the number of states seen history for each iteration and saves it in analysis_dir
    num_states_seen_vs_k_plot_path = os.path.join(analysis_dir, "num_states_seen_vs_k.png")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure()
    if log_scale:
        plt.semilogy(list(np.array(states_set_history)[:, -1]), marker='o')
        plt.ylim(top=max_states * 2)
    else:
        plt.plot(list(np.array(states_set_history)[:, -1]), marker='o')
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
        plt.semilogy(list(np.array(states_set_history)[:, 1]), list(np.array(states_set_history)[:, -1]))
        plt.ylim(top=max_states * 2)
    else:
        plt.plot(list(np.array(states_set_history)[:, 1]), list(np.array(states_set_history)[:, -1]))
        plt.ylim(-0.1 * max_states, max_states * 1.1)
    plt.xlabel("Number of Samples")
    plt.ylabel("Num Seen States")
    plt.title("Num Seen States vs Number of Samples")
    plt.grid()
    plt.savefig(num_states_seen_vs_samples_plot_path)
    plt.close()

def plot_transitions_seen_history(analysis_dir, transitions_seen_history, max_transitions, log_scale=False):
    # plots the number of transitions seen history for each iteration and saves it in analysis_dir
    num_transitions_seen_vs_k_plot_path = os.path.join(analysis_dir, "num_transitions_seen_vs_k.png")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure()
    if log_scale:
        plt.semilogy(list(np.array(transitions_seen_history)[:, -1]), marker='o')
        plt.ylim(top=max_transitions * 2)
    else:
        plt.plot(list(np.array(transitions_seen_history)[:, -1]), marker='o')
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
        plt.semilogy(list(np.array(transitions_seen_history)[:, 1]), list(np.array(transitions_seen_history)[:, -1]))
        plt.ylim(top=max_transitions * 2)
    else:
        plt.plot(list(np.array(transitions_seen_history)[:, 1]), list(np.array(transitions_seen_history)[:, -1]))
        plt.ylim(-0.1 * max_transitions, max_transitions * 1.1)
    plt.xlabel("Number of Samples")
    plt.ylabel("Num Seen Transitions")
    plt.title("Num Seen Transitions vs Number of Samples")
    plt.grid()
    plt.savefig(num_transitions_seen_vs_samples_plot_path)
    plt.close()

def plot_policy_accuracy_history(analysis_dir, policy_accuracy_history, log_scale=False):
    # plots the policy accuracy history for each iteration and saves it in analysis_dir
    policy_accuracy_vs_k_plot_path = os.path.join(analysis_dir, "policy_accuracy_vs_k.png")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure()
    if log_scale:
        plt.semilogx(list(np.array(policy_accuracy_history)[:, 0]), list(np.array(policy_accuracy_history)[:, -1]), marker='o')
        plt.xlabel("Iteration (log scale)")
        plt.ylim(top=2)
    else:
        plt.plot(list(np.array(policy_accuracy_history)[:, 0]), list(np.array(policy_accuracy_history)[:, -1]), marker='o')
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
        plt.semilogy(list(np.array(policy_accuracy_history)[:, 1]), list(np.array(policy_accuracy_history)[:, -1]))
        plt.xlabel("Number of Samples (log scale)")
        plt.ylim(top=2)  
    else:
        plt.plot(list(np.array(policy_accuracy_history)[:, 1]), list(np.array(policy_accuracy_history)[:, -1]))
        plt.xlabel("Number of Samples")
        plt.ylim(-0.1, 1.1)
    plt.ylabel("Policy Accuracy")
    plt.title("Policy Accuracy vs Number of Samples")
    plt.grid()
    plt.savefig(policy_accuracy_vs_samples_plot_path)
    plt.close()

def plot_bvi_history(analysis_dir, bvi_history, log_scale=False):
    # plots the bvi error history for each iteration and saves it in analysis_dir
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

def plot_value_bounds(analysis_dir, learning_history):
    """Plot lower and upper bounds vs iteration k and vs number of samples."""
    bounds_vs_k_plot_path = os.path.join(analysis_dir, "value_bounds_vs_k.png")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    k_values = list(np.array(learning_history)[:, 0])
    lower_bounds = list(np.array(learning_history)[:, -2])
    upper_bounds = list(np.array(learning_history)[:, -1])
    
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
    
    sample_counts = list(np.array(learning_history)[:, 1])
    lower_bounds = list(np.array(learning_history)[:, -2])
    upper_bounds = list(np.array(learning_history)[:, -1])
    
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

def run_analysis(
    analysis_dir: str,
    learning_history: list,
    states_set_history: list,
    transitions_seen_history: list,
    policy_accuracy_history: list,
    true_error_history: list,
    max_states: int,
    max_transitions: int,
    true_confidence_error: float,
    true_p_min: float,
):
    """
    Run analysis and generate plots for the learning process.

    Args:
        analysis_dir (str): Directory to save analysis plots and data.
    """
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    plot_error_history(analysis_dir=analysis_dir, learning_history=learning_history, log_scale=True)
    plot_states_set_history(analysis_dir=analysis_dir, states_set_history=states_set_history, max_states=max_states, log_scale=False)
    plot_transitions_seen_history(analysis_dir=analysis_dir, transitions_seen_history=transitions_seen_history, max_transitions=max_transitions, log_scale=False)
    plot_policy_accuracy_history(analysis_dir=analysis_dir, policy_accuracy_history=policy_accuracy_history, log_scale=False)
    plot_value_bounds(analysis_dir=analysis_dir, learning_history=learning_history)
    
    # Save all histories to json file
    analysis_data = {
        "true_confidence_error": true_confidence_error,
        "true_p_min": true_p_min,
        "max_states": max_states,
        "max_transitions": max_transitions,
        "learning_history": learning_history,
        "states_set_history": states_set_history,
        "transitions_seen_history": transitions_seen_history,
        "policy_accuracy_history": policy_accuracy_history,
        "true_error_history": true_error_history,
    }
    analysis_data_path = os.path.join(analysis_dir, "analysis_data.json")
    with open(analysis_data_path, "w") as f:
        json.dump(analysis_data, f, indent=4)