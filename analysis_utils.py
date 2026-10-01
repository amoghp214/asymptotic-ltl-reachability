import numpy as np
import matplotlib.pyplot as plt
import json
import math
import os

#: Shown on the policy convergence plots when the states behind the policy are
#: MEC-collapsed super-states rather than raw MDP states. Their identity can
#: change from stage to stage, so "the same state kept its action" is not a
#: well-defined question for them.
COLLAPSED_STATES_NOTE = ("Note: states are MEC-collapsed super-states, whose identity can change "
                         "between stages.\nChurn and retention are indicative only for this learner.")

with open("optimal_policy_reachability.json", "r") as f:
    OPTIMAL_POLICY_REACHABILITY = json.load(f)

class PolicyRecorder:
    """
    Record the U-greedy policy at the end of every stage, in a compact form.

    The policy is what the PAC guarantee is actually about: the bound is on
    V^{pi_U} - V*, not on U - L. Storing it per stage lets the analysis ask how
    much the extracted policy still moves from round to round.

    State and action objects are interned, because states are long tuples and
    writing them out once per stage would dominate the file.
    """

    def __init__(self, states_collapsed=False):
        # states_collapsed marks a learner whose policy is keyed on MEC-collapsed
        # super-states, so the analysis can warn that stages are not comparable.
        self.states_collapsed = states_collapsed
        self.state_index = []   # interned state -> position is its id
        self.action_index = []  # interned action -> position is its id
        self.policy_history = []  # [(k, {state_id: [action_id, ...]}), ...]
        self._state_ids = dict()
        self._action_ids = dict()

    def _intern(self, value, index_list, id_map):
        key = str(value)
        if key not in id_map:
            id_map[key] = len(index_list)
            index_list.append(key)
        return id_map[key]

    def record(self, k, best_actions):
        """
        Store the policy for stage k.

        Args:
            k (int): Stage number.
            best_actions (dict): {state: [actions]} as the model-free learners
                hold it, or {state: action} as the model-based learner does;
                a bare action is stored as a one-element list.
        """
        snapshot = dict()
        for state, actions in (best_actions or {}).items():
            if not isinstance(actions, (list, tuple, set)):
                actions = [actions]
            state_id = self._intern(state, self.state_index, self._state_ids)
            snapshot[str(state_id)] = sorted(
                self._intern(action, self.action_index, self._action_ids) for action in actions
            )
        self.policy_history.append([k, snapshot])

    def to_dict(self):
        """The three keys that go into analysis_data.json."""
        return {
            "state_index": self.state_index,
            "action_index": self.action_index,
            "policy_history": self.policy_history,
            "policy_states_collapsed": self.states_collapsed,
        }


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

def plot_policy_accuracy_history(analysis_dir, policy_accuracy_history, benchmark_name="", log_scale=False):
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

    if benchmark_name in OPTIMAL_POLICY_REACHABILITY:
        print(f"Benchmark {benchmark_name} has optimal reachability: {OPTIMAL_POLICY_REACHABILITY[benchmark_name]}")
        optimal_reachability = OPTIMAL_POLICY_REACHABILITY[benchmark_name]
        plt.axhline(y=optimal_reachability, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=10, label=f'Optimal Reachability = {optimal_reachability:.2f}')
        plt.legend()

    plt.ylabel("Policy Accuracy (Reachability)")
    plt.title("Policy Accuracy (Reachability) History")
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
    
    if benchmark_name in OPTIMAL_POLICY_REACHABILITY:
        optimal_reachability = OPTIMAL_POLICY_REACHABILITY[benchmark_name]
        plt.axhline(y=optimal_reachability, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=10, label=f'Optimal Reachability = {optimal_reachability:.2f}')
        plt.legend()

    plt.ylabel("Policy Accuracy (Reachability)")
    plt.title("Policy Accuracy (Reachability) vs Number of Samples")
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

def plot_value_bounds(analysis_dir, learning_history, benchmark_name=""):
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
    if benchmark_name in OPTIMAL_POLICY_REACHABILITY:
        optimal_reachability = OPTIMAL_POLICY_REACHABILITY[benchmark_name]
        plt.axhline(y=optimal_reachability, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=10, label=f'Optimal Reachability = {optimal_reachability:.2f}')

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
    if benchmark_name in OPTIMAL_POLICY_REACHABILITY:
        optimal_reachability = OPTIMAL_POLICY_REACHABILITY[benchmark_name]
        plt.axhline(y=optimal_reachability, color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=10, label=f'Optimal Reachability = {optimal_reachability:.2f}')

    plt.legend()
    plt.grid()
    plt.savefig(bounds_vs_samples_plot_path)
    plt.close()

def policy_snapshots(trial_data):
    """
    The stored per-stage policy as {k: {state_id: frozenset(action_ids)}}.

    Returns an empty dict for results produced before policies were recorded.
    """
    snapshots = dict()
    for k, snapshot in trial_data.get('policy_history', []):
        snapshots[int(k)] = {state_id: frozenset(action_ids)
                             for state_id, action_ids in snapshot.items()}
    return snapshots


def p_min_stage(trial_data):
    """
    First stage whose p_k has reached the model's true p_min.

    The PAC guarantee only bites once the stage's p_k is at or below the real
    minimum transition probability, so that is where the retention analysis
    starts. Returns None when no stage in the run got that far.
    """
    true_p_min = trial_data.get('true_p_min', 0)
    for row in trial_data.get('learning_history', []):
        if row[3] <= true_p_min:
            return int(row[0])
    return None


def projected_p_min_stage(trial_data):
    """
    The stage at which p_k reaches the true p_min, projected when not there yet.

    p_k halves every stage, so the crossing point follows from any single
    recorded stage even before a run gets near it. That lets a plot drawn
    mid-run still show where the stage guarantees start to hold.

    Returns:
        tuple: (k, projected). k is None when it cannot be worked out;
        projected is True when the run has not actually reached that stage.
    """
    true_p_min = trial_data.get('true_p_min', 0)
    learning_history = trial_data.get('learning_history', [])
    if not learning_history or true_p_min <= 0:
        return None, False

    reached = p_min_stage(trial_data)
    if reached is not None:
        return reached, False

    # Not there yet: extrapolate the halving schedule from the last stage run.
    last_k, last_p_k = int(learning_history[-1][0]), learning_history[-1][3]
    if last_p_k <= 0:
        return None, False
    return last_k + max(1, math.ceil(math.log2(last_p_k / true_p_min))), True


def compute_policy_metrics(trial_data):
    """
    Policy churn and early-action retention for one trial.

    Churn at stage k is the fraction of states seen in both k-1 and k whose
    U-greedy action set changed. It is the convergence signal: the guarantee is
    on V^{pi_U} - V*, so what matters is that the extracted policy stops moving.

    Retention at stage k is the fraction of the (state, action) pairs that were
    U-greedy at the p_min stage and are still U-greedy at k, which answers
    whether an action chosen early is kept or dropped as stages deepen.
    Non-monotonicity counts pairs that were dropped at some earlier stage and
    have come back, so a flat retention curve hiding churn underneath is visible.

    Returns:
        tuple: (churn_rows, retention_rows, nonmonotone_rows), each a list of
        [k, num_samples, value]. All three are empty when the trial predates
        policy recording.
    """
    snapshots = policy_snapshots(trial_data)
    if not snapshots:
        return [], [], []

    samples_at_k = {int(row[0]): row[1] for row in trial_data.get('learning_history', [])}
    stages = sorted(snapshots)

    churn_rows = []
    for previous_k, k in zip(stages, stages[1:]):
        previous, current = snapshots[previous_k], snapshots[k]
        shared = set(previous) & set(current)
        if not shared:
            continue
        changed = sum(1 for state_id in shared if previous[state_id] != current[state_id])
        churn_rows.append([k, samples_at_k.get(k, 0), changed / len(shared)])

    # The retention baseline is the p_min stage; fall back to the first stage so
    # a run that never reaches p_min still produces a curve.
    reference_k = p_min_stage(trial_data)
    if reference_k is None or reference_k not in snapshots:
        reference_k = stages[0]

    baseline = {(state_id, action_id)
                for state_id, action_ids in snapshots[reference_k].items()
                for action_id in action_ids}

    retention_rows, nonmonotone_rows = [], []
    dropped_so_far = set()
    for k in stages:
        if k < reference_k:
            continue
        current = {(state_id, action_id)
                   for state_id, action_ids in snapshots[k].items()
                   for action_id in action_ids}
        if not baseline:
            continue
        kept = baseline & current
        # A pair counts as non-monotone once it has been absent and returned.
        returned = kept & dropped_so_far
        dropped_so_far |= (baseline - current)

        retention_rows.append([k, samples_at_k.get(k, 0), len(kept) / len(baseline)])
        nonmonotone_rows.append([k, samples_at_k.get(k, 0), len(returned) / len(baseline)])

    return churn_rows, retention_rows, nonmonotone_rows


def annotate_collapsed_states(note=COLLAPSED_STATES_NOTE):
    """Caption the current figure with the MEC-collapsed-states caveat."""
    plt.subplots_adjust(bottom=0.26)
    plt.figtext(0.5, 0.015, note, ha='center', va='bottom', fontsize=7.5,
                style='italic', color='dimgray', wrap=True)


def plot_policy_churn_history(analysis_dir, policy_churn_history, states_collapsed=False, log_scale=False):
    # plots how much the U-extracted policy changed between consecutive stages
    policy_churn_vs_k_plot_path = os.path.join(analysis_dir, "policy_churn_vs_k.png")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure()
    if log_scale:
        plt.semilogy(list(np.array(policy_churn_history)[:, 0]), list(np.array(policy_churn_history)[:, -1]), marker='o')
        plt.ylim(top=2)
    else:
        plt.plot(list(np.array(policy_churn_history)[:, 0]), list(np.array(policy_churn_history)[:, -1]), marker='o')
        plt.ylim(-0.1, 1.1)
    plt.xlabel("Iteration")
    plt.ylabel("Fraction of States with Changed Action")
    plt.title("Policy Churn Between Consecutive Stages")
    plt.grid()
    if states_collapsed:
        annotate_collapsed_states()
    plt.savefig(policy_churn_vs_k_plot_path)
    plt.close()

    policy_churn_vs_samples_plot_path = os.path.join(analysis_dir, "policy_churn_vs_samples.png")
    plt.figure()
    if log_scale:
        plt.semilogy(list(np.array(policy_churn_history)[:, 1]), list(np.array(policy_churn_history)[:, -1]), marker='o')
        plt.ylim(top=2)
    else:
        plt.plot(list(np.array(policy_churn_history)[:, 1]), list(np.array(policy_churn_history)[:, -1]), marker='o')
        plt.ylim(-0.1, 1.1)
    plt.xlabel("Number of Samples")
    plt.ylabel("Fraction of States with Changed Action")
    plt.title("Policy Churn vs Number of Samples")
    plt.grid()
    if states_collapsed:
        annotate_collapsed_states()
    plt.savefig(policy_churn_vs_samples_plot_path)
    plt.close()

def plot_policy_retention_history(analysis_dir, policy_retention_history, p_min_stage_k=None,
                                  p_min_projected=False, states_collapsed=False, log_scale=False):
    # plots how many of the p_min-stage actions are still chosen at each later stage.
    # p_min_stage_k marks where p_k reaches the true p_min, which is where the
    # stage guarantees start to hold; it is drawn even when the run has not got
    # there yet, so the curve can be read against how far off that still is.
    p_min_label = None
    if p_min_stage_k is not None:
        p_min_label = f"p_k <= p_min at k={p_min_stage_k}"
        if p_min_projected:
            p_min_label += " (projected)"

    policy_retention_vs_k_plot_path = os.path.join(analysis_dir, "policy_retention_vs_k.png")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.figure()
    if log_scale:
        plt.semilogy(list(np.array(policy_retention_history)[:, 0]), list(np.array(policy_retention_history)[:, -1]), marker='o')
        plt.ylim(top=2)
    else:
        plt.plot(list(np.array(policy_retention_history)[:, 0]), list(np.array(policy_retention_history)[:, -1]), marker='o')
        plt.ylim(-0.1, 1.1)
    if p_min_label:
        plt.axvline(x=p_min_stage_k, color='green', linestyle=':', linewidth=2, label=p_min_label)
        plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Fraction of p_min-Stage Actions Kept")
    plt.title("Retention of p_min-Stage Actions")
    plt.grid()
    if states_collapsed:
        annotate_collapsed_states()
    plt.savefig(policy_retention_vs_k_plot_path)
    plt.close()

    # On the samples axis the marker needs the sample count at that stage, which
    # is only known once the run has actually reached it.
    p_min_samples = next((row[1] for row in policy_retention_history
                          if int(row[0]) == p_min_stage_k), None) if p_min_stage_k is not None else None

    policy_retention_vs_samples_plot_path = os.path.join(analysis_dir, "policy_retention_vs_samples.png")
    plt.figure()
    if log_scale:
        plt.semilogy(list(np.array(policy_retention_history)[:, 1]), list(np.array(policy_retention_history)[:, -1]), marker='o')
        plt.ylim(top=2)
    else:
        plt.plot(list(np.array(policy_retention_history)[:, 1]), list(np.array(policy_retention_history)[:, -1]), marker='o')
        plt.ylim(-0.1, 1.1)
    if p_min_samples is not None:
        plt.axvline(x=p_min_samples, color='green', linestyle=':', linewidth=2, label=p_min_label)
        plt.legend()
    plt.xlabel("Number of Samples")
    plt.ylabel("Fraction of p_min-Stage Actions Kept")
    plt.title("Retention of p_min-Stage Actions vs Number of Samples")
    plt.grid()
    if states_collapsed:
        annotate_collapsed_states()
    plt.savefig(policy_retention_vs_samples_plot_path)
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
    benchmark_name: str = "Benchmark",
    policy_recorder=None,
    policy_churn_history: list = None,
    policy_retention_history: list = None,
    p_min_stage_k: int = None,
    p_min_projected: bool = False,
    policy_states_collapsed: bool = False,
):
    """
    Run analysis and generate plots for the learning process.

    Args:
        benchmark_name (str): Name of the benchmark, used for labeling plots and reporting in analysis data.
        analysis_dir (str): Directory to save analysis plots and data.
        learning_history (list): List of tuples (k, num_samples, error, L(s0), U(s0)) for each iteration.
        states_set_history (list): List of tuples (k, num_samples, num_seen_states) for each iteration.
        transitions_seen_history (list): List of tuples (k, num_samples, num_seen_transitions) for each iteration.
        policy_accuracy_history (list): List of tuples (k, num_samples, policy_accuracy) for each iteration.
        true_error_history (list): List of tuples (k, num_samples, true_error) for each iteration.
        max_states (int): Maximum number of states in the MDP, used for setting y-axis limits in plots.
        max_transitions (int): Maximum number of transitions in the MDP, used for setting y-axis limits in plots.
        true_confidence_error (float): The true confidence error of the final policy, used for reporting in analysis data.
        true_p_min (float): The true p_min of the MDP, used for reporting in analysis data.
        policy_recorder (PolicyRecorder or None): If given, its per-stage U-greedy
            policies are stored alongside the histories, and the churn and
            retention curves are derived from them for the stages done so far.
        policy_churn_history (list): Pre-computed [(k, num_samples, churn), ...],
            as analyse_performance.py passes when plotting medians over trials.
            Derived from policy_recorder when not given.
        policy_retention_history (list): Pre-computed [(k, num_samples, retention), ...].
        p_min_stage_k (int): Stage at which p_k reaches the true p_min, marked on
            the retention plot. Derived from policy_recorder when not given.
        p_min_projected (bool): True when that stage has not been reached yet and
            the marker is an extrapolation of the halving schedule.
        policy_states_collapsed (bool): True when the policy is keyed on
            MEC-collapsed super-states, which captions the plots with the caveat.
            Taken from policy_recorder when one is given.
    """
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    plot_error_history(analysis_dir=analysis_dir, learning_history=learning_history, log_scale=False)
    plot_states_set_history(analysis_dir=analysis_dir, states_set_history=states_set_history, max_states=max_states, log_scale=False)
    plot_transitions_seen_history(analysis_dir=analysis_dir, transitions_seen_history=transitions_seen_history, max_transitions=max_transitions, log_scale=False)
    plot_policy_accuracy_history(analysis_dir=analysis_dir, policy_accuracy_history=policy_accuracy_history, benchmark_name=benchmark_name, log_scale=False)
    plot_value_bounds(analysis_dir=analysis_dir, learning_history=learning_history, benchmark_name=benchmark_name)

    # A learner passes its recorder and the curves are derived here over the
    # stages finished so far; analyse_performance.py passes medians it has
    # already computed across trials. No band either way -- the banded versions
    # live in analyse_performance.py, which has the trials to take a stdev over.
    if policy_recorder is not None and policy_churn_history is None and policy_retention_history is None:
        trial_data = {
            'policy_history': policy_recorder.policy_history,
            'learning_history': learning_history,
            'true_p_min': true_p_min,
        }
        policy_churn_history, policy_retention_history, _ = compute_policy_metrics(trial_data)
        p_min_stage_k, p_min_projected = projected_p_min_stage(trial_data)
        policy_states_collapsed = policy_recorder.states_collapsed

    if policy_churn_history:
        plot_policy_churn_history(analysis_dir=analysis_dir, policy_churn_history=policy_churn_history, states_collapsed=policy_states_collapsed, log_scale=False)
    if policy_retention_history:
        plot_policy_retention_history(analysis_dir=analysis_dir, policy_retention_history=policy_retention_history, p_min_stage_k=p_min_stage_k, p_min_projected=p_min_projected, states_collapsed=policy_states_collapsed, log_scale=False)
    
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
    if policy_recorder is not None:
        analysis_data.update(policy_recorder.to_dict())
    analysis_data_path = os.path.join(analysis_dir, "analysis_data.json")
    with open(analysis_data_path, "w") as f:
        json.dump(analysis_data, f, indent=4)
