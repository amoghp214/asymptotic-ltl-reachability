import json
import os
import argparse
from pathlib import Path
from statistics import median
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np

from analysis_utils import run_analysis, OPTIMAL_POLICY_REACHABILITY

def load_analysis_data(file_path):
    """Load analysis data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_median_list(data_list):
    """Calculate median for each position in a list of lists."""
    if not data_list:
        return []
    
    max_iterations_done = max([len(data) for data in data_list])
    median_list = []
    
    for i in range(max_iterations_done):
        median_i_row = []
        for j in range(len(data_list[0][0])):
            ijth_elements = [data[i][j] for data in data_list if len(data) > i]
            median_i_row.append(median(ijth_elements))
        median_list.append(median_i_row)
    
    return median_list

def get_benchmark_medians(benchmark_data):
    """Calculate median metrics for a benchmark across its trials."""

    learning_histories = []
    states_set_histories = []
    transitions_seen_histories = []
    policy_accuracy_histories = []
    true_error_histories = []
    
    if not benchmark_data:
        return None

    for trial_data in benchmark_data.values():
        learning_histories.append(trial_data['learning_history'])
        states_set_histories.append(trial_data['states_set_history'])
        transitions_seen_histories.append(trial_data['transitions_seen_history'])
        policy_accuracy_histories.append(trial_data['policy_accuracy_history'])
        true_error_histories.append(trial_data['true_error_history'])
    
    benchmark_metrics = {}

    # These are constant across trials, so read them from any surviving trial
    # rather than assuming trial '1' was loaded.
    reference_trial = next(iter(benchmark_data.values()))
    benchmark_metrics['true_confidence_error'] = reference_trial['true_confidence_error']
    benchmark_metrics['true_p_min'] = reference_trial['true_p_min']
    benchmark_metrics['max_states'] = reference_trial['max_states']
    benchmark_metrics['max_transitions'] = reference_trial['max_transitions']
    benchmark_metrics['learning_history'] = get_median_list(learning_histories)
    benchmark_metrics['states_set_history'] = get_median_list(states_set_histories)
    benchmark_metrics['transitions_seen_history'] = get_median_list(transitions_seen_histories)
    benchmark_metrics['policy_accuracy_history'] = get_median_list(policy_accuracy_histories)
    benchmark_metrics['true_error_history'] = get_median_list(true_error_histories)

    return benchmark_metrics

def refine_results(data):
    """Calculate median metrics from analysis data."""
    
    
    final_results = {}

    for benchmark_name, benchmark_data in data.items():
        medians = get_benchmark_medians(benchmark_data)
        if medians is None:
            print(f"Skipping {benchmark_name}: no readable trials.")
            continue
        final_results[benchmark_name] = medians

    return final_results

def extract_benchmarks(base_dir='./results', variant=""):
    """Process all benchmark analysis files."""
    results = dict() # {benchmark_name: {trial_number: {metrics_dict}}}
    
    benchmarks_path = Path(base_dir)
    if not benchmarks_path.exists():
        print(f"Base directory {base_dir} not found.")
        return results
    
    for benchmark_dir in benchmarks_path.iterdir():
        if not benchmark_dir.is_dir():
            continue
        
        # Get benchmark name and trial number from dir name
        name = benchmark_dir.name
        parts = name.split('_')
        assert len(parts) >= 3, f"Unexpected benchmark directory name format: {name}"
        benchmark_name = '_'.join(parts[:-3])
        trial_number = parts[-2]
        variant_type = parts[-1]
        if variant and variant_type != variant:
            continue

        if benchmark_name not in results.keys():
            results[benchmark_name] = dict()

        analysis_file = benchmark_dir / 'analysis_data.json'

        # Only record a trial once its data is successfully read. Missing or
        # unreadable files are skipped rather than fatal: on Lustre an offline
        # OST makes stat/read fail with ENOTCONN/EIO, and one bad trial should
        # not discard the rest of the run.
        try:
            data = load_analysis_data(analysis_file)
        except OSError as e:
            print(f"Skipping unreadable {analysis_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {analysis_file}: {e}")
            continue

        results[benchmark_name][trial_number] = data
    
    return results

def plot_results(results, output_dir='./analysis/'):
    for benchmark_name, benchmark_results in results.items():
        os.makedirs(os.path.join(output_dir, benchmark_name), exist_ok=True)
        output_path = Path(output_dir) / benchmark_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        run_analysis(
            benchmark_name=benchmark_name,
            analysis_dir=str(output_path),
            learning_history=benchmark_results['learning_history'],
            states_set_history=benchmark_results['states_set_history'],
            transitions_seen_history=benchmark_results['transitions_seen_history'],
            policy_accuracy_history=benchmark_results['policy_accuracy_history'],
            true_error_history=benchmark_results['true_error_history'],
            max_states=benchmark_results['max_states'],
            max_transitions=benchmark_results['max_transitions'],
            true_confidence_error=benchmark_results['true_confidence_error'],
            true_p_min=benchmark_results['true_p_min']
        )
        
        print(f"{benchmark_name} plots saved to {output_path}")

def save_results(results, base_output_dir='./analysis/'):
    """Save aggregated results to JSON file."""
    for benchmark_name, benchmark_results in results.items():
        os.makedirs(os.path.join(base_output_dir, benchmark_name), exist_ok=True)
        output_path = Path(base_output_dir) / benchmark_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / 'performance_analysis.json'
        
        with open(output_file, 'w') as f:
            json.dump(dict(benchmark_results), f, indent=2)
        
        print(f"{benchmark_name} results saved to {output_file}")
    
def compute_stdev_history(raw_results, benchmark_name, history_key, value_index, num_points):
    """
    Standard deviation of one metric across trials, at each iteration.

    Args:
        raw_results (dict): {benchmark: {trial: analysis_data}} from extract_benchmarks.
        benchmark_name (str): Benchmark to compute for.
        history_key (str): Which history to read out of each trial's data.
        value_index (int): Column of that history holding the metric.
        num_points (int): Number of iterations in the median history.

    Returns:
        list: (k, stdev) per iteration, k being 1-based to match the histories.
    """
    stdev_history = []
    for i in range(num_points):
        values_at_i = []
        for trial_data in raw_results.get(benchmark_name, {}).values():
            history = trial_data.get(history_key, [])
            if len(history) > i:
                values_at_i.append(history[i][value_index])
        stdev_history.append((i + 1, float(np.std(values_at_i)) if values_at_i else 0.0))
    return stdev_history


def save_stdev_data(analysis_dir, key, stdev_history):
    """
    Record one stdev history in analysis_dir/stdev_data.json.

    compare_performance.py reads this file to draw the same bands when it
    combines variants: the medians in analysis_data.json cannot reproduce them.
    """
    stdev_data_path = os.path.join(analysis_dir, 'stdev_data.json')
    stdev_data = {}
    if os.path.exists(stdev_data_path):
        try:
            with open(stdev_data_path, 'r') as f:
                stdev_data = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {stdev_data_path} ({e}), rewriting it.")
            stdev_data = {}

    stdev_data[key] = [list(point) for point in stdev_history]
    with open(stdev_data_path, 'w') as f:
        json.dump(stdev_data, f, indent=2)


def draw_stdev_band(ax, x, y, stdevs, log_scale=False):
    """Shade y +/- stdev, colouring each segment by how large the stdev is."""
    upper = y + stdevs
    lower = y - stdevs
    if log_scale:
        # avoid non-positive values for log scale
        lower = np.maximum(lower, 1e-12)

    cmap = plt.get_cmap('viridis')
    max_s = stdevs.max() if len(stdevs) > 0 else 0.0
    if max_s == 0:
        colors = [cmap(0.5)] * max(1, len(stdevs))
    else:
        colors = [cmap(val / max_s) for val in stdevs]

    # fill per-segment so colour can vary with stdev
    for i in range(len(x) - 1):
        ax.fill_between([x[i], x[i + 1]], [lower[i], lower[i + 1]], [upper[i], upper[i + 1]],
                        color=colors[i], alpha=0.3, linewidth=0)
    if len(x) == 1:
        ax.fill_between([x[0] - 0.5, x[0] + 0.5], [lower[0], lower[0]], [upper[0], upper[0]],
                        color=colors[0], alpha=0.3, linewidth=0)


def plot_history_w_stdev(analysis_dir, history, stdev_history, value_index,
                         vs_k_filename, vs_samples_filename, ylabel,
                         vs_k_title, vs_samples_title, ylim=(-0.1, 1.1),
                         optimal_reachability=None, log_scale=False):
    """
    Plot one metric against k and against sample count, with the stdev shaded.

    Args:
        analysis_dir (str): Directory to save the plots in.
        history (list): Median history rows; column 0 is k and column 1 is samples.
        stdev_history (list): (k, stdev) pairs from compute_stdev_history.
        value_index (int): Column of `history` holding the metric.
        vs_k_filename, vs_samples_filename (str): Output file names.
        ylabel, vs_k_title, vs_samples_title (str): Axis label and titles.
        ylim (tuple): (bottom, top) for the y axis.
        optimal_reachability (float or None): If given, draw the V* line.
        log_scale (bool): Use a log y axis.
    """
    arr = np.array(history, dtype=float)
    if arr.size == 0:
        return

    os.makedirs(analysis_dir, exist_ok=True)
    y_values = arr[:, value_index]

    # Align stdevs to the median history by k, which both share.
    stdev_map = {int(k): float(s) for k, s in stdev_history}
    stdevs = np.array([stdev_map.get(int(k), 0.0) for k in arr[:, 0]], dtype=float)

    for filename, x_values, xlabel, title in (
        (vs_k_filename, arr[:, 0], "Iteration", vs_k_title),
        (vs_samples_filename, arr[:, 1], "Number of Samples", vs_samples_title),
    ):
        plt.figure()
        if log_scale:
            plt.semilogy(x_values, y_values, marker='o')
            plt.ylim(top=ylim[1])
        else:
            plt.plot(x_values, y_values, marker='o')
            plt.ylim(*ylim)

        draw_stdev_band(plt.gca(), x_values, y_values, stdevs, log_scale=log_scale)

        if optimal_reachability is not None:
            plt.axhline(y=optimal_reachability, color='red', linestyle='--', linewidth=2,
                        alpha=0.8, zorder=10,
                        label=f'Optimal Reachability = {optimal_reachability:.2f}')
            plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.savefig(os.path.join(analysis_dir, filename))
        plt.close()


def plot_error_stdev_curve(final_results, raw_results, output_dir='./analysis/'):
    """Error vs k and vs samples, with the across-trial stdev shaded."""
    for benchmark_name in final_results.keys():
        output_path = Path(output_dir) / benchmark_name
        output_path.mkdir(parents=True, exist_ok=True)
        analysis_dir = str(output_path)

        learning_history = final_results[benchmark_name]['learning_history']
        # calculate stdev of error at each iteration across trials for a benchmark
        error_stdev_history = compute_stdev_history(
            raw_results, benchmark_name, 'learning_history', -3, len(learning_history))
        save_stdev_data(analysis_dir, 'error_stdev_history', error_stdev_history)

        plot_history_w_stdev(
            analysis_dir=analysis_dir,
            history=learning_history,
            stdev_history=error_stdev_history,
            value_index=-3,
            vs_k_filename="error_w_std_vs_k.png",
            vs_samples_filename="error_w_std_vs_samples.png",
            ylabel="Error (U - L)",
            vs_k_title="Learning Error History",
            vs_samples_title="Learning Error vs Number of Samples",
        )


def plot_policy_accuracy_stdev_curve(final_results, raw_results, output_dir='./analysis/'):
    """Policy accuracy vs k and vs samples, with the across-trial stdev shaded."""
    for benchmark_name in final_results.keys():
        output_path = Path(output_dir) / benchmark_name
        output_path.mkdir(parents=True, exist_ok=True)
        analysis_dir = str(output_path)

        policy_accuracy_history = final_results[benchmark_name]['policy_accuracy_history']
        # calculate stdev of policy accuracy at each iteration across trials
        policy_accuracy_stdev_history = compute_stdev_history(
            raw_results, benchmark_name, 'policy_accuracy_history', -1, len(policy_accuracy_history))
        save_stdev_data(analysis_dir, 'policy_accuracy_stdev_history', policy_accuracy_stdev_history)

        plot_history_w_stdev(
            analysis_dir=analysis_dir,
            history=policy_accuracy_history,
            stdev_history=policy_accuracy_stdev_history,
            value_index=-1,
            vs_k_filename="policy_accuracy_w_std_vs_k.png",
            vs_samples_filename="policy_accuracy_w_std_vs_samples.png",
            ylabel="Policy Accuracy (Reachability)",
            vs_k_title="Policy Accuracy (Reachability) History",
            vs_samples_title="Policy Accuracy (Reachability) vs Number of Samples",
            optimal_reachability=OPTIMAL_POLICY_REACHABILITY.get(benchmark_name),
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate and plot benchmark analysis results.")
    parser.add_argument("-V", "--variant", type=str, default="",
                        help="Learner variant whose results to analyse (default: '').")
    args = parser.parse_args()

    output_dir = os.path.join('./analysis', args.variant)

    print("Processing analysis data files...")
    raw_results = extract_benchmarks(variant=args.variant)
    final_results = refine_results(raw_results)
    plot_results(final_results, output_dir=output_dir)

    plot_error_stdev_curve(final_results, raw_results, output_dir=output_dir)
    plot_policy_accuracy_stdev_curve(final_results, raw_results, output_dir=output_dir)

    
    # if final_results:
    #     save_results(final_results, base_output_dir=output_dir)
    #     print(f"Processed and saved refined results for {len(final_results.keys())} benchmark(s).")
    # else:
    #     print("No analysis data files found.")

    print("Analysis complete.")