import json
import os
from pathlib import Path
from statistics import median
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np

from analysis_utils import run_analysis

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
    
    for trial_data in benchmark_data.values():
        learning_histories.append(trial_data['learning_history'])
        states_set_histories.append(trial_data['states_set_history'])
        transitions_seen_histories.append(trial_data['transitions_seen_history'])
        policy_accuracy_histories.append(trial_data['policy_accuracy_history'])
        true_error_histories.append(trial_data['true_error_history'])
    
    benchmark_metrics = {}

    benchmark_metrics['true_confidence_error'] = benchmark_data['1']['true_confidence_error']
    benchmark_metrics['true_p_min'] = benchmark_data['1']['true_p_min']
    benchmark_metrics['max_states'] = benchmark_data['1']['max_states']
    benchmark_metrics['max_transitions'] = benchmark_data['1']['max_transitions']
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
        final_results[benchmark_name] = get_benchmark_medians(benchmark_data)

    return final_results

def extract_benchmarks(base_dir='./results'):
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
        benchmark_name = '_'.join(parts[:-2])
        trial_number = parts[-1]

        if benchmark_name not in results.keys():
            results[benchmark_name] = dict()
        if trial_number not in results[benchmark_name].keys():
            results[benchmark_name][trial_number] = dict()
            
        analysis_file = benchmark_dir / 'analysis_data.json'
        assert analysis_file.exists(), f"Analysis file not found: {analysis_file}"
        
        try:
            data = load_analysis_data(analysis_file)
            results[benchmark_name][trial_number] = data
        except Exception as e:
            print(f"Error processing {analysis_file}: {e}")
    
    return results

def plot_results(results, output_dir='./analysis/'):
    for benchmark_name, benchmark_results in results.items():
        os.makedirs(os.path.join(output_dir, benchmark_name), exist_ok=True)
        output_path = Path(output_dir) / benchmark_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        run_analysis(
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
    
def plot_error_stdev_curve(final_results, raw_results, output_dir='./analysis/'):
    for benchmark_name in final_results.keys():
        os.makedirs(os.path.join(output_dir, benchmark_name), exist_ok=True)
        output_path = Path(output_dir) / benchmark_name
        output_path.mkdir(parents=True, exist_ok=True)
        analysis_dir = str(output_path)
        learning_history = final_results[benchmark_name]['learning_history']
        # calculate stdev of error at each iteration across trials for a benchmark
        error_stdev_history = []
        num_trials = len(raw_results[benchmark_name])
        max_iterations_done = len(learning_history)
        for i in range(max_iterations_done):
            errors_at_i = []
            for trial_data in raw_results[benchmark_name].values():
                if len(trial_data['learning_history']) > i:
                    errors_at_i.append(trial_data['learning_history'][i][-3])
            if errors_at_i:
                stdev_i = np.std(errors_at_i)
                error_stdev_history.append((i + 1, stdev_i))
            else:
                error_stdev_history.append((i + 1, 0.0))
        plot_error_history_w_stdev(analysis_dir, learning_history, error_stdev_history, log_scale=False)


def plot_error_history_w_stdev(analysis_dir, learning_history, error_stdev_history, log_scale=False):
    # plots the learning error history for each iteration and saves it to error_history_plot_path
    # Prepare arrays for plotting and create wrappers that inject the stdev shaded area
    arr = np.array(learning_history)
    if arr.size == 0:
        return

    y_arr = arr[:, -3].astype(float)
    x_idx = np.arange(len(y_arr))

    # Align stdevs to the learning_history indices. error_stdev_history is [(k, stdev), ...]
    if error_stdev_history:
        ks = [int(t[0]) for t in error_stdev_history]
        min_k = min(ks)
        stdev_map = {int(k): float(s) for k, s in error_stdev_history}
        stdevs = np.array([stdev_map.get(i + min_k, 0.0) for i in range(len(y_arr))], dtype=float)
    else:
        stdevs = np.zeros_like(y_arr)

    # function to draw translucent, per-segment colored fill between (y - stdev) and (y + stdev)
    def _draw_shaded(ax, x, y, s, log_scale):
        upper = y + s
        lower = y - s
        if log_scale:
            # avoid non-positive values for log scale
            lower = np.maximum(lower, 1e-12)

        cmap = plt.get_cmap('viridis')
        max_s = s.max() if len(s) > 0 else 0.0
        if max_s == 0:
            colors = [cmap(0.5)] * max(1, len(s))
        else:
            colors = [cmap(val / max_s) for val in s]

        # fill per-segment so color can vary with stdev
        for i in range(len(x) - 1):
            xi = [x[i], x[i + 1]]
            yi_lower = [lower[i], lower[i + 1]]
            yi_upper = [upper[i], upper[i + 1]]
            ax.fill_between(xi, yi_lower, yi_upper, color=colors[i], alpha=0.3, linewidth=0)
        if len(x) == 1:
            ax.fill_between([x[0] - 0.5, x[0] + 0.5], [lower[0], lower[0]], [upper[0], upper[0]],
                            color=colors[0], alpha=0.3, linewidth=0)

    # wrap plotting functions so we can inject the shaded area when the error-vs-iteration plot is drawn
    orig_plot = plt.plot
    orig_semilogy = plt.semilogy

    y_list_ref = list(y_arr)  # reference to identify the error-history plot when plotting

    def _is_same_array(a, b):
        try:
            return np.array(a, dtype=float).shape == np.array(b, dtype=float).shape and \
                   np.allclose(np.array(a, dtype=float), np.array(b, dtype=float))
        except Exception:
            return False

    def wrapped_plot(*args, **kwargs):
        res = orig_plot(*args, **kwargs)
        # detect call: plot(y) where y matches our error list -> iteration-error plot
        if len(args) >= 1 and _is_same_array(args[0], y_list_ref):
            ax = plt.gca()
            _draw_shaded(ax, x_idx, y_arr, stdevs, log_scale=False)
        return res

    def wrapped_semilogy(*args, **kwargs):
        res = orig_semilogy(*args, **kwargs)
        # detect call: semilogy(y) where y matches our error list -> iteration-error plot (log scale)
        if len(args) >= 1 and _is_same_array(args[0], y_list_ref):
            ax = plt.gca()
            _draw_shaded(ax, x_idx, y_arr, stdevs, log_scale=True)
        return res

    plt.plot = wrapped_plot
    plt.semilogy = wrapped_semilogy
    error_vs_k_plot_path = os.path.join(analysis_dir, "error_w_std_vs_k.png")
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

    error_vs_samples_plot_path = os.path.join(analysis_dir, "error_w_std_vs_samples.png")
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


if __name__ == '__main__':
    print("Processing analysis data files...")
    raw_results = extract_benchmarks()
    final_results = refine_results(raw_results)
    plot_results(final_results)

    plot_error_stdev_curve(final_results, raw_results)
    # plot_policy_accuracy_stdev_curve(final_results, raw_results)

    
    # if final_results:
    #     save_results(final_results)
    #     print(f"Processed and saved refined results for {len(final_results.keys())} benchmark(s).")
    # else:
    #     print("No analysis data files found.")

    print("Analysis complete.")