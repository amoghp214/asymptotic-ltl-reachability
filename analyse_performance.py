import json
import os
from pathlib import Path
from statistics import median
from collections import defaultdict

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
    

if __name__ == '__main__':
    print("Processing analysis data files...")
    raw_results = extract_benchmarks()
    final_results = refine_results(raw_results)
    plot_results(final_results)
    
    # if final_results:
    #     save_results(final_results)
    #     print(f"Processed and saved refined results for {len(final_results.keys())} benchmark(s).")
    # else:
    #     print("No analysis data files found.")

    print("Analysis complete.")