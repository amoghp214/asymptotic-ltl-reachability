#!/usr/bin/env python3
"""
Compare benchmark results across learner variants.

analyse_performance.py writes one analysis_data.json per benchmark under
analysis/<variant>/<benchmark>/. This script reads several of those variant
folders at once and redraws the same plots with one curve per variant, so the
algorithms can be compared directly on a single set of axes.

Output goes to analysis/<variants joined by '_' in alphabetical order>/.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from analysis_utils import OPTIMAL_POLICY_REACHABILITY, annotate_collapsed_states
from variants import variant_names, variant_help


def load_variant(analysis_dir, variant):
    """
    Load every benchmark's analysis data for one variant.

    Args:
        analysis_dir (str): Root analysis directory, normally './analysis'.
        variant (str): Variant name, which is also its folder name.

    Returns:
        dict: {benchmark_name: analysis_data_dict} for benchmarks that have a
        readable analysis_data.json. Benchmarks whose file is missing or
        unparseable are skipped with a warning.
    """
    variant_path = Path(analysis_dir) / variant
    benchmarks = dict()

    if not variant_path.is_dir():
        print(f"Warning: no folder for variant '{variant}' at {variant_path}, skipping it.")
        return benchmarks

    for benchmark_dir in sorted(variant_path.iterdir()):
        if not benchmark_dir.is_dir():
            continue

        analysis_file = benchmark_dir / 'analysis_data.json'
        if not analysis_file.exists():
            print(f"Warning: {analysis_file} not found, skipping {variant}/{benchmark_dir.name}.")
            continue

        try:
            with open(analysis_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error processing {analysis_file}: {e}")
            continue

        # analyse_performance.py writes the across-trial stdevs next door. It is
        # optional: without it the _w_std plots simply carry no band.
        data['_stdev_data'] = dict()
        stdev_file = benchmark_dir / 'stdev_data.json'
        if stdev_file.exists():
            try:
                with open(stdev_file, 'r') as f:
                    data['_stdev_data'] = json.load(f)
            except Exception as e:
                print(f"Warning: could not read {stdev_file} ({e}), plotting without bands.")

        # Policy churn and retention are derived quantities: analyse_performance.py
        # computes them per trial and writes the medians here, because policies
        # cannot be averaged back out of analysis_data.json.
        policy_file = benchmark_dir / 'policy_metrics.json'
        if policy_file.exists():
            try:
                with open(policy_file, 'r') as f:
                    data.update(json.load(f))
            except Exception as e:
                print(f"Warning: could not read {policy_file} ({e}), skipping its policy plots.")

        benchmarks[benchmark_dir.name] = data

    return benchmarks


def variant_colors(variants):
    """Assign each variant a fixed colour so it looks the same in every plot."""
    cmap = plt.get_cmap('tab10')
    return {variant: cmap(i % 10) for i, variant in enumerate(variants)}


def plot_comparison(plot_path, series, history_key, y_index, xlabel, ylabel,
                    title, ylim, colors, x_index=0, optimal_reachability=None,
                    marker='o', vline_at=None, vline_label=None, note=None):
    """
    Draw one metric for several variants on a single set of axes.

    Args:
        plot_path (str): Where to save the .png.
        series (list): (variant_name, analysis_data_dict) in plotting order.
        history_key (str): Which history in the analysis data to read.
        y_index (int): Column of that history holding the value to plot.
        xlabel, ylabel, title (str): Axis labels and title.
        ylim (tuple): (bottom, top) for the y axis.
        colors (dict): variant name -> colour.
        x_index (int): Column holding the x value; 0 is k, 1 is sample count.
        optimal_reachability (float or None): If given, draw the V* line.
        marker (str): Marker style for the per-variant curves.
    """
    plt.figure(figsize=(10, 6))
    plotted = False

    for variant, data in series:
        history = data.get(history_key, [])
        if not history:
            continue
        arr = np.array(history, dtype=float)
        plt.plot(arr[:, x_index], arr[:, y_index], marker=marker,
                 label=variant, color=colors[variant], linewidth=2)
        plotted = True

    if not plotted:
        plt.close()
        return

    if optimal_reachability is not None:
        plt.axhline(y=optimal_reachability, color='red', linestyle='--', linewidth=2,
                    alpha=0.8, zorder=10,
                    label=f'Optimal Reachability = {optimal_reachability:.2f}')

    if vline_at is not None:
        plt.axvline(x=vline_at, color='green', linestyle=':', linewidth=2, label=vline_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.title(title)
    plt.legend()
    plt.grid()
    if note:
        annotate_collapsed_states(note)
    plt.savefig(plot_path)
    plt.close()


def plot_comparison_w_stdev(plot_path, series, history_key, y_index, stdev_key,
                            xlabel, ylabel, title, ylim, colors, x_index=0,
                            optimal_reachability=None, vline_at=None, vline_label=None, note=None):
    """
    Same as plot_comparison, with each variant's across-trial stdev shaded.

    The band is drawn in the variant's own colour rather than the viridis scale
    analyse_performance.py uses, because several variants share these axes and
    the colour has to say which curve a band belongs to.

    Args:
        stdev_key (str): Key in the benchmark's stdev_data.json holding the
            (k, stdev) pairs for this metric.

    Other arguments match plot_comparison.
    """
    plt.figure(figsize=(10, 6))
    plotted = False

    for variant, data in series:
        history = data.get(history_key, [])
        if not history:
            continue
        arr = np.array(history, dtype=float)
        x_values = arr[:, x_index]
        y_values = arr[:, y_index]

        # Align stdevs to the history by k, which both are indexed on.
        stdev_history = data.get('_stdev_data', {}).get(stdev_key, [])
        stdev_map = {int(k): float(s) for k, s in stdev_history}
        stdevs = np.array([stdev_map.get(int(k), 0.0) for k in arr[:, 0]], dtype=float)

        plt.plot(x_values, y_values, marker='o', label=variant,
                 color=colors[variant], linewidth=2)
        plt.fill_between(x_values, y_values - stdevs, y_values + stdevs,
                         color=colors[variant], alpha=0.2, linewidth=0)
        plotted = True

    if not plotted:
        plt.close()
        return

    if optimal_reachability is not None:
        plt.axhline(y=optimal_reachability, color='red', linestyle='--', linewidth=2,
                    alpha=0.8, zorder=10,
                    label=f'Optimal Reachability = {optimal_reachability:.2f}')

    if vline_at is not None:
        plt.axvline(x=vline_at, color='green', linestyle=':', linewidth=2, label=vline_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.title(title)
    plt.legend()
    plt.grid()
    if note:
        annotate_collapsed_states(note)
    plt.savefig(plot_path)
    plt.close()


def plot_value_bounds_comparison(plot_path, series, xlabel, title, colors,
                                 x_index=0, optimal_reachability=None):
    """
    Draw L(s0) and U(s0) for several variants on a single set of axes.

    Each variant keeps one colour: the lower bound is solid, the upper bound is
    dashed, and the gap between them is shaded.
    """
    plt.figure(figsize=(10, 6))
    plotted = False

    for variant, data in series:
        history = data.get('learning_history', [])
        if not history:
            continue
        arr = np.array(history, dtype=float)
        x_values = arr[:, x_index]
        lower_bounds = arr[:, -2]
        upper_bounds = arr[:, -1]

        plt.plot(x_values, lower_bounds, marker='o', linestyle='-', linewidth=2,
                 color=colors[variant], label=f'{variant} L(s0)')
        plt.plot(x_values, upper_bounds, marker='s', linestyle='--', linewidth=2,
                 color=colors[variant], label=f'{variant} U(s0)')
        plt.fill_between(x_values, lower_bounds, upper_bounds,
                         color=colors[variant], alpha=0.15)
        plotted = True

    if not plotted:
        plt.close()
        return

    if optimal_reachability is not None:
        plt.axhline(y=optimal_reachability, color='red', linestyle='--', linewidth=2,
                    alpha=0.8, zorder=10,
                    label=f'Optimal Reachability = {optimal_reachability:.2f}')

    plt.xlabel(xlabel)
    plt.ylabel("Value")
    plt.ylim(-0.1, 1.1)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    plt.close()


def compare_benchmark(output_dir, benchmark_name, series):
    """
    Write the full set of comparison plots for one benchmark.

    Args:
        output_dir (str): Directory to save the plots in.
        benchmark_name (str): Benchmark being plotted, used for the V* line.
        series (list): (variant_name, analysis_data_dict) in plotting order.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = variant_colors([variant for variant, _ in series])

    # Axis ceilings come from the largest model any variant reported, so the
    # same benchmark is drawn on one scale across all of them.
    max_states = max(data.get('max_states', 0) for _, data in series)
    max_transitions = max(data.get('max_transitions', 0) for _, data in series)

    optimal_reachability = OPTIMAL_POLICY_REACHABILITY.get(benchmark_name)

    # Where p_k reaches the true p_min. It is a property of the benchmark, not of
    # the variant, so the first variant that recorded it speaks for all of them.
    p_min_stage_k, p_min_label = None, None
    for _, data in series:
        if data.get('p_min_stage_k') is not None:
            p_min_stage_k = int(data['p_min_stage_k'])
            p_min_label = f"p_k <= p_min at k={p_min_stage_k}"
            if data.get('p_min_projected'):
                p_min_label += " (projected)"
            break

    # Several variants share these axes, so the caveat names the ones it is about.
    collapsed_variants = [variant for variant, data in series
                          if data.get('policy_states_collapsed')]
    collapsed_note = None
    if collapsed_variants:
        verb, curve = ("keys", "that curve") if len(collapsed_variants) == 1 else ("key", "those curves")
        collapsed_note = (f"Note: {', '.join(collapsed_variants)} {verb} the policy on MEC-collapsed "
                          "super-states, whose identity can change between stages.\n"
                          f"Churn and retention are indicative only for {curve}.")

    def policy_note(history_key):
        """The caveat, on the policy plots only."""
        if history_key in ('policy_churn_history', 'policy_retention_history'):
            return collapsed_note
        return None

    def p_min_vline(history_key, x_index):
        """The marker's x position: the stage itself, or its sample count."""
        if p_min_stage_k is None or history_key != 'policy_retention_history':
            return None
        if x_index == 0:
            return p_min_stage_k
        for _, data in series:
            for row in data.get(history_key, []):
                if int(row[0]) == p_min_stage_k:
                    return row[1]
        return None

    # (filename, history key, y column, x column, x label, y label, title, ylim, V* line)
    plots = [
        ("error_vs_k.png", 'learning_history', -3, 0,
         "Iteration", "Error (U - L)", "Learning Error History", (-0.1, 1.1), None),
        ("error_vs_samples.png", 'learning_history', -3, 1,
         "Number of Samples", "Error (U - L)", "Learning Error vs Number of Samples", (-0.1, 1.1), None),
        ("num_states_seen_vs_k.png", 'states_set_history', -1, 0,
         "Iteration", "Num Seen States", "Num Seen States History",
         (-0.1 * max_states, max_states * 1.1), None),
        ("num_states_seen_vs_samples.png", 'states_set_history', -1, 1,
         "Number of Samples", "Num Seen States", "Num Seen States vs Number of Samples",
         (-0.1 * max_states, max_states * 1.1), None),
        ("num_transitions_seen_vs_k.png", 'transitions_seen_history', -1, 0,
         "Iteration", "Num Seen Transitions", "Num Seen Transitions History",
         (-0.1 * max_transitions, max_transitions * 1.1), None),
        ("num_transitions_seen_vs_samples.png", 'transitions_seen_history', -1, 1,
         "Number of Samples", "Num Seen Transitions", "Num Seen Transitions vs Number of Samples",
         (-0.1 * max_transitions, max_transitions * 1.1), None),
        ("policy_accuracy_vs_k.png", 'policy_accuracy_history', -1, 0,
         "Iteration", "Policy Accuracy (Reachability)", "Policy Accuracy (Reachability) History",
         (-0.1, 1.1), optimal_reachability),
        ("policy_accuracy_vs_samples.png", 'policy_accuracy_history', -1, 1,
         "Number of Samples", "Policy Accuracy (Reachability)",
         "Policy Accuracy (Reachability) vs Number of Samples", (-0.1, 1.1), optimal_reachability),
        ("policy_churn_vs_k.png", 'policy_churn_history', -1, 0,
         "Iteration", "Fraction of States with Changed Action",
         "Policy Churn Between Consecutive Stages", (-0.1, 1.1), None),
        ("policy_churn_vs_samples.png", 'policy_churn_history', -1, 1,
         "Number of Samples", "Fraction of States with Changed Action",
         "Policy Churn vs Number of Samples", (-0.1, 1.1), None),
        ("policy_retention_vs_k.png", 'policy_retention_history', -1, 0,
         "Iteration", "Fraction of p_min-Stage Actions Kept",
         "Retention of p_min-Stage Actions", (-0.1, 1.1), None),
        ("policy_retention_vs_samples.png", 'policy_retention_history', -1, 1,
         "Number of Samples", "Fraction of p_min-Stage Actions Kept",
         "Retention of p_min-Stage Actions vs Number of Samples", (-0.1, 1.1), None),
    ]

    for filename, history_key, y_index, x_index, xlabel, ylabel, title, ylim, optimal in plots:
        plot_comparison(
            plot_path=os.path.join(output_dir, filename),
            series=series,
            history_key=history_key,
            y_index=y_index,
            x_index=x_index,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            ylim=ylim,
            colors=colors,
            optimal_reachability=optimal,
            vline_at=p_min_vline(history_key, x_index),
            vline_label=p_min_label,
            note=policy_note(history_key),
        )

    # (filename, history key, y column, stdev key, x column, x label, y label, title, V* line)
    stdev_plots = [
        ("error_w_std_vs_k.png", 'learning_history', -3, 'error_stdev_history', 0,
         "Iteration", "Error (U - L)", "Learning Error History", None),
        ("error_w_std_vs_samples.png", 'learning_history', -3, 'error_stdev_history', 1,
         "Number of Samples", "Error (U - L)", "Learning Error vs Number of Samples", None),
        ("policy_accuracy_w_std_vs_k.png", 'policy_accuracy_history', -1,
         'policy_accuracy_stdev_history', 0, "Iteration", "Policy Accuracy (Reachability)",
         "Policy Accuracy (Reachability) History", optimal_reachability),
        ("policy_accuracy_w_std_vs_samples.png", 'policy_accuracy_history', -1,
         'policy_accuracy_stdev_history', 1, "Number of Samples", "Policy Accuracy (Reachability)",
         "Policy Accuracy (Reachability) vs Number of Samples", optimal_reachability),
        ("policy_churn_w_std_vs_k.png", 'policy_churn_history', -1, 'policy_churn_stdev_history', 0,
         "Iteration", "Fraction of States with Changed Action",
         "Policy Churn Between Consecutive Stages", None),
        ("policy_churn_w_std_vs_samples.png", 'policy_churn_history', -1, 'policy_churn_stdev_history', 1,
         "Number of Samples", "Fraction of States with Changed Action",
         "Policy Churn vs Number of Samples", None),
        ("policy_retention_w_std_vs_k.png", 'policy_retention_history', -1,
         'policy_retention_stdev_history', 0, "Iteration", "Fraction of p_min-Stage Actions Kept",
         "Retention of p_min-Stage Actions", None),
        ("policy_retention_w_std_vs_samples.png", 'policy_retention_history', -1,
         'policy_retention_stdev_history', 1, "Number of Samples",
         "Fraction of p_min-Stage Actions Kept",
         "Retention of p_min-Stage Actions vs Number of Samples", None),
    ]

    for filename, history_key, y_index, stdev_key, x_index, xlabel, ylabel, title, optimal in stdev_plots:
        plot_comparison_w_stdev(
            plot_path=os.path.join(output_dir, filename),
            series=series,
            history_key=history_key,
            y_index=y_index,
            stdev_key=stdev_key,
            x_index=x_index,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            ylim=(-0.1, 1.1),
            colors=colors,
            optimal_reachability=optimal,
            vline_at=p_min_vline(history_key, x_index),
            vline_label=p_min_label,
            note=policy_note(history_key),
        )

    plot_value_bounds_comparison(
        plot_path=os.path.join(output_dir, "value_bounds_vs_k.png"),
        series=series,
        xlabel="Iteration (k)",
        title="Value Bounds for Initial State vs Iteration",
        colors=colors,
        x_index=0,
        optimal_reachability=optimal_reachability,
    )
    plot_value_bounds_comparison(
        plot_path=os.path.join(output_dir, "value_bounds_vs_samples.png"),
        series=series,
        xlabel="Number of Samples",
        title="Value Bounds for Initial State vs Number of Samples",
        colors=colors,
        x_index=1,
        optimal_reachability=optimal_reachability,
    )


def compare_variants(variants, analysis_dir='./analysis'):
    """
    Draw comparison plots for every benchmark the given variants have in common.

    Args:
        variants (list): Variant names to compare.
        analysis_dir (str): Root analysis directory.

    Returns:
        str: The output directory the plots were written to.
    """
    # The output folder names the variants in alphabetical order, so comparing
    # the same set in a different order lands in the same place.
    ordered_variants = sorted(set(variants))
    output_root = os.path.join(analysis_dir, '_'.join(ordered_variants))

    # With a single variant the output folder name is that variant's own folder,
    # so writing there would overwrite the per-variant plots analyse_performance.py
    # made. Refuse rather than clobber them.
    for variant in ordered_variants:
        if os.path.abspath(output_root) == os.path.abspath(os.path.join(analysis_dir, variant)):
            raise SystemExit(
                f"Refusing to write into {output_root}: that is variant '{variant}'s own "
                f"results folder. Pass at least two variants to compare."
            )

    loaded = {variant: load_variant(analysis_dir, variant) for variant in ordered_variants}

    all_benchmarks = sorted({name for benchmarks in loaded.values() for name in benchmarks})
    if not all_benchmarks:
        print("No benchmark data found for any of the requested variants.")
        return output_root

    for benchmark_name in all_benchmarks:
        series = [(variant, loaded[variant][benchmark_name])
                  for variant in ordered_variants if benchmark_name in loaded[variant]]

        missing = [variant for variant in ordered_variants if benchmark_name not in loaded[variant]]
        if missing:
            print(f"Note: {benchmark_name} has no data for {', '.join(missing)}; "
                  f"plotting the {len(series)} variant(s) that do.")

        output_dir = os.path.join(output_root, benchmark_name)
        compare_benchmark(output_dir, benchmark_name, series)
        print(f"{benchmark_name} comparison plots saved to {output_dir}")

    return output_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across several learner variants."
    )
    parser.add_argument(
        "-V",
        "--variants",
        type=str,
        nargs='+',
        required=True,
        choices=variant_names(),
        help="Variants to compare; each is read from analysis/<variant>/. " + variant_help()
    )
    parser.add_argument(
        "-d",
        "--analysis_dir",
        type=str,
        default="./analysis",
        help="Root analysis directory to read from and write to (default: './analysis')."
    )

    args = parser.parse_args()

    print("Comparing variants: " + ", ".join(sorted(set(args.variants))))
    output_root = compare_variants(args.variants, analysis_dir=args.analysis_dir)
    print(f"Comparison complete. Output written under {output_root}")
