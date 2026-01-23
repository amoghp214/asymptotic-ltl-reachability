# LTL-Reachability: PAC Statistical Model Checking

Implementation of novel algorithm to guarantee asymptotic reachability of LTL specifications for black box models. Theoretically, the optimal policy is found with 100% confidence. Research submitted to ICML 2026.

This codebase provides an end-to-end framework for learning reachability objectives in Markov Decision Processes with **Probably Approximately Correct (PAC)** guarantees, requiring only:
- A simulator/environment (no knowledge of transition function)
- An error tolerance parameter (δ)
- A lower bound on minimum transition probability (p_min)

## Features

✅ **All Algorithms Implemented:**
1. **ErrorBound** - Calculate margin of error for probability estimates
2. **DistributeError** - Distribute error budget across state-action pairs
3. **LowerBound** - Compute lower bound on transition probability
4. **EstimateTransition** - Estimate transition probabilities from samples
5. **UpdateBVIBounds** - Update value bounds for a single state (Bounded Value Iteration)
6. **BVI** - Bounded Value Iteration for all states
7. **NumSamplesEC** - Calculate samples needed for end component detection
8. **IsStayingPair** - Check if state-action pair stays within a set
9. **GetStayingPairs** - Get all staying state-action pairs
10. **ConfidentlyDetectEC** - Detect end components with PAC guarantees
11. **FindSCCs** - Find strongly connected components (Tarjan's algorithm)
12. **GetStayingMDP** - Extract sub-MDP with staying transitions
13. **FindAllECs** - Recursively find all end components
14. **DetectLoop** - Detect loops that prevent goal reachability

## Project Structure

```
ltl-reachability/
├── mdp.py                  # MDP class with state, action, and transition management
├── algorithm.py            # Core algorithms (all 14 algorithms)
├── main.py                 # End-to-end learner and examples
├── test_algorithms.py      # Comprehensive unit tests
├── logs/                   # Execution logs and learning traces
│   └── *.log               # Timestamped log files with algorithm progress
├── mdp_models/             # Discovered MDP models (JSON format)
│   └── *.json              # Learned transition functions from simulations
├── plots/                  # Visualization outputs
│   ├── *.png               # Generated plots and graphs
│   └── *.pdf               # High-quality plot exports
├── results/                # Final numerical results and statistics
│   ├── *.csv               # Tabular results for analysis
│   └── *.json              # Structured result data
├── README.md               # This file
└── LICENSE
```

## Installation

### Using Conda (Recommended)

Create a conda environment with all dependencies:

```bash
conda env create -f environment.yaml
conda activate ltl-reachability
```

To update an existing environment:
```bash
conda env update -f environment.yaml --prune
```

### Manual Installation

```bash
pip install numpy matplotlib tqdm
```

## Quick Start

### Running the Main Algorithm

The `main.py` file contains the complete learning pipeline with several built-in examples:

```bash
python main.py
```

This executes three example scenarios:

1. **Simple 3-State MDP** - Basic example with two actions
   - Demonstrates core reachability learning
   - Shows how to set up a simple simulator
   
2. **IJ 3 PRISM Baseline** - PRISM case study task with stochasticity
   - 32 states
   - Lesser complex state-action space
   - Toy example
   
3. **IJ 10 PRISM Baseline** - PRISM benchmark task with stochasticity
   - 1024 states
   - Complex task
   - Requires millions of samples to start converging

**Output**: The script generates:
- Console output with learning progress and final results
- Log files in `logs/` directory with detailed outputs regarding the analysis and state of the partial MDP discovered
- Ground truth (baseline) MDP models in `mdp_models/` (JSON/JANI format)
- Plots of learning curves and performance metrics in `results/` directory

### Basic Usage Example

```python
from main import MDPSimulator, LTLReachabilityLearner
import numpy as np

# Create a simulator for your MDP
simulator = MDPSimulator()
simulator.add_transition(0, 'action1', 1, 0.8)  # State 0 → State 1 with prob 0.8
simulator.add_transition(0, 'action1', 0, 0.2)  # Self-loop with prob 0.2
# ... add more transitions ...

# Create learner with PAC parameters
learner = LTLReachabilityLearner(mdp_simulator=mdp_sim)

# Learn the model from simulator
analysis_path = "./results/ij3_analysis"
learner.learn(analysis_path)

# Print MDP learning summary
learner.print_summary(true_confidence_error=0.01, true_p_min=0.5, output_path="./logs/ij3_learning_log.json")
```

### Run Tests

```bash
python test_mdp.py
python test_ltl_reachability_learner.py
```

The test suite validates:
- ✅ Probability estimation
- ✅ Bounded value iteration
- ✅ End component detection
- ✅ SCC algorithms
- ✅ Loop detection
- ✅ MDP class functionality

## Understanding Output Directories

### `logs/` Directory

Contains details of partial MDPs discovered during the learning phase. Each log file records all the instance variables of the partial MDP. This includes:
- Topology
- Transition Probabilities
- States
- Goal States
- State Action Pairs
- State Value Bounds
- State-Action Value Bounds
- Sample Counts
- Learned Policy
- Initial State
- State to MEC Mapping
- MEC to States Mapping
- Confidence Error
- Minimum Transition Probability
- Error
- Learning Error History
- Seen States History
- Seen Transitions History
- Policy Accuracy History

**Usage**: Check logs for debugging, understanding learning dynamics, and verifying PAC guarantees are being maintained.

**Example log file**: `logs/ltl_reachability_20250115_143022.log`

### `mdp_models/` Directory

Stores ground truth (baseline) MDP models in JSON format after learning completes.

**File format**:
```json
{
  "states": [0, 1, 2, ...],
  "goal_states": [5],
  "transitions": {
    "0": {
      "action1": {
        "1": 0.8,
        "0": 0.2
      }
    }
  },
  "value_bounds": {
    "0": [0.2, 0.5],
    "1": [0.4, 0.7]
  }
}
```

**Usage**: 
- Inspect true models for a better understanding of the learning requirements
- Sanity checks for benchmarks

### `results/` Directory

Visualization outputs including:
- **Reachability bounds plots**: Shows error (epsilon) of algorithm throughout the learning process
- **Seen states over time**: The number of states seen in the partial MDP over algorithmic iterations and sampling
- **Seen transitions over time**: The number of transitions seen in the partial MDP over algorithmic iterations and sampling
- **MDP policy accuracy over time**: The accuracy of the learned MDP policy over algorithmic iterations and sampling

**Formats**: PNG (quick viewing)

**Usage**: Analyze learning behavior, check for algorithm convergence, and present results.

<!-- ## Algorithm Overview

### Core Concepts

**PAC Learning**: We want to compute reachability probability $P(s \xrightarrow{*} T)$ with:
- Confidence: With probability ≥ 1 - δ
- Accuracy: Within error ε of true value

**Key Insight**: Instead of learning the full transition function, we maintain **lower bounds** on transition probabilities. This is sufficient for determining reachability!

### Main Learning Loop

```
1. Initialize discovered MDP as empty
2. For each sample iteration:
   a. Choose state-action pair (exploration strategy)
   b. Sample next state from environment
   c. Update transition probability estimates with error bounds
   d. Detect and deflate end components (loops without goal)
   e. Optionally reset if goal reached
3. Final step: Run Bounded Value Iteration to compute reachability bounds
```

### Bounded Value Iteration (BVI)

For each non-goal state $s$, we maintain bounds $[\underline{V}(s), \overline{V}(s)]$:

$$\overline{V}(s) = \max_a \left[ \sum_s' \underline{P}(s' | s,a) \overline{V}(s') + (1 - \sum_s' \underline{P}(s' | s,a)) \cdot 1 \right]$$

Where $\underline{P}(s' | s,a)$ are the lower bounds on transition probabilities.

### End Component Detection

An **end component** is a set of states from which the agent cannot escape. We need to identify these to avoid overestimating reachability.

The algorithm:
1. Find strongly connected components (Tarjan's algorithm)
2. Check if components are "staying" (all transitions remain inside)
3. Verify with sufficient samples (NumSamplesEC)


## Algorithms Details

### Probability Estimation

These establish error bounds using Chernoff inequalities:
- Given n samples of a probability p
- We can estimate p with margin of error: $c = \sqrt{\ln(1/\delta) / (2n)}$

### Bounded Value Iteration

Instead of learning exact probabilities, we work with lower bounds and compute upper/lower bounds on value functions iteratively.

### Loop Detection and Avoidance

Essential for stochastic systems where the agent might get trapped in cycles. We:
1. Detect potential loops via SCC analysis
2. Require sufficient samples to confirm
3. Deflate ECs by marking them as unrewarding -->

## Algorithm Parameters

**error_tolerance (δ)**: 
- Smaller → More samples needed for same accuracy
- Typical values: 0.01 to 0.1

**p_min**:
- Minimum transition probability to detect
- If system has transitions with probability < p_min, algorithm may miss them
- Typical values: 0.01 to 0.1


## Example: Simple 3-State MDP

States: 0, 1, 2 (2 is goal)

```
State 0:
  - Action 'a': 70% → State 1, 30% → State 0
  - Action 'b': 100% → State 0

State 1:
  - Action 'a': 60% → State 2, 40% → State 1
  - Action 'b': 50% → State 0, 50% → State 1

State 2: (goal)
  - Both actions loop back to 2
```

Expected reachability from 0 to 2: ≈ 0.42


## Notes

- Algorithm works with **unknown** transition functions
- Only requires ability to sample from environment
- PAC guarantees hold with high probability
- Handles stochastic environments
- Automatically detects and avoids traps (end components)

## License

See LICENSE file for details.
