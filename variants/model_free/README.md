# model_free/

What each file is, what it changed, and what that cost.

| file | what it is | what changed vs. the row above it | what improved | what was bad |
|---|---|---|---|---|
| `original.py` | the learner as of **2026-09-01**, before any of this work | -- | -- | **0 of 81 updates ever fired**: `c` was hard-coded to `0.0001`, so no bound could move. Intermittently **unsound** (L > U on six_state at 50M, not at 5M or 500M). No zero-progress guard, so it can **hang** when U(s0) reaches 0. Its own `print_summary` raises `AttributeError: 'ModelFreeTDQLearner' object has no attribute 'discovered_mdp'` after learning finishes -- left unpatched so the copy stays faithful. |
| `current.py` | re-export of the live `model_free_tdql_learner.py` | Hoeffding/empirical-Bernstein radii replacing the constant `c`; tables persist across stages via `prev_stage`; `H` and batch sizes recomputed; trajectory cache `succ`/`preds`; EC detection and downward-only deflation fixed; `U >= L` guard on every update | bounds actually converge, and all runs on the ten-model suite are sound | keeps every successor count ever seen, which is what makes it not model-free (below) |
| `cumulative.py` | frozen copy of `current` at **2026-09-10** | none -- it *is* `current` at that date, kept as the reference the windowed arm is measured against | best gaps on record (table below) | **not model-free.** `attempt_update` computes `Σ p̂(s'\|s,a)·f(s')` from counts that are never reset. On philosophers and consensus.2 it recovers the true transition distribution to within 0.01 in L1 for >90% of visited pairs, median error 0 |
| `windowed.py` | `current` with `reset_evidence_each_round = True` | successor counts cleared at the end of every round | **model-free in the strict sense** -- nothing accumulates into a transition estimate across rounds | far slower. **Four of ten models never leave gap 1.0** at any budget measured; pacman 82x worse than cumulative. Cause: clearing the evidence floors the radius at `c(m) ~ eps_k/2`, so deep-goal models can never fire an update |
| `windowed_sweep_saturated.py` | `windowed` plus `sweep_each_round` and `skip_saturated` | end-of-round backward sweep over `preds` before the evidence is cleared; episode ends once every action at the current state has filled its window | nothing that survives the noise floor | **a wash on bounds, and 7-10x slower at 500M** (table below). Differences vs. plain `windowed` sit inside the run-to-run spread (`MDP.topology` is a set, so action order is hash-randomised), and the same four models stay at 1.0. Both changes redistribute samples *within* a round; neither touches the radius floor that is actually binding |

All variants keep the same simulator interface: reset to s0, ask for the actions
in the current state, sample an action in the current state. `original.py` is the
exception only in that it was never audited against `StrictSimulator`.

## Gap at s0, equal sample budgets

`cumulative` (best on record):

| model         | depth |     5M |    50M |   500M |
|---------------|------:|-------:|-------:|-------:|
| 6-state       |     1 | 0.0040 | 0.0010 | 0.0003 |
| ij.3          |     2 | 0.0069 | 0.0023 | 0.0007 |
| philosophers  |     4 | 0.0280 | 0.0061 | 0.0015 |
| rabin.3       |     4 | 0.0498 | 0.0173 | 0.0051 |
| pacman        |     7 | 0.1574 | 0.0135 | 0.0028 |
| ij.10         |     9 | 1.0000 | 0.6424 | 0.1903 |
| consensus.2   |    12 | 0.3737 | 0.0967 | 0.0268 |
| zeroconf      |    17 | 0.0305 | 0.0086 | 0.0028 |
| csma.2-2      |    47 | 0.9766 | 0.2728 | 0.0527 |
| firewire      |    76 | 0.8690 | 0.1057 | 0.0323 |

`windowed` vs. `windowed_sweep_saturated`, measured 2026-09-10:

| model         | depth | 5M win | 5M w+s+s | 50M win | 50M w+s+s | 500M win | 500M w+s+s |
|---------------|------:|-------:|---------:|--------:|----------:|---------:|-----------:|
| 6-state       |     1 | 0.0088 |   0.0087 |  0.0023 |    0.0024 |   0.0007 |    running |
| ij.3          |     2 | 0.0176 |   0.0175 |  0.0059 |    0.0059 |   0.0013 |    running |
| philosophers  |     4 | 0.1441 |   0.1383 |  0.0189 |    0.0331 |   0.0069 |     0.0069 |
| rabin.3       |     4 | 0.2144 |   0.2339 |  0.0993 |    0.0993 |   0.0737 |     0.0738 |
| pacman        |     7 | 0.4309 |   0.3625 |  0.2491 |    0.2608 |   0.2311 |    running |
| ij.10         |     9 | 1.0000 |   1.0000 |  1.0000 |    1.0000 |   1.0000 |     1.0000 |
| consensus.2   |    12 | 1.0000 |   1.0000 |  1.0000 |    1.0000 |   1.0000 |    running |
| zeroconf      |    17 | 0.1462 |   0.1494 |  0.0314 |    0.0351 |   0.0084 |     0.0085 |
| csma.2-2      |    47 | 1.0000 |   1.0000 |  1.0000 |    1.0000 |   1.0000 |     1.0000 |
| firewire      |    76 | 1.0000 |   1.0000 |  1.0000 |    1.0000 |   1.0000 |    running |

Five of the ten w+s+s runs at 500M were still going when this was written.

### The second problem with w+s+s: it is far slower

Where both arms finished at 500M, they agree to four decimals and w+s+s took
7-10x the wall clock:

| model        | window | w+s+s | same gap? |
|--------------|-------:|------:|-----------|
| philosophers |  2,825s | 32,921s | yes, 0.0069 |
| rabin.3      |  4,397s | 34,658s | yes, 0.0737 / 0.0738 |
| csma.2-2     |  4,055s | 33,588s | yes, 1.0000 |
| ij.10        |  5,002s | 38,049s | yes, 1.0000 |

`sweep_each_round` does O(|SA|) extra work per round and the round count grows
with the budget, so the cost compounds -- 12% slower at 50M on rabin.3, 8x at
500M. Caveat: the arms were not timed side by side in one controlled run (both
were 7 concurrent workers, but on different occasions), so read the ratio as
indicative rather than measured.

All runs above are sound: L(s0) <= V*(s0) <= U(s0), L <= U at every pair, and
max_a U(s,a) >= V(s) at every state. Records in `results/full_comparison/`.

`original.py` is not in these tables because it produces no movement to measure.
What it does produce is unearned: on ij.3 at k=1 it spends 852,310 samples and
reports an interval of width 0.00057, because `c` is hard-coded to `0.0001`.
The live learner reports 0.42 at the same point and narrows from there.

## Tests

`cumulative_test.py` covers `cumulative.py`. Note that before this reorganisation
the same file imported `TDQL_PAC_Stage` from the **production** module, so it was
testing the live learner rather than the frozen copy it is named after; the import
now points at `cumulative.py`.

The live learner's own suite is `model_free_tdql_learner_test.py` at the repo root.
