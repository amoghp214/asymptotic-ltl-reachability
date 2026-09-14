"""
The live model-free learner.

This module deliberately re-exports the class from model_free_tdql_learner.py
rather than holding a copy of it. That file is the one the repo runs, the one
the test suite covers, and the one `windowed` and `windowed_sweep_saturated`
subclass. A copy here would be a second definition that drifts the first time
production is edited, and `--variant current` would quietly stop running the
learner it names.

The frozen snapshots in this folder are `original.py` (2026-09-01) and
`cumulative.py` (2026-09-10). Those are copies on purpose: they record states
the code was in and are not meant to track it.
"""

from model_free_tdql_learner import ModelFreeTDQLearner, TDQL_PAC_Stage

__all__ = ["ModelFreeTDQLearner", "TDQL_PAC_Stage"]
