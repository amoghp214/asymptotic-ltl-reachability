"""
The live model-based learner.

This module deliberately re-exports the class from ltl_reachability_learner.py
rather than holding a copy of it. That file is still imported directly by
main.py's example functions, so a copy here would be a second definition that
drifts the first time either one is edited, and `--variant model-based` would
quietly stop running the learner the repo actually uses.

If you want a frozen snapshot of the model-based learner to compare against
later, add it here as a separate, dated file -- do not turn this one into a copy.
"""

from ltl_reachability_learner import LTLReachabilityLearner

__all__ = ["LTLReachabilityLearner"]
