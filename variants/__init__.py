"""
Registry of learner variants.

Two families live here, one folder each:

  model_based/  the approach that builds an explicit MDP estimate and runs
                bounded value iteration on it (LTLReachabilityLearner).
  model_free/   the TDQL ladder, which never materialises a transition matrix
                as a model object -- though see model_free/README.md, because
                one of its variants does recover transition probabilities.

Each name below maps to a module exporting a learner class whose constructor
takes the same arguments as LTLReachabilityLearner and ModelFreeTDQLearner, and
which provides learn(analysis_dir) and print_summary(...). That is all main.py
needs, so adding a variant is a file plus one line here.
"""

from importlib import import_module

#: name -> (module path, exported class, one-line description for --help)
VARIANTS = {
    # --- model-free ---------------------------------------------------------
    "current": (
        "variants.model_free.current",
        "ModelFreeTDQLearner",
        "the live learner in model_free_tdql_learner.py with its default flags",
    ),
    "original": (
        "variants.model_free.original",
        "ModelFreeTDQLearner",
        "the learner as of 2026-09-01, before any of this work; does not converge, may hang",
    ),
    "optimized": (
        "variants.model_free.optimized",
        "ModelFreeTDQLearner",
        "the original with every model-free optimisation applied; standalone",
    ),
    "cumulative": (
        "variants.model_free.cumulative",
        "ModelFreeTDQLearner",
        "successor counts never reset; best gaps on record, but NOT model-free",
    ),
    "windowed": (
        "variants.model_free.windowed",
        "ModelFreeTDQLearner",
        "successor counts cleared every round; model-free, much worse gaps",
    ),
    "windowed-sweep-saturated": (
        "variants.model_free.windowed_sweep_saturated",
        "ModelFreeTDQLearner",
        "windowed, plus the end-of-round backward sweep and the saturated-state episode exit",
    ),
    # --- model-based --------------------------------------------------------
    "model-based": (
        "variants.model_based.current",
        "LTLReachabilityLearner",
        "the live model-based learner in ltl_reachability_learner.py",
    ),
}


def variant_names():
    """Names accepted by --variant, in registry order."""
    return list(VARIANTS)


def variant_help():
    """One string describing every variant, for the --help text."""
    return " | ".join(f"'{name}' = {desc}" for name, (_, _, desc) in VARIANTS.items())


def resolve_learner(name):
    """
    Import the module registered under `name` and return its learner class.

    Raises:
        KeyError: if `name` is not registered.
    """
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; choose from {', '.join(VARIANTS)}")
    module_path, class_name, _ = VARIANTS[name]
    return getattr(import_module(module_path), class_name)
