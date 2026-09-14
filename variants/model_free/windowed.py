"""
The current learner, restricted to a bounded window of trajectory evidence.

Successor counts are cleared at the end of every round, so the learner never
holds more history than the round it is in. This is the configuration that is
model-free in the strict sense: nothing accumulates into an estimate of the
transition function across rounds.

It converges far more slowly. Four of the ten benchmark models never leave
gap 1.0 at any budget measured -- see README.md.
"""

from model_free_tdql_learner import ModelFreeTDQLearner as _BaseLearner


class ModelFreeTDQLearner(_BaseLearner):

    def make_stage(self, **kwargs):
        stage = super().make_stage(**kwargs)
        stage.reset_evidence_each_round = True
        return stage
