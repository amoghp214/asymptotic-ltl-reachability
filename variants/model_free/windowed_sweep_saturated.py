"""
The windowed learner plus the two round-level changes measured on 2026-09-10.

  sweep_each_round  -- sweep backwards over the predecessor index at the end of
                       a round, before the evidence is cleared, so a round's
                       samples reach more than the layer that produced them.
  skip_saturated    -- end an episode once every action at the current state has
                       filled its window, rather than drawing samples that
                       cannot move a bound.

Measured against plain `windowed` at 5M and 50M samples, the two together are a
wash: differences sit inside the run-to-run spread, and the same four models
stay at gap 1.0. Kept as a variant so the result stays reproducible.
"""

from model_free_tdql_learner import ModelFreeTDQLearner as _BaseLearner


class ModelFreeTDQLearner(_BaseLearner):

    def make_stage(self, **kwargs):
        stage = super().make_stage(**kwargs)
        stage.reset_evidence_each_round = True
        stage.sweep_each_round = True
        stage.skip_saturated = True
        return stage
