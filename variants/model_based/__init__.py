"""
The model-based approach: build an explicit estimate of the MDP from samples,
then run bounded value iteration on that estimate.

This family openly maintains a transition model -- that is the whole design, not
a leak. Contrast model_free/, where holding transition counts is a property we
have to argue about.
"""
