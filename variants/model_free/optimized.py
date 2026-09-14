"""
A model-free optimisation of the original TDQL PAC learner.

This file starts from `original.py` (the learner as of 2026-09-01) and applies
only those changes that keep the algorithm model-free and add no assumption
about the MDP. It is standalone: nothing imports it and it imports nothing from
the rest of the learner family, so it can be read on its own and diffed against
`original.py` to see exactly what changed.

THE RULE APPLIED
----------------
A change was accepted only if both hold:

  * It stores nothing from which a transition probability can be recovered. The
    learner may hold bounds, counters, and running sums of *bound values* at
    sampled successors. It may not hold, for any (s,a), which successors were
    seen or how often.
  * It assumes nothing about the MDP that the original did not. No knowledge of
    end components, of p_min beyond the parameter already passed in, of the
    state space size, or of the transition structure.

EVERY OPTIMISATION CONSIDERED
-----------------------------
Applied (each marked `DEVIATION n` at its site below):

  1  calculate_c            the radius is computed from the batch actually
                            collected instead of being the constant 0.0001.
                            This is a soundness fix, not a speed-up.
  2  calculate_c_bernstein  an empirical-Bernstein radius computed from the
                            running sum and sum of squares of the sampled bound
                            values. Model-free: it is the variance of the m
                            numbers observed, not of any transition estimate.
  3  calculate_m            batch size from a Hoeffding/Bernstein budget rather
                            than 10(10+|SA|)/delta^2, which no pair ever fills.
  4  calculate_num_b_hat    the DQL update budget is per pair, so it scales with
                            the number of pairs.
  5  calculate_H, _Z        horizon linear in the discovered state count rather
                            than quadratic in |SA|.
  6  calculate_eps_u        smallest committed improvement tied to this stage's
                            accuracy rather than held at 0.001/2^k.
  7  calculate_ec_patience  closure-walk patience sized by the component's cover
                            time instead of a flat 100 per pair.
  8  detect_ec              close the component over observed successors, and
                            stop the walk at goal states.
  9  deflate_ec             deflation may only lower U; goal-ness and action
                            sets are read from what the walk recorded, not from
                            the ground-truth model. The latter is itself a
                            model-freeness fix -- the original queried the model
                            about states it was only remembering.
 10  simulate_and_update    a write is refused if it would put U below L.
 11  run                    the stage ends when a full sweep moves nothing, and a
                            round that draws no samples ends it outright.
 12  run, __init__          bound writes land in place and the per-pair evidence
                            persists across rounds, instead of a shadow table
                            swapped in at a round barrier and aggregates rebuilt
                            from scratch each round. This is the change that most
                            affects how fast a bound travels.
 13  simulate_and_update    an episode ends when it stands on a state whose U is
                            already 0, instead of burning its whole horizon in a
                            region the algorithm has already finished with.
 14  print_summary          report the learned bounds. The inherited version reads
                            a self.discovered_mdp that a model-free learner never
                            builds, so it raises AttributeError on every run.

Rejected, with the reason:

  X  successor cache succ[(s,a)][s']            -- this IS a transition model.
     It is a maximum-likelihood estimate of P(s'|s,a) and the update built on it
     computes sum_s' p_hat(s'|s,a) f(s'), a Bellman backup against that model.
     Measured on philosophers and consensus.2 it recovers the true transition
     distribution to within 0.01 in L1 for >90% of visited pairs. Rejected
     outright, in both its cumulative and its per-round-windowed form.

  X  Weissman / per-cell Bernstein radius       -- both are functions of the
     successor histogram and its support size, so they cannot be computed
     without the rejected cache. The Bernstein idea survives in a model-free
     form as DEVIATION 2.

  X  predecessor index preds[s'] = {(s,a)}      -- stores learned transition
     structure (which pairs lead where). Even without probabilities that is a
     model of the topology, and planning over it is planning over a model.

  X  backward sweep over preds                  -- needs the index above, and
     measured to buy nothing: at 500M samples it produced gaps identical to four
     decimals while taking 7-10x the wall clock.

  X  saturated-state episode exit               -- depends on the windowed
     evidence machinery, and measured to make no difference outside the
     run-to-run spread.

  X  carrying bounds across stages              -- stores no model and assumes
     nothing about the MDP, so on the letter it passes the rule above, but it
     departs from the pseudocode, which gives each stage its own tables. The
     stages are meant to be independent and every stage here starts from U = 1,
     L = 0, as the original does. This costs real convergence speed: each stage
     re-derives what the one before it had already established. That cost is
     accepted deliberately rather than traded for a guarantee the proof as
     written does not make.

KNOWN CAVEAT, PART INHERITED AND PART ADDED
-------------------------------------------
u_agg accumulates U at the greedy successor, and greedy sets are refreshed as
bounds are written, so the function being averaged can change part-way through a
batch. Hoeffding and Bernstein are applied as though it were fixed. The original
already had this -- it refreshes greedy sets on every write -- and DEVIATION 12
makes it more frequent by letting writes land immediately rather than at a round
barrier. It is recorded here so nobody mistakes the radius for something
stronger than it is: the radii are honest about the batch size, not about the
batch being drawn from a stationary function.

This is a statement about the *analysis*, not about model-freeness or about the
assumptions on the MDP, which are unchanged.
"""

import copy
import math
import random
from typing import Set, Tuple, List, Optional, Any
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from mdp import MDP
from mdp_simulator import MDPSimulator
from analysis_utils import run_analysis, plot_bvi_history
from tqdm import tqdm
import json
import os
import time


class TDQL_PAC_Stage:
    """
    Class to encapsulate all variales and methods for a single TDQL PAC stage.
    """

    def __init__(
        self,
        k: int,
        error: float,
        confidence_error: float,
        p_min: float,
        mu: float,
        mdp_simulator: MDPSimulator,
    ):
        self.k = k
        self.error = error
        self.confidence_error = confidence_error
        self.p_min = p_min
        self.mu = mu
        self.mdp_simulator = mdp_simulator
        self.s0 = self.mdp_simulator.gt_mdp.initial_state
        self.sample_count = 0

        # Goal-ness of every state the walk has actually stood in, recorded at visit
        # time. Used by deflate_ec so that it never has to ask the model about a
        # state it is only remembering (DEVIATION 9).
        self.goal_seen = set()

        self.seen_states = set() # Set of all discovered states in the current PAC stage
        self.seen_state_action_pairs = set() # Set of all discovered state-action pairs in the current PAC stage
        self.ecs = set() # Set of all detected ECs in the current PAC stage
        self.corresponding_ecs = dict() # {s: ec, ...} mapping of states to their corresponding ECs
        self.u = dict() # {s: {a: upper_bound, ...}, ...}
        self.u[self.s0] = dict()
        self.l = dict() # {s: {a: lower_bound, ...}, ...}
        self.l[self.s0] = dict()

        # Round-scoped tables. Rebound at the top of every round in run(); kept in
        # sync inside simulate_and_update so refresh_best_actions() can read them.
        self.u_new = self.u
        self.l_new = self.l

        self.best_actions = dict() # {s: [best_actions], ...}

        # DEVIATION 12: the per-pair evidence lives on the stage, not on the round.
        # In the original these are rebuilt from scratch at the top of every round, so
        # the partial batch of every pair except the one that triggered the commit is
        # thrown away -- on a model of any size, nearly all of the evidence collected.
        self.u_agg = dict()
        self.l_agg = dict()
        self.u_agg_sq = dict()
        self.l_agg_sq = dict()
        self.curr_counts = dict()
        self.num_updates = 0

        self.eps_u = self.calculate_eps_u(self.k)
        self.m = self.calculate_m()

    def calculate_log_term(self):
        """log(2 / delta_sa): this stage's confidence split over the pairs seen so far."""
        n = max(1, len(self.seen_state_action_pairs))
        return math.log(2.0 * n / self.confidence_error)

    def calculate_m(self):
        """
        DEVIATION 3: batch size from a concentration budget.

        The original asks for 10(10 + |SA|)/delta^2 samples of a single pair before
        it will attempt one update. At delta = 1/2 on a 400-pair model that is
        16,400 samples for one pair, and delta halves every stage; no pair ever
        fills a batch, which is why the original fires no updates at all on most
        models. Hoeffding needs only log(2/delta_sa) / (2 * accuracy^2) to estimate
        a mean in [0,1] to within `accuracy` -- logarithmic in |SA| and in 1/delta.

        The Bernstein branch is a gamble, not an assumption: it sizes the batch for
        a pair whose sampled values turn out to have little variance. Nothing about
        the MDP is assumed -- if the variance is in fact high, the radius computed
        at write time stays wide, the write is refused, and the batch is spent
        without moving a bound. It can cost samples. It cannot make a bound wrong.
        """
        accuracy = max(self.error / 2.0, 1e-6)
        hoeffding = math.ceil(self.calculate_log_term() / (2.0 * accuracy ** 2))
        bernstein = math.ceil(3.0 * math.log(3.0 / self.confidence_error) / accuracy)
        return int(max(1, min(hoeffding, bernstein)))

    def calculate_num_b_hat(self, err_u):
        """
        DEVIATION 4: the DQL update budget is per pair, not global.

        Each bound lives in [0,1] and every commit moves it by at least eps_u, so
        one pair can commit at most 1/eps_u times. That is a per-pair bound; the
        original applies it to the stage as a whole, which throttles the stage to a
        handful of writes and ends it long before a lower bound has had time to
        travel from the goal back to s0.
        """
        return int(1 * 10 // err_u) * (len(self.seen_state_action_pairs) + 1)

    def calculate_N(self, k):
        return 10 * self.calculate_m() * (len(self.seen_state_action_pairs) + 1)
    
    def calculate_H(self, k):
        """
        DEVIATION 5: horizon linear in the states discovered, not quadratic in |SA|.

        Reaching H is the signal that the greedy walk is trapped, so H must exceed
        any legitimate goal-reaching walk while still being reachable by a walk that
        really is stuck. 100 k^2 (|SA|^2 + 100) is a horizon no episode ever reaches
        on a model of any size, so the trap signal never fires and no end component
        is ever deflated.
        """
        return 10 * k * (len(self.seen_states) + 100)
    
    def calculate_Z(self, k):
        """Step budget for reading an end component off the greedy walk (see DEVIATION 5)."""
        return 100 * k * (len(self.seen_states) + 100)

    def calculate_ec_patience(self, ec_size):
        """
        DEVIATION 7: patience sized by the component's cover time.

        The closure walk is confined to the component, so covering it is a coupon
        collector problem needing about n log n visits. Every pair the walk has not
        tried is one deflate_ec will mistake for an exit sitting at U = 1, and a
        single such pair is enough to make the deflation lower nothing. A flat 100
        per pair is far too small on a large component and wasteful on a small one.
        """
        n = max(1, ec_size)
        return int(math.ceil(self.k * (10.0 * n * math.log(n + 1.0) + 1000.0)))
    
    def calculate_eps_u(self, k):
        """
        DEVIATION 6: the smallest committed improvement tracks this stage's accuracy.

        Held at 0.001/2^k it becomes smaller than any accuracy the stage can
        certify, so writes that are pure noise keep committing, the stage never
        looks saturated, and it runs its entire budget whether or not anything is
        still being learned. Tying it to `error` also keeps the DQL per-pair update
        bound 1/eps_u finite at every stage.
        """
        return self.error / 4.0

    def calculate_sa_confidence_error(self, k, confidence_error):
        return confidence_error / (len(self.seen_state_action_pairs) + 1)

    def calculate_c(self, m=None):
        """
        DEVIATION 1: the radius goes with the batch actually collected.

        This is a soundness fix, not a speed-up. The original returns the constant
        0.0001 regardless of how many samples the estimate rests on, which claims a
        precision no batch has earned: an unlucky batch pushes U below the true
        value, and U and L cross. That was observed -- L > U at s0 on the six-state
        model at 50M samples, though not at 5M or 500M, which is exactly the
        signature of a radius that is right by luck rather than by construction.

        Two-sided Hoeffding for a mean in [0,1] over m samples, at half of the
        pair's confidence budget so that taking the minimum with the Bernstein
        radius below is itself sound.
        """
        m = self.m if m is None else m
        delta_half = max(self.confidence_error, 1e-300) / 2.0
        n = max(1, len(self.seen_state_action_pairs))
        return math.sqrt(math.log(2.0 * n / delta_half) / (2.0 * max(1, m)))

    def calculate_c_bernstein(self, total, total_sq, m):
        """
        DEVIATION 2: an empirical-Bernstein radius, computed model-free.

        This is the one change that makes a large difference to how fast bounds
        close, and it is worth being precise about why it is model-free.

        The rejected optimisation kept a histogram succ[(s,a)][s'] and bounded the
        empirical transition *distribution*. That is a transition model. What this
        does instead is bound the mean of the m numbers the walk actually observed
        -- the bound values U(s'_i) at the successors it happened to land in. It
        needs only their sum and their sum of squares. No successor identity is
        stored, nothing is kept once the batch closes, and nothing here can be read
        back as a probability.

        The win is the same one the histogram version was after. A pair whose
        sampled values barely vary -- which includes every deterministic pair, and
        on these benchmarks 77% to 96% of pairs are deterministic -- has a radius
        of order 1/m rather than 1/sqrt(m), so reaching accuracy eps costs 1/eps
        samples instead of 1/eps^2.

        Maurer-Pontil, one-sided at delta/2 per side, applied two-sided. Nothing is
        assumed from low observed variance: the 3 log(.)/m term is exactly what
        covers the possibility that the variance is real but has not shown up yet.
        """
        m = max(1, m)
        if m < 2:
            return float("inf")
        delta_half = max(self.confidence_error, 1e-300) / 2.0
        n = max(1, len(self.seen_state_action_pairs))
        log_b = math.log(6.0 * n / delta_half)

        mean = total / m
        # Unbiased sample variance, floored at 0 against round-off.
        var = max(0.0, (total_sq - m * mean * mean) / (m - 1))
        return math.sqrt(2.0 * var * log_b / m) + 3.0 * log_b / m

    def calculate_radius(self, total, total_sq, m):
        """
        The radius used for a write: the tighter of DEVIATION 1 and DEVIATION 2,
        each spending half the pair's confidence so the minimum is sound, and
        capped at 1 because no bound in [0,1] needs a wider one.
        """
        return min(1.0,
                   self.calculate_c(m),
                   self.calculate_c_bernstein(total, total_sq, m))
    
    def sample_best_action(self, s):
        """
        Sample the best action from the current state s based on the upper bound of the value function.

        Args:
            s: The current state to sample the best action from.
        
        Returns:
            a: The sampled best action from the current state s.
        """
        return random.choice(self.best_actions[s])

    def sample_action(self, s, mu):
        """
        With probability mu, sample a random action from the current state s.
        With probability 1-mu, sample the best action from the current state s.

        Args:
            s: The current state to sample an action from.
            mu: The probability of sampling a random action.
        Returns:
            a: The sampled action from the current state s.
        """
        if (random.random() < mu):
            a = self.mdp_simulator.gt_mdp.sample_random_action_from_state(s)
        else:
            a = self.sample_best_action(s)
        return a
    
    def get_best_actions(self, s, u_table=None):
        """
        Get the U-maximal actions at s, read off the given upper-bound table.

        Args:
            s: The state to get the best actions for.
            u_table: The table to read. Defaults to self.u_new, the round-scoped
                     table that accumulates this round's writes.

        Returns:
            list: The U-maximal actions at s.
        """
        row = (self.u_new if u_table is None else u_table)[s]
        best = max(row.values())
        return [a_hat for a_hat in row if row[a_hat] == best]

    def refresh_best_actions(self, s, u_table=None):
        """Recompute and store the cached greedy action set at s."""
        self.best_actions[s] = self.get_best_actions(s, u_table)
    
    def detect_ec(self, s, patience_factor=None, min_patience=None):
        """
        Given that the inputted state is part of an EC, find
        all states and state-action pairs in the EC.

        Args:
            s: The state that is part of an EC.
            patience_factor: Steps-without-a-new-pair, per pair already found, before
                             the component is treated as covered.
            min_patience: Floor on that window, for the early steps where the
                          discovered set is still tiny.

        Returns:
            ec: The set of state-action pairs that make up the EC.
        """
        ec = set()
        Z = self.calculate_Z(self.k)
        steps_since_new = 0
        sim_step = self.mdp_simulator.step
        goal_states = self.mdp_simulator.gt_mdp.goal_states

        for _ in range(0, Z):
            # DEVIATION 8a: a walk that meets a goal returns nothing.
            # The original walks straight through goal states, so a component that
            # merely passes near a goal comes back with the goal bundled into it,
            # deflate_ec declines any component containing a goal, and a genuine trap
            # is never collapsed.
            #
            # Returning the empty set rather than the prefix collected so far is what
            # the pseudocode's DETECT_EC does, and it matters: a walk that reached a
            # goal was not circulating in an end component, so the pairs it visited on
            # the way are not one either. Deflating that prefix to its "best exit"
            # could push U below the true value. Nothing is collapsed, and nothing is
            # memoised -- this walk simply produced no evidence of a trap.
            if s in goal_states:
                return set()
            a = self.sample_action(s, mu=0)
            s_new, _ = sim_step(s, a)
            n_before = len(ec)
            ec.add((s, a))
            if len(ec) > n_before:
                steps_since_new = 0
            else:
                steps_since_new += 1
                # The walk has been confined to pairs it has already seen for far longer
                # than the component's cover time; treat it as covered and stop. Erring
                # short is safe: a smaller ec means a larger exit set, hence a bestExit
                # that is >= the true one, hence a less aggressive (still valid) deflation.
                # DEVIATION 8b: patience from calculate_ec_patience, sized by the
                # component rather than a flat 100 per pair.
                if steps_since_new >= self.calculate_ec_patience(len(ec)):
                    break
            s = s_new

        self.ecs.add(frozenset(ec))
        for (s, a) in ec:
            self.corresponding_ecs[s] = ec
        
        return ec
    
    def deflate_ec(self, ec, u):
        """
        Perform a pseudo-collapse on the EC. Deflate all the upper bounds of the
        state-action pairs in the EC to the upper bound fo the best exit action

        Args:
            ec: The set of state-action pairs that make up the EC.
            u: The current upper bound of the value function.
        
        Returns:
            u_new: The updated upper bound of the value function after collapsing the EC.
        """
        u_new = u
        ec_states = set()
        for (s, a) in ec:
            ec_states.add(s)
            # DEVIATION 9a: goal-ness from what the walk recorded, not from the model.
            # The original calls gt_mdp.is_goal_state(s) for every state in the
            # component. Those are states the learner is *remembering*, not the one it
            # is standing in, so this is a query about the model rather than an
            # observation -- the only genuine model-freeness violation in the original.
            # goal_seen is filled at visit time, so it answers the same question from
            # information the walk already had.
            if s in self.goal_seen:
                return u_new  # If the EC contains a goal state, we don't collapse it
        
        # Alg. COLLAPSE_EC: bestExit is 0 when the EC has no exits. U >= 0 is an
        # invariant here (init 1; writes are u_pred + c >= 0; deflations write
        # bestExit >= 0), so seeding at 0 also leaves the with-exits case unchanged.
        best_exit_action_value = 0
        for s in ec_states:
            # DEVIATION 9b: the action set comes from the bound row.
            # Same problem as 9a: gt_mdp.get_actions(s) asks the model about a
            # remembered state. The bound table already holds one entry per action of
            # every state the walk has stood in -- handle_new_state seeded it from the
            # actions available where the walk actually was -- so its keys are the same
            # set, obtained by observation.
            actions = u_new.get(s, self.u.get(s, {}))
            for a in actions:
                if (s, a) not in ec:
                    exit_value = u[s][a]
                    if exit_value > best_exit_action_value:
                        best_exit_action_value = exit_value
        
        for (s, a) in ec:
            # DEVIATION 9c: deflation may only lower U.
            # COLLAPSE_EC overwrites U with bestExit unconditionally. If the component
            # was read short -- a pair inside it that the closure walk never tried looks
            # like an exit sitting at U = 1 -- bestExit can exceed a U that real evidence
            # had already pushed down, and the deflation raises it back. U is an upper
            # bound being driven downward; nothing in the algorithm may push it up.
            if best_exit_action_value < u_new[s][a]:
                u_new[s][a] = best_exit_action_value

        # The deflation changed U at every EC state, so the cached greedy sets are stale.
        # Refreshing them is what makes the collapse steer the walk out of the component.
        for s in ec_states:
            self.refresh_best_actions(s, u_new)

        return u_new

    
    def simulate_and_update(self, u, l, u_new, l_new, u_agg, l_agg, u_agg_sq, l_agg_sq, curr_counts, eps_u, m, mu):
        """
        Simulate the MDP and update the upper and lower bounds of the value function.

        Args:
            u (dict): Current upper bounds of the value function.
            l (dict): Current lower bounds of the value function.
            u_new (dict): New upper bounds of the value function.
            l_new (dict): New lower bounds of the value function.
            u_agg (dict): Aggregated upper bounds of the value function.
            l_agg (dict): Aggregated lower bounds of the value function.
            curr_counts (dict): Current counts of state-action pairs.
            eps_u (float): Error tolerance for the upper bound.
            m (int): Number of samples to collect.
            mu (float): Learning rate or exploration parameter.

        Returns:
            u_new (dict): Updated upper bounds of the value function.
            l_new (dict): Updated lower bounds of the value function.
        """
        # Keep the round-scoped tables reachable from the refresh helpers.
        self.u_new, self.l_new = u_new, l_new

        def handle_new_state_action_pair(s_new, a_hat, is_goal_state):
            u[s_new][a_hat] = u[s_new].get(a_hat, 1)
            l[s_new][a_hat] = l[s_new].get(a_hat, 0 if not is_goal_state else 1)
            u_new[s_new][a_hat] = u_new[s_new].get(a_hat, 1)
            l_new[s_new][a_hat] = l_new[s_new].get(a_hat, 0 if not is_goal_state else 1)
            u_agg[s_new][a_hat] = u_agg[s_new].get(a_hat, 0)
            l_agg[s_new][a_hat] = l_agg[s_new].get(a_hat, 0)
            # DEVIATION 2 (cont.): sums of squares, for the empirical-Bernstein radius.
            u_agg_sq[s_new][a_hat] = u_agg_sq[s_new].get(a_hat, 0.0)
            l_agg_sq[s_new][a_hat] = l_agg_sq[s_new].get(a_hat, 0.0)
            curr_counts[s_new][a_hat] = 0
            self.seen_state_action_pairs.add((s_new, a_hat))


        def handle_new_state(s_new, is_goal_state):
            u[s_new] = dict()
            l[s_new] = dict()
            u_new[s_new] = dict()
            l_new[s_new] = dict()
            u_agg[s_new] = dict()
            l_agg[s_new] = dict()
            u_agg_sq[s_new] = dict()
            l_agg_sq[s_new] = dict()
            curr_counts[s_new] = dict()
            self.seen_states.add(s_new)
            # DEVIATION 9a (cont.): record goal-ness at visit time, so deflate_ec never
            # needs to ask the model about a state it is only remembering.
            if is_goal_state:
                self.goal_seen.add(s_new)

            actions = self.mdp_simulator.gt_mdp.get_actions(s_new)
            for a_hat in actions:
                handle_new_state_action_pair(s_new, a_hat, is_goal_state)
            
            self.refresh_best_actions(s_new)

        curr_state = self.s0
        curr_action = None
        t = 0
        H = self.calculate_H(self.k)

        # Hoist the hot attribute chains: each `self.mdp_simulator.gt_mdp.X` costs three
        # LOAD_ATTRs per loop iteration, and this loop runs up to H times.
        sim_step = self.mdp_simulator.step
        goal_states = self.mdp_simulator.gt_mdp.goal_states
        seen_states = self.seen_states
        sample_action = self.sample_action

        while t < H and (curr_state not in goal_states):
            if (curr_state not in seen_states):
                handle_new_state(curr_state, curr_state in goal_states)

            # DEVIATION 13: stop an episode that is standing on a dead state.
            # U(s) == 0 certifies that no policy reaches a goal from s, so no sample
            # drawn from here can move any bound. Without this the walk spends the
            # rest of its horizon inside a region it has already finished with --- on
            # a model with an absorbing non-goal state, which the EC machinery will
            # have just deflated to 0, that is very nearly the whole stage budget.
            # Reads one entry of the learner's own upper-bound table. No model, no
            # assumption: if U(s) is 0 the algorithm has already certified it.
            if u[curr_state][self.best_actions[curr_state][0]] <= 0.0:
                break

            curr_action = sample_action(curr_state, mu)
            next_state, reward = sim_step(curr_state, curr_action)
            if (next_state not in seen_states):
                handle_new_state(next_state, next_state in goal_states)

            self.sample_count += 1
            # handle_new_state seeds curr_counts[s][a] for every action of every state it
            # discovers, and run() reseeds them each round, so both keys always exist here
            # --- as the `u_agg[...] += ...` two lines below has always relied on.
            curr_counts[curr_state][curr_action] += 1
            
            u_sample = self.u[next_state][self.best_actions[next_state][0]]
            l_sample = self.l[next_state][self.best_actions[next_state][0]]
            u_agg[curr_state][curr_action] += u_sample
            l_agg[curr_state][curr_action] += l_sample
            # DEVIATION 2 (cont.): the squared sums are everything the Bernstein radius
            # needs. Two floats per pair, discarded when the batch closes -- no successor
            # identity is retained, so nothing here is a transition estimate.
            u_agg_sq[curr_state][curr_action] += u_sample * u_sample
            l_agg_sq[curr_state][curr_action] += l_sample * l_sample

            # print(f"t: {t}, reward: {reward}, sample_count: {self.sample_count}, u_agg: {u_agg[curr_state][curr_action]}, l_agg: {l_agg[curr_state][curr_action]}, curr_counts: {curr_counts[curr_state][curr_action]}")

            if curr_counts[curr_state][curr_action] >= m:
                curr_counts[curr_state][curr_action] = 0

                # DEVIATIONS 1 and 2: each bound gets the radius earned by its own
                # batch. U and L average different functions of the same successors, so
                # their observed variances differ and they deserve different radii.
                c_u = self.calculate_radius(u_agg[curr_state][curr_action],
                                            u_agg_sq[curr_state][curr_action], m)
                c_l = self.calculate_radius(l_agg[curr_state][curr_action],
                                            l_agg_sq[curr_state][curr_action], m)

                curr_u_pred_sa = u_agg[curr_state][curr_action] / m
                if (curr_u_pred_sa + c_u <= self.u[curr_state][curr_action] - eps_u):
                    # DEVIATION 10: refuse a write that would put U below L.
                    # U >= L is the invariant that makes the pair of bounds mean
                    # anything. Each write is individually sound, but the two are
                    # certified by different batches, and an unlucky pair of batches can
                    # still cross them. Skipping the write costs a batch; letting it
                    # through publishes an interval that does not contain the value.
                    u_candidate = curr_u_pred_sa + c_u
                    if u_candidate >= l_new[curr_state][curr_action]:
                        u_new[curr_state][curr_action] = u_candidate
                        self.refresh_best_actions(curr_state)
                        self.num_updates += 1
                curr_u_pred_sa = 0
                u_agg[curr_state][curr_action] = 0
                u_agg_sq[curr_state][curr_action] = 0.0

                curr_l_pred_sa = l_agg[curr_state][curr_action] / m
                if (curr_l_pred_sa - c_l >= self.l[curr_state][curr_action] + eps_u):
                    l_candidate = curr_l_pred_sa - c_l
                    if l_candidate <= u_new[curr_state][curr_action]:
                        l_new[curr_state][curr_action] = l_candidate
                        self.num_updates += 1
                curr_l_pred_sa = 0
                l_agg[curr_state][curr_action] = 0
                l_agg_sq[curr_state][curr_action] = 0.0
                
            
            curr_state = next_state
            t += 1

        # Detection is restricted to the purely greedy simulations, so that the walk that
        # triggers detection and the greedy walk detect_ec reads the component off with
        # are drawn from the same kernel.
        if (t >= H and mu == 0):
            if (curr_state in self.corresponding_ecs.keys()):
                ec = self.corresponding_ecs[curr_state]
            else:
                ec = self.detect_ec(curr_state)
            u_new = self.deflate_ec(ec, u_new)
        
        return u_new, l_new
    
    def bounds_at_s0(self):
        """Greedy bounds at the initial state: (U, L)."""
        u_row = self.u.get(self.s0)
        l_row = self.l.get(self.s0)
        if not u_row:
            return 1.0, 0.0
        return max(u_row.values()), (max(l_row.values()) if l_row else 0.0)

    def summarize_learning(self):
        return (
            self.best_actions.copy(),  # learned policy for the current stage
            self.u[self.s0][self.best_actions[self.s0][0]],
            self.l[self.s0][self.best_actions[self.s0][0]],
            self.u[self.s0][self.best_actions[self.s0][0]] - self.l[self.s0][self.best_actions[self.s0][0]],
            self.sample_count
        )

    
    def run(self):
        """
        Run a single stage of the PAC learning algorithm using TDQL.

        DEVIATION 12: bound writes land directly in self.u and self.l, and the
        per-pair evidence persists across rounds.

        The original commits into a shadow table that is swapped in only at the
        round barrier, and rebuilds the aggregates each round. Two costs. A bound
        moves at most one state per *round* instead of one per commit, so a lower
        bound needs as many rounds as there are states between s0 and the goal --
        which is why models with a deep goal never converge. And the rebuild
        discards the partial batch of every pair except the one that triggered the
        commit, which on a model of any size is nearly all of the evidence drawn.

        This is Gauss-Seidel where the original is Jacobi. It stores nothing new --
        the tables are the same bounds and counters -- and assumes nothing about the
        MDP. It does mean a batch can average a bound that moved part-way through
        it; see the caveat in the module docstring.

        Returns:
            policy: The learned policy for the current stage.
            upper_bound_s0: Upper bound of the value function for the initial state.
            lower_bound_s0: Lower bound of the value function for the initial state.
            error: The error of the learned policy.
            total_sample_counts: Total number of samples collected during this stage.
        """
        commits_at_entry = self.num_updates
        samples_at_entry = self.sample_count
        commits_seen = self.num_updates
        samples_at_last_commit = self.sample_count
        states_seen = len(self.seen_states)
        episodes_at_last_progress = 0

        j = 0
        progress = tqdm(desc="TDQL", unit="sample", leave=False)
        try:
            while True:
                # Batch size and stage budget are re-read as the walk discovers more
                # of the model, so neither is fixed by what was known before the
                # first state was seen.
                self.m = self.calculate_m()
                budget = 2 * self.calculate_N(self.k)
                max_commits = self.calculate_num_b_hat(self.eps_u)
                drawn = self.sample_count - samples_at_entry
                progress.total = budget
                progress.update(drawn - progress.n)

                if drawn >= budget:
                    break
                if self.num_updates - commits_at_entry >= max_commits:
                    break

                # DEVIATION 11: saturation exit.
                # The stage is done when it has extracted everything its accuracy
                # allows, and the rest of the budget would buy nothing. Progress means
                # either a bound moved or a new state was found. Both conditions are
                # required before quitting:
                #   - a full sweep's worth of samples, enough for every discovered
                #     pair to have filled its batch, and
                #   - enough episodes for the walk to have restarted from s0 many
                #     times, in both its exploring and its purely greedy form.
                # The episode floor is what stops one long episode from ending the
                # stage: a walk that falls into an absorbing non-goal state burns its
                # whole horizon there, which alone outweighs a sweep over the two or
                # three pairs discovered so far, and the stage would quit before it
                # had seen the model at all.
                if self.num_updates > commits_seen or len(self.seen_states) > states_seen:
                    commits_seen = self.num_updates
                    states_seen = len(self.seen_states)
                    samples_at_last_commit = self.sample_count
                    episodes_at_last_progress = j
                elif (self.sample_count - samples_at_last_commit
                          >= self.m * (len(self.seen_state_action_pairs) + 1)
                      and j - episodes_at_last_progress >= max(64, len(self.seen_states))):
                    break

                mu_j = self.mu if j % 2 == 0 else 0.0
                before = self.sample_count
                self.simulate_and_update(
                    u=self.u,
                    l=self.l,
                    u_new=self.u,
                    l_new=self.l,
                    u_agg=self.u_agg,
                    l_agg=self.l_agg,
                    u_agg_sq=self.u_agg_sq,
                    l_agg_sq=self.l_agg_sq,
                    curr_counts=self.curr_counts,
                    eps_u=self.eps_u,
                    m=self.m,
                    mu=mu_j
                )
                j += 1

                # DEVIATION 11 (cont.): the walk left s0 without drawing a sample,
                # which happens once U(s0) has been deflated to 0 -- the goal is
                # unreachable and every bound is final. The original has no guard for
                # this and spins forever against a budget it can no longer consume.
                if self.sample_count == before:
                    break

                # This stage was asked for an accuracy of self.error; once the bounds
                # at s0 are that close there is nothing left for it to do, and the
                # next stage's tighter error takes over.
                upper_s0, lower_s0 = self.bounds_at_s0()
                if upper_s0 - lower_s0 <= self.error:
                    break
        finally:
            progress.close()

        return self.summarize_learning()


    

class ModelFreeTDQLearner:
    """
    Main learner class that orchestrates PAC learning of LTL reachability using
    the model-free paradigm.
    """
    
    def __init__(
        self, 
        mdp_simulator: MDPSimulator, 
        min_num_iterations=5, 
        max_num_iterations=50, 
        convergence_threshold=0.001, 
        num_policy_accuracy_sims=1000,
        true_confidence_error=0.01,
        true_p_min=0.01
    ):
        """
        Initialize the learner.
        
        Args:
            mdp_simulator (MDPSimulator): The MDP simulator to interact with the true MDP.
            min_num_iterations (int): Minimum number of learning iterations.
            max_num_iterations (int): Maximum number of learning iterations.
            convergence_threshold (float): Threshold for convergence based on error.
            num_policy_accuracy_sims (int): Number of simulations to estimate policy accuracy.
            true_confidence_error (float): FOR ANALYSIS PURPOSES ONLY
            true_p_min (float): FOR ANALYSIS PURPOSES ONLY

        """
        self.mdp_sim = mdp_simulator  # Nests the true MDP
        self.learning_history = []  # [(k, num_samples, error), ...]
        self.states_set_history = []  # [(k, num_samples, seen_states_set), ...]
        self.transitions_seen_history = []  # [(k, num_samples, transitions_seen), ...]
        self.policy_accuracy_history = []  # [(k, num_samples, policy_accuracy), ..x.]
        self.true_error_history = []  # [(k, num_samples, true_error), ...]  FOR ANALYSIS PURPOSES ONLY - using the true p_min and true confidence error

        self.min_num_iterations = min_num_iterations
        self.max_num_iterations = max_num_iterations
        self.convergence_threshold = convergence_threshold
        self.num_policy_accuracy_sims = num_policy_accuracy_sims
        self.true_confidence_error = true_confidence_error  # FOR ANALYSIS PURPOSES ONLY - not used by the algorithm
        self.true_p_min = true_p_min  # FOR ANALYSIS PURPOSES ONLY - not used by the algorithm
        self.final_stage = None  # DEVIATION 14: the last stage run, holding the learned tables.

    
    def calculate_policy_accuracy(self, tdql_pac_stage, max_steps=100):
        """
        Calculate the policy accuracy of the discovered MDP.

        Returns:
            float: The calculated policy accuracy.
        """
        n = self.num_policy_accuracy_sims
        successful_runs = 0
        for _ in tqdm(range(0, n), desc="Policy accuracy sims", unit="sim"):
            # start from the ground-truth initial state
            curr_state = self.mdp_sim.gt_mdp.initial_state

            for _ in range(0, max_steps):
                if curr_state in self.mdp_sim.gt_mdp.goal_states:
                    successful_runs += 1
                    break

                if (curr_state not in tdql_pac_stage.seen_states):
                    break

                action = tdql_pac_stage.sample_best_action(curr_state)
                if action is None:
                    break

                next_state, _ = self.mdp_sim.step(curr_state, action)
                curr_state = next_state

        return successful_runs / n
    



    def learn(self, analysis_dir=""):
        """
        Main learning loop for PAC learning of LTL reachability.
        Iteratively simulates, updates, and runs BVI on the partial
        discovered MDP until convergence. Confidence error and minimum
        transition probability are halved each iteration.

        Args:
            analysis_dir (str): Directory to save analysis plots and data.
        """

        self.learning_history = []
        prev_collapsed_mdp_MEC_states = dict()  # {super_state: set of states in MEC}
        error = 1
        confidence_error = 1 ### TODO: will update this later in the final algorithm.
        p_min = 1  # NOTE: Optimization? Update p_min by either dividing by 2 or updating it to the lowest seen transition probability?
        k = 0
        prev_ecs = set()

        mu = 0.01

        while True:
            
            # Accept control c interrupt and safely exit while saving progress
            try:
                # Update Hyperparameters
                k += 1
                print("Iteration:", k)
                error_k = error / (2.0 ** k)  # DEVIATION 3/6: now used, to size m and eps_u
                confidence_error_k = confidence_error / (2.0 ** k)
                p_k = p_min / (2.0 ** k)

                # Each stage builds its own tables from scratch, as the pseudocode
                # specifies: the stages are independent.
                tdql_pac_stage_k = TDQL_PAC_Stage(k=k, error=error_k, confidence_error=confidence_error_k, p_min=p_k, mu=mu, mdp_simulator=self.mdp_sim)
                policy_k, upper_bound_s0, lower_bound_s0, curr_error, total_sample_counts = tdql_pac_stage_k.run()
                initial_state = self.mdp_sim.gt_mdp.initial_state
                self.final_stage = tdql_pac_stage_k  # DEVIATION 14

                self.learning_history.append([k, total_sample_counts, confidence_error_k, p_k, curr_error, lower_bound_s0, upper_bound_s0])

                if (analysis_dir != ""):
                    print("Num Sample Counts:", total_sample_counts)
                    print("Explored states:", len(tdql_pac_stage_k.seen_states), "/", len(self.mdp_sim.gt_mdp.states))

                    self.states_set_history.append((k, total_sample_counts, len(tdql_pac_stage_k.seen_states)))
                    self.transitions_seen_history.append((k, total_sample_counts, -1))
                    self.true_error_history.append((k, total_sample_counts, 0))

                    # Calculate policy accuracy
                    self.policy_accuracy_history.append(
                        (k, 
                         total_sample_counts, 
                         confidence_error_k, 
                         p_k, 
                         self.calculate_policy_accuracy(
                             tdql_pac_stage=tdql_pac_stage_k, 
                             max_steps=int(len(tdql_pac_stage_k.seen_states)**2 / p_k)
                            )
                        )
                    )
                    print("Policy accuracy:", self.policy_accuracy_history[-1][-1])

                    # Run iteration analysis, save plots and data
                    run_analysis(
                        analysis_dir=analysis_dir,
                        learning_history=self.learning_history,
                        states_set_history=self.states_set_history,
                        transitions_seen_history=self.transitions_seen_history,
                        policy_accuracy_history=self.policy_accuracy_history,
                        true_error_history=self.true_error_history,
                        max_states=len(self.mdp_sim.gt_mdp.states),
                        max_transitions=sum(len(sat_counts) for sat_counts in self.mdp_sim.gt_mdp.transition_probabilities.values()),
                        true_confidence_error=self.true_confidence_error,
                        true_p_min=self.true_p_min
                    )

                print("Error:", self.learning_history[-1])
                
                if (k > 1 and self.has_converged(tdql_pac_stage_k, self.learning_history[-2], curr_error, prev_ecs)):  # NOTE: -2 because we want the one before the current iteration (current iteration is -1 index).
                    print("Algorithm has converged, exiting...")
                    break

                prev_ecs = tdql_pac_stage_k.ecs  # Update the previous ECs for the next iteration
            except KeyboardInterrupt:
                print("Learning interrupted by user. Exiting and saving progress...")
                break

    
    # DEVIATION 14: report the bounds, not a discovered MDP.
    # The inherited print_summary reads self.discovered_mdp, deep-copies it,
    # collapses its MECs and runs a final BVI over it. A model-free learner has no
    # such object -- the attribute is never assigned, so the original raises
    # AttributeError here on every run that reaches the end. What this learner
    # actually produces is the last stage's tables, so that is what gets reported.
    def print_summary(self, true_confidence_error, true_p_min, k=10, output_path="", analysis_dir=""):
        """
        Print a summary of the learned bounds and policy, or save it as JSON if
        output_path is provided.

        The model-free learner never builds a transition model --- that is the
        point of it --- so there is no discovered MDP to collapse and no final BVI
        to run over one. What it produces is the last stage's tables: the greedy
        policy, the per-pair bounds, and the histories accumulated by learn().

        Args:
            true_confidence_error (float): FOR ANALYSIS PURPOSES ONLY - recorded, not used.
            true_p_min (float): FOR ANALYSIS PURPOSES ONLY - recorded, not used.
            k (int): unused, kept so callers written against the model-based learner still work.
            output_path (str): if set, write the summary as JSON here instead of printing it.
            analysis_dir (str): unused, kept for call-site compatibility.
        """
        print("Generating learned summary...")
        stage = self.final_stage
        if stage is None:
            print("No bounds have been learned yet.")
            return

        def sa_key(state, action):
            return f"{state}||{action}"

        upper_s0, lower_s0 = stage.bounds_at_s0()
        goal_states_seen = stage.seen_states & self.mdp_sim.gt_mdp.goal_states

        summary = {
            "states_count": len(stage.seen_states),
            "state_action_pairs_count": len(stage.seen_state_action_pairs),
            "initial_state": str(stage.s0),
            "goal_states": [str(s) for s in goal_states_seen],
            "end_components_found": len(stage.ecs),
            "total_samples": stage.sample_count,
            "bound_updates_committed": stage.num_updates,
            "true_confidence_error": true_confidence_error,
            "true_p_min": true_p_min,
            "sa_value_bounds": {},
            "s_value_bounds": {},
            "error": float(upper_s0 - lower_s0),
            "lower_bound_s0": float(lower_s0),
            "upper_bound_s0": float(upper_s0),
            "learned_policy": {},
            "learning_history": self.learning_history,
            "states_seen_history": self.states_set_history,
            "transitions_seen_history": self.transitions_seen_history,
            "policy_accuracy_history": self.policy_accuracy_history,
        }

        print("Compiling summary data...")
        for state in stage.seen_states:
            u_row, l_row = stage.u.get(state, {}), stage.l.get(state, {})
            if not u_row:
                continue
            for action in u_row:
                summary["sa_value_bounds"][sa_key(state, action)] = [
                    float(l_row.get(action, 0.0)), float(u_row[action])
                ]
            # The state value is the max over actions of each table, matching how
            # bounds_at_s0 reports the initial state.
            summary["s_value_bounds"][str(state)] = [
                float(max(l_row.values())), float(max(u_row.values()))
            ]
            summary["learned_policy"][str(state)] = [str(a) for a in stage.best_actions.get(state, [])]
        print("Summary data compilation complete.")

        if output_path:
            dirpath = os.path.dirname(output_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            return

        print("\n" + "="*60)
        print("LEARNED BOUNDS SUMMARY")
        print("="*60)
        print(f"States seen: {summary['states_count']}")
        print(f"State-Action Pairs seen: {summary['state_action_pairs_count']}")
        print(f"Initial State: {summary['initial_state']}")
        print(f"Goal States seen: {summary['goal_states']}")
        print(f"End components found: {summary['end_components_found']}")
        print(f"Samples drawn: {summary['total_samples']}")
        print(f"Bound updates committed: {summary['bound_updates_committed']}")

        print("\nState Value Bounds:")
        for state, bounds in summary["s_value_bounds"].items():
            print(f"  {state}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")

        print("\nState Action Value Bounds:")
        for sa, bounds in summary["sa_value_bounds"].items():
            print(f"  {sa}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")

        print(f"\nBounds at s0: [{lower_s0:.6f}, {upper_s0:.6f}]  (error {summary['error']:.6f})")

        print("\nLearned Policy:")
        for state, best_actions in summary["learned_policy"].items():
            print(f"  State: {state} --> Action(s): {best_actions}")

        print("="*60 + "\n")
        print("Analysis")
        print("="*60 + "\n")
        print("Analysis History (k, total_samples, error, num_seen_states, num_seen_transitions, policy_accuracy, lower_bound, upper_bound):")
        for i in range(0, len(self.learning_history)):
            record = self.learning_history[i]
            seen = self.states_set_history[i][-1] if i < len(self.states_set_history) else "-"
            trans = self.transitions_seen_history[i][-1] if i < len(self.transitions_seen_history) else "-"
            acc = self.policy_accuracy_history[i][-1] if i < len(self.policy_accuracy_history) else "-"
            print(f"  {record[0]}, {record[1]}, {record[-3]}, {seen}, {trans}, {acc}, {record[-2]}, {record[-1]}")


    def has_converged(self, curr_tdql_pac_stage, prev_iter_history, curr_error, prev_ecs=None):
        """
        Check if the learning process has converged based on the change in MDP error.
        
        Args:
            curr_tdql_pac_stage (TDQLPACStage): The current TDQL PAC stage.
            prev_iter_history (tuple): A tuple containing the previous iteration's parameters:
                (k, total_num_samples, delta, p_min, error).
            prev_ecs (set): The set of previously detected ECs. If provided, the function checks if the current ECs have changed.
        """
        # Make sure MEC states have not changed
        if (prev_ecs is not None):
            if (prev_ecs != curr_tdql_pac_stage.ecs):
                return False

        min_iterations = self.min_num_iterations
        max_iterations = self.max_num_iterations
        threshold = self.convergence_threshold
        prev_k, prev_total_num_samples, prev_delta, prev_p_min, prev_error, prev_l, prev_u =  prev_iter_history
        if (max_iterations and prev_k >= max_iterations):
            return True  # Reached maximum number of iterations
        if (min_iterations and prev_k < min_iterations):
            return False  # Need at least 'min_iterations' iterations to check for convergence

        # NOTE: the curr_error may be greater than the prev_error if a new MEC was found.
        # assert curr_error <= prev_error, f"Current error should be less than previous error if BVI is working correctly, got curr_error: {curr_error}, prev_error: {prev_error}"
        print(prev_error, curr_error, (prev_error - curr_error), threshold)
        return (prev_error - curr_error) < threshold