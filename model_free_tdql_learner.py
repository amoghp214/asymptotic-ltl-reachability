import copy
import math
import random
from typing import Set, Tuple, List, Optional, Any
from collections import Counter, deque
import numpy as np
import matplotlib.pyplot as plt

from mdp import MDP
from mdp_simulator import MDPSimulator
from analysis_utils import run_analysis
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
        prev_stage: "TDQL_PAC_Stage" = None,
    ):
        self.k = k
        self.error = error
        self.confidence_error = confidence_error
        self.p_min = p_min
        self.mu = mu
        self.mdp_simulator = mdp_simulator
        self.s0 = self.mdp_simulator.gt_mdp.initial_state
        self.sample_count = 0

        self.seen_states = set() # Set of all discovered states in the current PAC stage
        self.seen_state_action_pairs = set() # Set of all discovered state-action pairs in the current PAC stage
        self.ecs = set() # Set of all detected ECs in the current PAC stage
        self.corresponding_ecs = dict() # {s: ec, ...} mapping of states to their corresponding ECs
        self.ec_rechecked = set() # States whose remembered component has already been re-read once.
        self.u = dict() # {s: {a: upper_bound, ...}, ...}
        self.u[self.s0] = dict()
        self.l = dict() # {s: {a: lower_bound, ...}, ...}
        self.l[self.s0] = dict()

        # Round-scoped tables. Rebound at the top of every round in run(); kept in
        # sync inside simulate_and_update so refresh_best_actions() can read them.
        self.u_new = self.u
        self.l_new = self.l

        self.best_actions = dict() # {s: [best_actions], ...}

        # Sampling state that run() owns across the whole stage. Held here rather
        # than rebuilt per round so that evidence accumulated at one pair survives
        # a commit at some other pair.
        self.u_agg = dict() # {s: {a: running sum of U(s'), ...}, ...}
        self.l_agg = dict() # {s: {a: running sum of L(s'), ...}, ...}
        self.curr_counts = dict() # {s: {a: samples since this pair last updated, ...}, ...}
        self.batch = dict() # {(s, a): samples this pair must collect before its next update}
        self.num_updates = 0 # Bound writes committed, over the life of the tables.

        # Cached trajectory window, kept as sufficient statistics rather than as raw
        # transitions: succ[(s, a)][s'] is how many times (s, a) was seen to land in
        # s'. The sample count for a pair is the sum of that row --- the mean
        # branching factor on these models is 1.05 to 1.23, so summing it is
        # cheaper than maintaining a second dict keyed by the same tuple. This is the
        # same information a buffer of trajectories carries, at O(branching factor)
        # per pair instead of O(samples), and it is what lets a pair's bound be
        # re-estimated against the CURRENT successor bounds at any moment, with no
        # new sampling. preds[s'] is the reverse index the backward sweep walks.
        self.succ = dict()
        self.preds = dict()

        # How long the forward path is allowed to stall before the backward sweep is
        # run: one sweep's worth of samples divided by this. Larger means the sweep
        # fires sooner. It interpolates between never sweeping and sweeping after
        # every commit, and the best setting is model-dependent --- a dense model
        # where forward updates keep landing wants it low, a model whose bound is
        # stuck one layer short of somewhere useful wants it high.
        self.sweep_stall_divisor = 4

        # Bound the evidence behind every estimate to one round's worth, the way the
        # proof's pseudocode does: it resets U_agg, L_agg and #(s,a) once per round,
        # so each pair contributes exactly its first m observations and the window is
        # m by construction. Off by default, which leaves the cache cumulative and
        # every other behaviour untouched.
        self.reset_evidence_each_round = False

        # Goal states the walk has actually stood in. deflate_ec needs to know whether
        # a remembered component contains a goal, and asking the simulator about a
        # state the walk is not currently at would be a query the interface does not
        # allow. Recorded at visit time instead.
        self.goal_seen = set()

        # (1) Sweep backwards at the end of every round, before the evidence is
        # cleared, rather than only when the forward path stalls. A round's samples
        # otherwise move a bound one state back from the goal; carrying them over the
        # observed predecessors first lets one round move it many states, which is the
        # whole reason a bounded window might keep up with a cumulative one.
        self.sweep_each_round = False
        # (3) End an episode once every action at the state it is standing in has
        # already filled its window. Those samples cannot tighten anything until the
        # next reset.
        self.skip_saturated = False

        # calculate_log_term is called on every attempted update and depends only
        # on how many pairs have been discovered, which changes rarely.
        self._log_term = None
        self._log_term_at = -1

        # Carry the tables forward from the previous stage. U only ever decreases
        # and L only ever increases, so both are still valid bounds under the
        # tighter (error_k, confidence_error_k) of this stage; starting from them
        # is what makes the k-ladder cumulative rather than a set of independent
        # restarts. Without this every stage re-derives what the last one learned.
        if prev_stage is not None:
            self.seen_states = prev_stage.seen_states
            self.seen_state_action_pairs = prev_stage.seen_state_action_pairs
            self.ecs = prev_stage.ecs
            self.corresponding_ecs = prev_stage.corresponding_ecs
            self.ec_rechecked = prev_stage.ec_rechecked
            self.u = prev_stage.u
            self.l = prev_stage.l
            self.u_new, self.l_new = self.u, self.l
            self.best_actions = prev_stage.best_actions
            self.u_agg = prev_stage.u_agg
            self.l_agg = prev_stage.l_agg
            self.curr_counts = prev_stage.curr_counts
            self.batch = prev_stage.batch
            self.num_updates = prev_stage.num_updates
            self.sample_count = prev_stage.sample_count
            self.succ = prev_stage.succ
            self.preds = prev_stage.preds
            self.reset_evidence_each_round = prev_stage.reset_evidence_each_round
            self.goal_seen = prev_stage.goal_seen
            self.sweep_each_round = prev_stage.sweep_each_round
            self.skip_saturated = prev_stage.skip_saturated
            self.u.setdefault(self.s0, dict())
            self.l.setdefault(self.s0, dict())

        self.eps_u = self.calculate_eps_u(self.k)
        self.m = self.calculate_m()

    def calculate_log_term(self):
        """
        log(2 / delta_sa) for the per-pair confidence delta_sa, i.e. this stage's
        confidence_error split by a union bound over the state-action pairs
        discovered so far.
        """
        n = len(self.seen_state_action_pairs)
        if n != self._log_term_at:
            self._log_term = math.log(2.0 * max(1, n) / self.confidence_error)
            self._log_term_at = n
        return self._log_term

    def calculate_m(self):
        """
        Samples a state-action pair must collect before an update is attempted.

        Hoeffding: to estimate a mean in [0, 1] to within an accuracy of
        `error / 2` with per-pair confidence delta_sa, m = log(2/delta_sa) /
        (2 * accuracy^2) suffices. That is logarithmic in |SA| and in 1/delta and
        quadratic only in 1/error, whereas a batch size linear in |SA| and
        quadratic in 1/delta grows fast enough that no pair ever fills a batch on
        a model of any size.
        """
        accuracy = max(self.error / 2.0, 1e-6)
        hoeffding = math.ceil(self.calculate_log_term() / (2.0 * accuracy ** 2))
        # A pair whose successors are concentrated reaches this accuracy on the
        # Bernstein radius in O(1/accuracy) rather than O(1/accuracy^2) samples,
        # so gating every pair on the Hoeffding batch makes the concentrated
        # majority wait for an accuracy it already has. Size the batch on the
        # cheaper of the two; attempt_update still refuses to write until the
        # radius it actually computes justifies the write.
        bernstein = math.ceil(3.0 * math.log(3.0 / self.confidence_error) / accuracy)
        return int(max(1, min(hoeffding, bernstein)))

    def calculate_initial_batch(self):
        """
        First rung of the per-pair batch ladder: the batch that buys an accuracy
        of 1/2, i.e. the loosest bound that is not vacuous. Cheap, and enough to
        move a bound off its initial value; later rungs pay for the precision.
        """
        return max(1, int(math.ceil(2.0 * self.calculate_log_term())))

    def calculate_num_b_hat(self, err_u):
        """
        Bound on attempted updates. In DQL this is a PER-PAIR bound: each bound
        lives in [0, 1] and every commit moves it by at least eps_u, so a single
        pair can commit at most 1 / eps_u times. Scaling by the number of pairs is
        what makes it a bound on the stage as a whole --- applied globally it
        throttles the stage to a handful of writes and ends it long before the
        lower bound has finished spreading.
        """
        return int(1 * 10 // err_u) * (len(self.seen_state_action_pairs) + 1)

    def calculate_N(self, k):
        """
        Half the sample budget for one stage, so run()'s 2 * N is one batch for
        every pair discovered so far, twice over.

        This is only a ceiling. What normally ends a stage is the saturation
        exit in run(): a full sweep in which no bound moved. Deep models need many
        sweeps for a lower bound to travel from the goal back to s0 and should get
        them; shallow ones saturate in two or three and should not pay for twenty.
        Re-read as the walk discovers more of the model, so a stage is not sized by
        what was known before it started.
        """
        return 10 * self.calculate_m() * (len(self.seen_state_action_pairs) + 1)
    
    def calculate_H(self, k):
        """
        Episode horizon.

        Reaching H is the signal that the greedy walk is trapped, so H must be
        comfortably longer than any legitimate goal-reaching walk --- truncating
        one would deflate U along a path that is not an end component --- while
        still being reached in bounded time by a walk that really is trapped. A
        multiple of the discovered state count satisfies both and, unlike a
        multiple of |SA|^2, does not grow to a horizon no episode can ever reach.
        """
        return 10 * k * (len(self.seen_states) + 100)
    
    def calculate_Z(self, k):
        """Step budget for reading an end component off the greedy walk."""
        return 100 * k * (len(self.seen_states) + 100)
    
    def calculate_ec_patience(self, ec_size):
        """
        Steps the closure walk tolerates without finding a new pair before it calls
        the component covered.

        The walk is confined to the component, so this is a coupon-collector
        problem: to have tried all n of its pairs it needs about n log n visits,
        and every pair it has not tried is one deflate_ec will mistake for an exit
        sitting at U=1 --- which is what stops a deflation from lowering anything.
        Scaling with k as well means the budget grows with the stage, so a
        miscovered component is transient rather than permanent.

        Sized by the component rather than by a per-pair sample requirement:
        demanding log(1/delta)/p_min draws of every pair is also sound but costs
        enough at large k to dominate the run.
        """
        n = max(1, ec_size)
        return int(math.ceil(self.k * (10.0 * n * math.log(n + 1.0) + 1000.0)))

    def calculate_eps_u(self, k):
        """
        Smallest bound improvement worth committing.

        Tied to this stage's accuracy rather than held at a constant: a write
        smaller than the accuracy the stage can certify is noise at this stage,
        and committing it keeps the stage looking productive forever, so it never
        saturates and runs its whole budget regardless of whether anything is
        still being learned. Scaling with error also keeps the DQL bound on the
        number of updates per pair, 1 / eps_u, finite at every stage.
        """
        return self.error / 4.0

    def calculate_sa_confidence_error(self, k, confidence_error):
        return confidence_error / (len(self.seen_state_action_pairs) + 1)

    def calculate_c(self, m=None):
        """
        Hoeffding radius for a batch of `m` samples. Must be the radius that goes
        with the batch actually collected, or U and L stop being bounds: a radius
        held at a small constant lets an unlucky batch push U below the true
        value, and the two bounds cross.
        """
        return math.sqrt(self.calculate_log_term() / (2.0 * (self.m if m is None else m)))

    def calculate_c_cached(self, n, support, row=None):
        """
        Radius for an estimate taken from the cached successor distribution of a
        pair with `n` samples over `support` distinct successors.

        Hoeffding bounds the mean of one fixed function of the successor. That is
        not enough here: the cache gets re-read every time a successor's bound
        moves, so the function being averaged changes, and a fresh union bound
        would be needed for every re-read. Bounding the empirical *distribution*
        instead gives a radius that holds for every bounded function at once, so
        the cache can be re-read as often as the sweep likes at no extra
        statistical cost.

        Two such bounds are computed and the tighter is returned, each spending
        half of the pair's confidence budget so that taking the minimum is itself
        sound:

        Weissman's L1 bound, halved to total variation. Pays `support * log 2`
        whatever the distribution looks like, so it is the better of the two only
        while n is small.

        An empirical-Bernstein bound per successor, summed. This is the one that
        matters here. It pays for the variance actually observed rather than the
        worst case, and a pair whose samples all landed on the same successor has
        none: its radius collapses from order 1/sqrt(n) to order 1/n, which turns
        the samples needed for an accuracy of eps from 1/eps^2 into 1/eps. On
        these benchmarks that is not a corner case --- between 77% and 96% of
        state-action pairs have a single successor. Nothing is assumed about the
        transition structure to get this: zero *observed* variance does not mean
        zero true variance, and the 3L/n term is exactly what covers a successor
        that exists but has not been sampled yet.
        """
        n = max(1, n)
        support = max(1, support)
        delta_sa = self.confidence_error / (len(self.seen_state_action_pairs) + 1)

        # Weissman, at delta_sa / 2.
        log_w = math.log(2.0 / delta_sa)
        c_weissman = math.sqrt((support * math.log(2.0) + log_w) / (2.0 * n))

        # Empirical Bernstein per cell, at delta_sa / 2 split over the support
        # plus one slot for the mass on successors never yet seen.
        log_b = math.log(3.0 * (support + 1) * 2.0 / delta_sa)
        tail = 3.0 * log_b / n
        if row:
            l1 = 0.0
            for cnt in row.values():
                p_hat = cnt / n
                l1 += math.sqrt(2.0 * p_hat * (1.0 - p_hat) * log_b / n) + tail
            l1 += tail  # unobserved mass
            c_bernstein = 0.5 * l1
        else:
            c_bernstein = float("inf")

        return min(1.0, c_weissman, c_bernstein)
    
    def sample_best_action(self, s):
        """
        Sample the best action from the current state s based on the upper bound of the value function.

        Args:
            s: The current state to sample the best action from.
        
        Returns:
            a: The sampled best action from the current state s.
        """
        acts = self.best_actions[s]
        # random.choice goes through _randbelow even for a one-element list, and
        # once U separates the actions most states have exactly one greedy action.
        if len(acts) == 1:
            return acts[0]
        return random.choice(acts)

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
    
    def attempt_update(self, s, a, u_new, l_new, eps_u):
        """
        Re-estimate the bounds at (s, a) from its cached successor distribution,
        read against the CURRENT bounds of those successors, and commit if the
        estimate improves on what is held by at least eps_u.

        This draws no samples. It is what makes the sweep possible: a pair whose
        successors have just improved can be brought up to date immediately,
        rather than having to wait out a fresh batch of m samples during which
        those successors were still stale.

        Returns:
            bool: True if the state value of s (the max over its actions) moved,
                  which is the condition for the predecessors of s to be worth
                  revisiting.
        """
        row = self.succ.get((s, a))
        if not row:
            return False
        n = 0
        for cnt in row.values():
            n += cnt
        if n <= 0:
            return False
        c = self.calculate_c_cached(n, len(row), row)

        u_pred = 0.0
        l_pred = 0.0
        for s2, cnt in row.items():
            u_row = self.u.get(s2)
            if not u_row:
                continue
            w = cnt / n
            u_pred += w * max(u_row.values())
            l_pred += w * max(self.l[s2].values())

        u_before = max(u_new[s].values())
        l_before = max(l_new[s].values())

        # A write that would put U below L, or L above U, means the confidence
        # interval behind it failed. Skip it rather than writing a repaired value:
        # the bound that is already held is still valid, and clamping to the other
        # bound records a number no evidence supports.
        if u_pred + c <= self.u[s][a] - eps_u:
            u_candidate = u_pred + c
            if u_candidate >= l_new[s][a]:
                u_new[s][a] = u_candidate
                self.refresh_best_actions(s)
                self.num_updates += 1
        if l_pred - c >= self.l[s][a] + eps_u:
            l_candidate = l_pred - c
            if l_candidate <= u_new[s][a]:
                l_new[s][a] = l_candidate
                self.num_updates += 1

        return (max(u_new[s].values()) != u_before) or (max(l_new[s].values()) != l_before)

    def sweep_backwards(self, seeds, u_new, l_new, eps_u, work_budget):
        """
        Push a bound change backwards through the cached transitions.

        Without this a bound crawls one state per sweep of fresh samples, because a
        predecessor only notices its successor improved when it next fills a whole
        batch --- so a model whose goal sits d steps from s0 needs on the order of d
        sampling sweeps, and the work to certify it goes as d times the cost of a
        sweep. Re-estimating from the cache instead costs O(branching factor) per
        pair, so one change can be carried all the way back to s0 within a single
        episode and the d factor disappears.

        Args:
            seeds: states whose value has just changed.
            work_budget: cap on re-estimations, so a large component cannot make one
                         episode arbitrarily expensive.
        """
        queue = deque(seeds)
        queued = set(seeds)
        work = 0
        while queue and work < work_budget:
            s2 = queue.popleft()
            queued.discard(s2)
            for (s, a) in self.preds.get(s2, ()):
                if s not in self.u or a not in self.u[s]:
                    continue
                work += 1
                if work >= work_budget:
                    break
                if self.attempt_update(s, a, u_new, l_new, eps_u) and s not in queued:
                    queue.append(s)
                    queued.add(s)

    def detect_ec(self, s, patience_factor=None, min_patience=None):
        """
        Given that the inputted state is part of an EC, find
        all states and state-action pairs in the EC.

        Args:
            s: The state that is part of an EC.
            patience_factor: Steps-without-a-new-pair, per pair already found, before
                             the component is treated as covered. Defaults to the
                             budget this stage's p_min implies (see below).
            min_patience: Floor on that window, for the early steps where the
                          discovered set is still tiny. Same default.

        Returns:
            ec: The set of state-action pairs that make up the EC.
        """
        # The walk has to run long enough to have tried every action at every state
        # it reaches. An action it never tried is missing from the returned set, and
        # deflate_ec then counts it as an exit still sitting at U=1 --- which is what
        # stops the deflation from lowering anything. The window is therefore sized
        # by the component (calculate_ec_patience) rather than left at a flat 1000
        # steps. Z remains the hard cap on the walk.
        adaptive_patience = patience_factor is None and min_patience is None
        if patience_factor is None:
            patience_factor = 0
        if min_patience is None:
            min_patience = 0

        ec = set()
        successors = dict()  # (s, a) -> the states this pair was observed to reach
        Z = self.calculate_Z(self.k)
        steps_since_new = 0
        sim_step = self.mdp_simulator.step

        goal_states = self.mdp_simulator.gt_mdp.goal_states
        for _ in range(0, Z):
            # The closure walk follows the greedy policy, which is only defined at
            # states the sampling walk has already discovered. Reaching an
            # undiscovered one means the component leads out of what is known, so
            # stop: the pair that led here is recorded as leaving and the fixpoint
            # below drops it.
            if s not in self.best_actions:
                break
            # A goal absorbs the objective: the sampling walk stops there, so nothing
            # beyond it is part of any component that matters. Walking through one
            # instead sweeps the goal's own pairs into the component, and deflate_ec
            # then declines to collapse the whole thing because it contains a goal.
            # A sink two steps from a goal is enough to trigger that, and the sink is
            # then never deflated at all --- one run spent 99.9% of its samples inside
            # a sink that had been bundled with a goal state on the first detection.
            if s in goal_states:
                break
            a = self.sample_action(s, mu=0)
            s_new, _ = sim_step(s, a)
            n_before = len(ec)
            ec.add((s, a))
            if (s, a) in successors:
                successors[(s, a)].add(s_new)
            else:
                successors[(s, a)] = {s_new}
            if len(ec) > n_before:
                steps_since_new = 0
            else:
                steps_since_new += 1
                # The walk has been confined to pairs it has already seen for far longer
                # than the component's cover time; treat it as covered and stop. Erring
                # short is safe: a smaller ec means a larger exit set, hence a bestExit
                # that is >= the true one, hence a less aggressive (still valid) deflation.
                budget = (self.calculate_ec_patience(len(ec)) if adaptive_patience
                          else max(min_patience, patience_factor * len(ec)))
                if steps_since_new >= budget:
                    break
            s = s_new

        # The walk returns the pairs it traversed, which is a path, not a component:
        # it happily includes an action that was seen leaving. deflate_ec treats
        # everything in the set as internal and every other action as an exit, so
        # handing it a path lets it read a genuine exit as internal, find no exits at
        # all, and deflate U to zero along states that can still reach the goal ---
        # U stops being an upper bound and the greedy walk stops moving. Restrict to
        # the largest sub-set closed under the transitions actually observed: drop
        # every pair seen leaving, drop the states that leaves without actions, and
        # repeat to a fixpoint.
        # Judge "does this pair leave?" on every successor ever recorded for it, not
        # just the ones this closure walk happened to see. The walk lasts a few
        # thousand steps, so an exit taken with probability 1e-4 is missed most of
        # the time, and the component comes back one pair too large --- which is
        # exactly the case that gets deflated wrongly. The cache has the pair's whole
        # history, so a rare exit seen once at any point in the run still counts.
        for pair in ec:
            cached = self.succ.get(pair)
            if cached:
                successors[pair] = successors[pair] | set(cached)

        ec_states = set(x for (x, _a) in ec)
        while ec:
            leaving = set(pair for pair in ec if not successors[pair] <= ec_states)
            if not leaving:
                break
            ec -= leaving
            ec_states = set(x for (x, _a) in ec)

        if not ec:
            return ec

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
        self.last_deflation_lowered = 0
        u_new = u
        ec_states = set()
        for (s, a) in ec:
            ec_states.add(s)
            if s in self.goal_seen:
                return u_new  # If the EC contains a goal state, we don't collapse it
        
        # Alg. COLLAPSE_EC: bestExit is 0 when the EC has no exits. U >= 0 is an
        # invariant here (init 1; writes are u_pred + c >= 0; deflations write
        # bestExit >= 0), so seeding at 0 also leaves the with-exits case unchanged.
        best_exit_action_value = 0
        for s in ec_states:
            # The action set of a state the walk has visited is already recorded as the
            # keys of its bound row; re-querying the simulator about a state the walk is
            # not standing in would step outside the interface.
            actions = u_new.get(s, self.u.get(s, {}))
            for a in actions:
                if (s, a) not in ec:
                    # An exit whose U has never been written is still at its
                    # initial 1, and reading it as 1 is the conservative choice
                    # anyway: it can only raise bestExit, i.e. weaken the deflation.
                    exit_value = u.get(s, {}).get(a, 1.0)
                    if exit_value > best_exit_action_value:
                        best_exit_action_value = exit_value
        
        lowered = 0
        for (s, a) in ec:
            # Deflate, never inflate. Inside a genuine end component every state has
            # the same value, bestExit, so a valid U is already >= bestExit there and
            # min() writes exactly bestExit --- identical behaviour. The two differ
            # only when the component is not really one, and there raising U throws
            # away everything sampling had established about those pairs. Observed
            # directly: a single spurious deflation pushes U(s0) back to 1 and the
            # run never recovers.
            if best_exit_action_value < u_new[s][a]:
                u_new[s][a] = best_exit_action_value
                lowered += 1

        # The deflation changed U at every EC state, so the cached greedy sets are stale.
        # Refreshing them is what makes the collapse steer the walk out of the component.
        for s in ec_states:
            self.refresh_best_actions(s, u_new)

        self.last_deflation_lowered = lowered
        return u_new

    
    def simulate_and_update(self, u, l, u_new, l_new, u_agg, l_agg, curr_counts, eps_u, m, mu):
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
        initial_batch = min(self.calculate_initial_batch(), m)

        def handle_new_state_action_pair(s_new, a_hat, is_goal_state):
            u[s_new][a_hat] = u[s_new].get(a_hat, 1)
            l[s_new][a_hat] = l[s_new].get(a_hat, 0 if not is_goal_state else 1)
            u_new[s_new][a_hat] = u_new[s_new].get(a_hat, 1)
            l_new[s_new][a_hat] = l_new[s_new].get(a_hat, 0 if not is_goal_state else 1)
            u_agg[s_new][a_hat] = u_agg[s_new].get(a_hat, 0)
            l_agg[s_new][a_hat] = l_agg[s_new].get(a_hat, 0)
            curr_counts[s_new][a_hat] = 0
            self.batch.setdefault((s_new, a_hat), initial_batch)
            self.seen_state_action_pairs.add((s_new, a_hat))


        def handle_new_state(s_new, is_goal_state):
            u[s_new] = dict()
            l[s_new] = dict()
            u_new[s_new] = dict()
            l_new[s_new] = dict()
            u_agg[s_new] = dict()
            l_agg[s_new] = dict()
            curr_counts[s_new] = dict()
            self.seen_states.add(s_new)

            actions = self.mdp_simulator.gt_mdp.get_actions(s_new)
            if is_goal_state:
                self.goal_seen.add(s_new)
            for a_hat in actions:
                handle_new_state_action_pair(s_new, a_hat, is_goal_state)
            
            self.refresh_best_actions(s_new)

        curr_state = self.s0
        curr_action = None
        t = 0
        H = self.calculate_H(self.k)
        half_H = H // 2
        saturating = self.skip_saturated and self.reset_evidence_each_round
        distinct_states = set()

        # Hoist the hot attribute chains: each `self.mdp_simulator.gt_mdp.X` costs three
        # LOAD_ATTRs per loop iteration, and this loop runs up to H times.
        sim_step = self.mdp_simulator.step
        goal_states = self.mdp_simulator.gt_mdp.goal_states
        seen_states = self.seen_states
        sample_action = self.sample_action
        succ = self.succ
        preds = self.preds
        batch_of = self.batch
        u_tab = self.u

        trajectory = []
        while t < H and (curr_state not in goal_states):
            if (curr_state not in seen_states):
                handle_new_state(curr_state, curr_state in goal_states)

            # U(s) == 0 certifies that no policy reaches a goal from s, so no
            # sample drawn from here can move any bound. Without this the walk
            # spends the rest of its horizon inside a region it has already
            # finished with --- on a model with an absorbing non-goal state that
            # is very nearly the whole budget.
            if u_tab[curr_state][self.best_actions[curr_state][0]] <= 0.0:
                break
            # (3) Every action available at the state the walk is standing in has
            # filled its window this round, so nothing sampled from here can move a
            # bound before the reset. Reads only the counters and this state's own row.
            if saturating:
                row_counts = curr_counts[curr_state]
                if row_counts and all(v >= m for v in row_counts.values()):
                    break
            # Confinement has to be judged on the TAIL of the episode, not the whole
            # of it. A walk that explores two hundred states and is then absorbed in
            # a sink for the remaining three thousand steps is confined -- but counted
            # over the whole episode its distinct-state count is dominated by the
            # exploring prefix and the test below rejects it. Measured on pacman: 301
            # episodes ended inside a sink and not one of them triggered detection.
            # Clearing once at the half-way mark makes the count cover the second half
            # only, which is the part that says whether the walk is still going
            # anywhere. Recording nothing before 64 steps keeps the common short
            # episode free of set inserts.
            if t == half_H:
                distinct_states.clear()
            if t >= 64:
                distinct_states.add(curr_state)

            curr_action = sample_action(curr_state, mu)
            next_state, reward = sim_step(curr_state, curr_action)
            if (next_state not in seen_states):
                handle_new_state(next_state, next_state in goal_states)

            self.sample_count += 1
            trajectory.append((curr_state, curr_action, next_state))

            curr_state = next_state
            t += 1

        # Consume the episode from its last transition backwards.
        #
        # The samples, the batches and the estimator are exactly the same either
        # way; only the order in which the batches are consumed changes. Read
        # forwards, a pair's batch closes while its successor's bound is still at
        # last sweep's value, so a lower bound travels one state back from the goal
        # per sweep and a model whose goal is fifty steps from s0 needs fifty
        # sweeps. Read backwards, the successor has already taken this episode's
        # update by the time the predecessor's batch closes, so one episode can
        # carry the bound the whole way back.
        # The loop targets are deliberately NOT curr_state / curr_action /
        # next_state. Python binds a for-target in the enclosing function scope and
        # leaves it bound after the loop, so reusing those names here overwrote the
        # state the walk had actually ended in --- and, because the trajectory is
        # consumed in reverse, left it holding the episode's FIRST state. The
        # detection block below reads curr_state, so it was handed a state near s0
        # instead of the trap the walk was stuck in, and every component it read off
        # was the opening stretch of the trajectory rather than an end component.
        for (step_state, step_action, step_next) in reversed(trajectory):
            # handle_new_state seeds curr_counts[s][a] for every action of every state it
            # discovers, and run() reseeds them each round, so both keys always exist here
            # --- as the `u_agg[...] += ...` two lines below has always relied on.
            curr_counts[step_state][step_action] += 1

            # Record the transition into the cache instead of folding it into a
            # running sum. A sum fixes each term at the bound its successor held
            # when the sample was drawn, so a batch collected while its successors
            # were still at zero averages to zero however good they later become,
            # and the pair has to wait out a whole fresh batch to notice. The
            # counts keep the estimate re-computable against current bounds.
            pair_key = (step_state, step_action)
            row = succ.get(pair_key)
            if row is None:
                row = succ[pair_key] = dict()
            row[step_next] = row.get(step_next, 0) + 1
            back = preds.get(step_next)
            if back is None:
                back = preds[step_next] = set()
                back.add(pair_key)
            elif pair_key not in back:
                back.add(pair_key)

            # Each pair carries its own batch size, growing after every attempted
            # update. Early rungs are cheap and give a loose but immediately useful
            # bound; later rungs cost more and tighten it. The batch now gates only
            # how often a pair is re-estimated, not what data the estimate sees.
            pair_batch = batch_of.get(pair_key, m)
            if pair_batch > m:
                # A batch larger than this stage's m buys an accuracy finer than
                # this stage was asked for. Capping here is what bounds the work
                # per pair per stage; without it the ladder runs away and a pair
                # that has updated twenty times never updates again.
                pair_batch = m

            if curr_counts[step_state][step_action] >= pair_batch:
                # Quadrupling halves the accuracy the batch buys, matching the
                # step the k-ladder takes between stages.
                batch_of[pair_key] = min(4 * pair_batch, m)
                curr_counts[step_state][step_action] = 0
                # The ordinary forward update, and nothing more. Sweeping backwards
                # from every commit costs a great deal on models where the forward
                # path is already making progress, so the sweep is held back until
                # run() sees that progress has actually stopped.
                self.attempt_update(step_state, step_action, u_new, l_new, eps_u)

        # Detection is restricted to the purely greedy simulations, so that the walk that
        # triggers detection and the greedy walk detect_ec reads the component off with
        # are drawn from the same kernel.
        # A greedy walk that spends the second half of the horizon revisiting a
        # handful of states is trapped in an end component. One that keeps reaching
        # fresh states is a long legitimate walk, and deflating the states along it
        # would push U below a true value, so require the tail of the walk to have
        # been genuinely confined before treating it as a component.
        # Detection is restricted to the purely greedy simulations, so that the walk
        # that triggers detection and the greedy walk detect_ec reads the component
        # off with are drawn from the same kernel.
        if (t >= H and mu == 0 and 20 * len(distinct_states) <= t - half_H):
            # A component is read off the walk once and remembered against every
            # state in it, so falling back into one the walk has already mapped costs
            # a deflation and no sampling at all. That is the common case: a walk
            # absorbed in a trap re-enters it on almost every episode.
            #
            # The remembered component can still be wrong --- an early walk that
            # wandered in from outside can record the path it travelled rather than
            # the trap, and deflating that lowers nothing because the pairs it never
            # tried are counted as exits still at U=1. So when a remembered component
            # deflates to nothing, the entry is re-read once, from this state, and
            # trusted thereafter. Bounded by one extra walk per state over the run.
            ec = self.corresponding_ecs.get(curr_state)
            if ec is None:
                ec = self.detect_ec(curr_state)
                u_new = self.deflate_ec(ec, u_new)
            else:
                u_new = self.deflate_ec(ec, u_new)
                if (self.last_deflation_lowered == 0
                        and curr_state not in self.ec_rechecked):
                    self.ec_rechecked.add(curr_state)
                    ec = self.detect_ec(curr_state)
                    u_new = self.deflate_ec(ec, u_new)
        
        return u_new, l_new
    
    def bounds_at_s0(self):
        """
        (U(s0), L(s0)) for the state value, i.e. max over actions of each table.
        max_a U dominates max_a L pairwise, so reporting both as maxima keeps
        U(s0) >= L(s0) even when the two are maximised by different actions.
        """
        if not self.u.get(self.s0):
            return 1.0, 0.0
        return (self.u[self.s0][self.best_actions[self.s0][0]],
                max(self.l[self.s0].values()))

    def summarize_learning(self):
        upper_s0, lower_s0 = self.bounds_at_s0()
        return (
            self.best_actions.copy(),  # learned policy for the current stage
            upper_s0,
            lower_s0,
            upper_s0 - lower_s0,
            self.sample_count
        )

    
    def run(self):
        """
        Run a single stage of the PAC learning algorithm using TDQL.
        Guarantees the policy found is within the specified error bounds
        with the given confidence, assuming p_min.

        Bound writes land directly in self.u and self.l rather than into a shadow
        table swapped in once per round. Two reasons. A write is visible to the very
        next bootstrap that reads it, instead of only after the round barrier, so a
        bound moves at most one state per round rather than per commit. And the
        aggregates are no longer rebuilt per round, which used to discard the
        partial batch of every pair except the one that triggered the commit --- on
        a model of any size, nearly all of the evidence collected.

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
        samples_at_last_sweep = self.sample_count
        samples_at_last_round = self.sample_count

        j = 0
        progress = tqdm(desc="TDQL", unit="sample", leave=False)
        try:
            while True:
                # Both the batch size and the stage budget are re-read as the walk
                # discovers more of the model, so neither is fixed by what was known
                # before the first state was seen.
                self.m = self.calculate_m()
                budget = 2 * self.calculate_N(self.k)
                # Re-read with the batch size and the budget: evaluated once at entry
                # it is computed while no pair has been discovered yet, which pins the
                # first stage to 1 / eps_u writes in total no matter how large the
                # model turns out to be.
                max_commits = self.calculate_num_b_hat(self.eps_u)
                drawn = self.sample_count - samples_at_entry
                progress.total = budget
                progress.update(drawn - progress.n)

                if drawn >= budget:
                    break
                if self.num_updates - commits_at_entry >= max_commits:
                    break
                # Saturation exit: this stage has extracted everything its accuracy
                # allows and the rest of the budget would buy nothing. Progress means
                # either a bound moved or a new state was found; a stage is done only
                # when neither has happened for
                #   - a full sweep's worth of samples, enough for every discovered
                #     pair to fill its batch, and
                #   - enough episodes for the walk to have restarted from s0 many
                #     times over, in both its exploring and its purely greedy form.
                # The episode floor is what keeps a single long episode from ending
                # the stage: a walk that falls into an absorbing non-goal state burns
                # its whole horizon there, which on its own outweighs a sweep over
                # the two or three pairs discovered so far, and the stage would quit
                # before it had seen the model at all.
                if self.num_updates > commits_seen or len(self.seen_states) > states_seen:
                    commits_seen = self.num_updates
                    states_seen = len(self.seen_states)
                    samples_at_last_commit = self.sample_count
                    episodes_at_last_progress = j
                elif (self.sample_count - samples_at_last_commit >= self.m * (len(self.seen_state_action_pairs) + 1)
                      and j - episodes_at_last_progress >= max(64, len(self.seen_states))):
                    break
                else:
                    # Forward sampling has stopped moving anything. Before spending
                    # more of the budget on samples that are not landing, re-derive
                    # every pair from the cached trajectory window against the bounds
                    # as they stand now and let the change cascade backwards. This is
                    # where a bound that has been sitting one state away from a
                    # region it could improve finally gets to cross it, and it costs
                    # no samples at all. Held to a quarter of the saturation window so
                    # it fires well before the stage would otherwise give up.
                    stall = self.sample_count - samples_at_last_commit
                    window = max(1, self.m * (len(self.seen_state_action_pairs) + 1)
                                 // max(1, self.sweep_stall_divisor))
                    if (stall >= window
                            and self.sample_count - samples_at_last_sweep >= window):
                        samples_at_last_sweep = self.sample_count
                        self.sweep_backwards(
                            list(self.seen_states), self.u, self.l, self.eps_u,
                            20 * (len(self.seen_state_action_pairs) + 1))

                # A round is one sweep: enough samples for every discovered pair to
                # have filled its batch once. Clearing the per-pair evidence at that
                # boundary makes the window exactly what the current accuracy needs,
                # rather than letting it grow without bound over the whole stage.
                if self.reset_evidence_each_round:
                    round_len = max(1, self.m * (len(self.seen_state_action_pairs) + 1))
                    if self.sample_count - samples_at_last_round >= round_len:
                        samples_at_last_round = self.sample_count
                        if self.sweep_each_round:
                            # (1) Spend the round's evidence before discarding it.
                            self.sweep_backwards(
                                list(self.seen_states), self.u, self.l, self.eps_u,
                                20 * (len(self.seen_state_action_pairs) + 1))
                        self.succ.clear()
                        self.preds.clear()
                        self.batch.clear()
                        for row in self.curr_counts.values():
                            for a in row:
                                row[a] = 0

                mu_j = self.mu if j % 2 == 0 else 0.0
                before = self.sample_count
                self.simulate_and_update(
                    u=self.u,
                    l=self.l,
                    u_new=self.u,
                    l_new=self.l,
                    u_agg=self.u_agg,
                    l_agg=self.l_agg,
                    curr_counts=self.curr_counts,
                    eps_u=self.eps_u,
                    m=self.m,
                    mu=mu_j
                )
                j += 1
                if self.sample_count == before:
                    # The walk left s0 without drawing a sample, which happens once
                    # U(s0) has been deflated to 0: the goal is unreachable and every
                    # bound is final. Nothing further can change, so stop rather than
                    # spin against a budget the loop can no longer consume.
                    break

                # This stage was asked for an accuracy of self.error; once the
                # bounds at s0 are that close there is nothing left for it to do
                # and the next stage's tighter error takes over.
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
        self.final_stage = None  # The last TDQL_PAC_Stage run, holding the learned tables.

    
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
    



    def make_stage(self, **kwargs):
        """
        Build the stage used for one iteration of the ladder.

        The only reason this is a method is so that a variant can subclass the
        learner and return a stage configured differently, without copying the
        loop in learn(). The default is the stage defined in this module, with
        its default flags.
        """
        return TDQL_PAC_Stage(**kwargs)


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
        prev_stage = None

        mu = 0.01

        while True:
            
            # Accept control c interrupt and safely exit while saving progress
            try:
                # Update Hyperparameters
                k += 1
                print("Iteration:", k)
                error_k = error / (2.0 ** k)  # Accuracy this stage is asked for; sets its batch size.
                confidence_error_k = confidence_error / (2.0 ** k)
                p_k = p_min / (2.0 ** k)

                # Hand the previous stage in so this one starts from the bounds it
                # ended with. The confidence errors are summable over k, so the
                # union bound over the whole ladder still costs a constant.
                tdql_pac_stage_k = self.make_stage(k=k, error=error_k, confidence_error=confidence_error_k, p_min=p_k, mu=mu, mdp_simulator=self.mdp_sim, prev_stage=prev_stage)
                policy_k, upper_bound_s0, lower_bound_s0, curr_error, total_sample_counts = tdql_pac_stage_k.run()
                initial_state = self.mdp_sim.gt_mdp.initial_state
                self.final_stage = tdql_pac_stage_k

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

                prev_ecs = set(tdql_pac_stage_k.ecs)  # Snapshot: the stage's own set is now carried forward and mutated in place.
                prev_stage = tdql_pac_stage_k
            except KeyboardInterrupt:
                print("Learning interrupted by user. Exiting and saving progress...")
                break

    
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
        if (curr_error <= threshold):
            return True  # The bounds are as tight as the caller asked for.
        if (min_iterations and prev_k < min_iterations):
            return False  # Need at least 'min_iterations' iterations to check for convergence

        # NOTE: the curr_error may be greater than the prev_error if a new MEC was found.
        # assert curr_error <= prev_error, f"Current error should be less than previous error if BVI is working correctly, got curr_error: {curr_error}, prev_error: {prev_error}"
        print(prev_error, curr_error, (prev_error - curr_error), threshold)
        return (prev_error - curr_error) < threshold