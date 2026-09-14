"""
Unit tests for the model-free TDQL learner.

Covers the pseudocode-fidelity behaviour of TDQL_PAC_Stage (greedy-action cache,
EC detection gating, EC deflation) and the sampling path in MDPSimulator.
"""

import random
import unittest

from mdp import MDP
from mdp_simulator import MDPSimulator
from variants.model_free.cumulative import TDQL_PAC_Stage


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

def build_simulator(states, actions, transitions, goals, initial):
    """
    Args:
        states (list): states to add.
        actions (list): (state, action) pairs to enable.
        transitions (list): (state, action, next_state, probability) tuples.
        goals (set): goal states.
        initial: initial state.
    """
    # Passed explicitly so each fixture is plainly independent.
    sim = MDPSimulator(mdp=MDP())
    for s in states:
        sim.add_state(s)
    if goals:
        sim.set_goal_states(set(goals))
    for s, a in actions:
        sim.add_action_to_state(s, a)
    for tr in transitions:
        sim.add_transition(*tr)
    sim.set_initial_state(initial)
    return sim


def six_state_simulator():
    """The benchmark from main.py:14-58."""
    return build_simulator(
        states=[1, 2, 3, 4, 5, 6],
        actions=[(1, 'b'), (1, 'c'), (1, 'd'), (2, 'b'), (3, 'a'),
                 (4, 'a'), (4, 'd'), (5, 'a'), (6, 'a')],
        transitions=[(1, 'b', 6, 0.1), (1, 'b', 5, 0.9),
                     (1, 'c', 1, 0.4), (1, 'c', 2, 0.05), (1, 'c', 3, 0.1),
                     (1, 'c', 4, 0.05), (1, 'c', 5, 0.2), (1, 'c', 6, 0.2),
                     (1, 'd', 4, 1),
                     (2, 'b', 2, 0.5), (2, 'b', 3, 0.5),
                     (3, 'a', 3, 0.5), (3, 'a', 4, 0.25), (3, 'a', 5, 0.25),
                     (4, 'a', 3, 0.5), (4, 'a', 6, 0.1), (4, 'a', 1, 0.4),
                     (4, 'd', 1, 0.9999), (4, 'd', 5, 0.0001),
                     (5, 'a', 5, 1), (6, 'a', 6, 1)],
        goals={3, 6},
        initial=1,
    )


def goalless_simulator():
    """1 -a-> 2 -a-> 2, no goal state, so every walk runs the full horizon H."""
    return build_simulator(
        states=[1, 2],
        actions=[(1, 'a'), (2, 'a')],
        transitions=[(1, 'a', 2, 1.0), (2, 'a', 2, 1.0)],
        goals=set(),
        initial=1,
    )


def make_stage(sim, k=1, confidence_error=1.0, mu=0.01):
    return TDQL_PAC_Stage(k=k, error=1.0, confidence_error=confidence_error,
                          p_min=1.0, mu=mu, mdp_simulator=sim)


def seed_tables(stage, rows, goals=()):
    """
    Populate a stage's tables directly, mimicking handle_new_state.

    Args:
        rows (dict): {state: [actions]}.
        goals (iterable): states whose L initialises to 1.
    """
    for s, acts in rows.items():
        stage.u[s] = {a: 1.0 for a in acts}
        stage.l[s] = {a: (1.0 if s in goals else 0.0) for a in acts}
        stage.seen_states.add(s)
        for a in acts:
            stage.seen_state_action_pairs.add((s, a))
    stage.u_new, stage.l_new = stage.u, stage.l
    for s in rows:
        stage.refresh_best_actions(s)


def fresh_round_state(stage):
    """Build the per-round tables run() creates, and return them."""
    u_new = {s: dict(r) for s, r in stage.u.items()}
    l_new = {s: dict(r) for s, r in stage.l.items()}
    u_agg, l_agg, counts = {}, {}, {}
    for s in stage.u:
        u_agg[s], l_agg[s], counts[s] = {}, {}, {}
        for a in stage.u[s]:
            u_agg[s][a] = l_agg[s][a] = counts[s][a] = 0
    return u_new, l_new, u_agg, l_agg, counts


def run_round(stage, num_simulations=None):
    """Run one round of the stage, mirroring TDQL_PAC_Stage.run()'s inner loop."""
    u_new, l_new, u_agg, l_agg, counts = fresh_round_state(stage)
    limit = num_simulations or (2 * stage.calculate_N(stage.k))
    updated = False
    for j in range(limit):
        mu_j = stage.mu if j % 2 == 0 else 0.0
        u_new, l_new = stage.simulate_and_update(
            stage.u, stage.l, u_new, l_new, u_agg, l_agg, counts,
            stage.eps_u, stage.m, mu_j)
        if u_new != stage.u or l_new != stage.l:
            updated = True
            break
    stage.u, stage.l = u_new, l_new
    return updated


# ----------------------------------------------------------------------------
# The greedy-action cache is read off U_new
# ----------------------------------------------------------------------------

class TestBestActionsCache(unittest.TestCase):

    def test_defaults_to_u_new_not_the_entry_table(self):
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd']})
        stage.u_new = {1: {'b': 0.2, 'c': 0.9, 'd': 0.5}}

        self.assertEqual(stage.get_best_actions(1), ['c'])
        # The entry table is untouched and still has every action tied at 1.0.
        self.assertCountEqual(stage.get_best_actions(1, stage.u), ['b', 'c', 'd'])

    def test_all_actions_tied_when_table_is_uniform(self):
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd']})
        self.assertCountEqual(stage.best_actions[1], ['b', 'c', 'd'])

    def test_refresh_picks_up_a_write(self):
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd']})
        stage.u_new = {1: dict(stage.u[1])}

        stage.u_new[1]['b'] = 0.1
        stage.u_new[1]['d'] = 0.1
        stage.refresh_best_actions(1)
        self.assertEqual(stage.best_actions[1], ['c'])

    def test_refresh_rebinds_rather_than_mutating(self):
        """summarize_learning()'s shallow .copy() is only safe if this holds."""
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd']})
        snapshot = stage.best_actions.copy()
        original = snapshot[1]

        stage.u_new[1]['c'] = 0.1
        stage.refresh_best_actions(1)
        self.assertIs(snapshot[1], original)
        self.assertCountEqual(original, ['b', 'c', 'd'])

    def test_ties_are_broken_uniformly_at_random(self):
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd']})
        random.seed(7)
        drawn = {stage.sample_best_action(1) for _ in range(200)}
        self.assertEqual(drawn, {'b', 'c', 'd'})

    def test_invariant_holds_after_running_rounds(self):
        """best_actions[s] must equal argmax over the committed table at every state."""
        random.seed(3)
        stage = make_stage(six_state_simulator())
        for _ in range(8):
            run_round(stage, num_simulations=40)

        self.assertTrue(stage.seen_states)
        for s in stage.seen_states:
            best = max(stage.u[s].values())
            expected = [a for a, v in stage.u[s].items() if v == best]
            self.assertCountEqual(stage.best_actions[s], expected, msg="state %r" % (s,))


# ----------------------------------------------------------------------------
# COLLAPSE_EC / deflate_ec
# ----------------------------------------------------------------------------

class TestUpdateGuard(unittest.TestCase):
    """A bound write is skipped, not repaired, when it would cross the other bound."""

    def _stage_with_one_pair(self, u_val, l_val, successor_u, successor_l):
        """
        A stage holding one pair (1, 'b') whose only observed successor is state 6,
        with the bounds at 1 and at 6 set explicitly. attempt_update then re-derives
        (1, 'b') from that cached successor, so the candidate it computes is driven
        entirely by state 6's bounds.
        """
        stage = make_stage(six_state_simulator(), confidence_error=1.0)
        seed_tables(stage, {1: ['b'], 6: ['a']})
        stage.u[1]['b'], stage.l[1]['b'] = u_val, l_val
        stage.u[6]['a'], stage.l[6]['a'] = successor_u, successor_l
        stage.refresh_best_actions(6)
        stage.succ[(1, 'b')] = {6: 5000}
        return stage

    def test_upper_write_that_would_fall_below_l_is_skipped(self):
        # L(1,b) is high; the successor drags the U estimate well under it.
        stage = self._stage_with_one_pair(u_val=1.0, l_val=0.9,
                                          successor_u=0.0, successor_l=0.0)
        stage.attempt_update(1, 'b', stage.u, stage.l, eps_u=0.0)

        self.assertEqual(stage.u[1]['b'], 1.0, "the crossing U write must be skipped")
        self.assertEqual(stage.l[1]['b'], 0.9, "L is untouched by a skipped U write")
        self.assertLessEqual(stage.l[1]['b'], stage.u[1]['b'])

    def test_lower_write_that_would_rise_above_u_is_skipped(self):
        # U(1,b) is low; the successor drags the L estimate above it.
        stage = self._stage_with_one_pair(u_val=0.1, l_val=0.0,
                                          successor_u=1.0, successor_l=1.0)
        stage.attempt_update(1, 'b', stage.u, stage.l, eps_u=0.0)

        self.assertEqual(stage.l[1]['b'], 0.0, "the crossing L write must be skipped")
        self.assertLessEqual(stage.l[1]['b'], stage.u[1]['b'])

    def test_a_write_that_does_not_cross_still_lands(self):
        """The guard must not block ordinary progress."""
        stage = self._stage_with_one_pair(u_val=1.0, l_val=0.0,
                                          successor_u=1.0, successor_l=1.0)
        stage.attempt_update(1, 'b', stage.u, stage.l, eps_u=0.0)

        self.assertGreater(stage.l[1]['b'], 0.0, "L should have moved up")
        self.assertLessEqual(stage.l[1]['b'], stage.u[1]['b'])

    def test_sandwich_holds_over_a_long_run(self):
        random.seed(97)
        stage = make_stage(six_state_simulator())
        for _ in range(12):
            run_round(stage, num_simulations=60)

        for s in stage.u:
            for a in stage.u[s]:
                self.assertLessEqual(stage.l[s][a], stage.u[s][a],
                                     "sandwich violated at (%r, %r)" % (s, a))


class TestDeflateEC(unittest.TestCase):

    def test_no_exit_deflates_to_zero_not_negative_infinity(self):
        stage = make_stage(goalless_simulator())
        seed_tables(stage, {2: ['a']})
        ec = {(2, 'a')}

        u_new = {s: dict(r) for s, r in stage.u.items()}
        result = stage.deflate_ec(ec, u_new)

        self.assertEqual(result[2]['a'], 0)
        self.assertFalse(any(v == float('-inf')
                             for row in result.values() for v in row.values()))

    def test_with_exits_uses_the_best_exit_value(self):
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd']})
        stage.u[1] = {'b': 0.9, 'c': 0.7, 'd': 0.4}
        stage.u_new = stage.u

        u_new = {1: dict(stage.u[1])}
        # 'b' and 'd' are inside the EC; 'c' is the only exit.
        result = stage.deflate_ec({(1, 'b'), (1, 'd')}, u_new)

        self.assertEqual(result[1]['b'], 0.7, "an in-EC pair above the best exit comes down to it")
        self.assertEqual(result[1]['c'], 0.7, "the exit action itself is untouched")
        self.assertEqual(result[1]['d'], 0.4,
                         "deflation never raises U: a pair already tighter than the best "
                         "exit keeps its value")

    def test_deflation_never_raises_the_upper_bound(self):
        """
        U must be non-increasing. Inside a genuine EC every state has value bestExit,
        so a valid U is already >= bestExit and writing bestExit is a decrease either
        way; the two only differ when the component is not really one, and there a
        raise throws away everything sampling had established about those pairs.
        """
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd']})
        stage.u[1] = {'b': 0.10, 'c': 0.95, 'd': 0.20}
        stage.u_new = stage.u

        u_new = {1: dict(stage.u[1])}
        before = dict(u_new[1])
        stage.deflate_ec({(1, 'b'), (1, 'd')}, u_new)

        for action, value in before.items():
            self.assertLessEqual(u_new[1][action], value,
                                 "deflation raised U(1, %r)" % (action,))

    def test_goal_containing_ec_is_left_alone(self):
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {3: ['a']}, goals={3})
        u_new = {3: {'a': 1.0}}

        result = stage.deflate_ec({(3, 'a')}, u_new)
        self.assertEqual(result[3]['a'], 1.0)

    def test_refreshes_the_greedy_cache_of_every_ec_state(self):
        """Without this the collapse cannot steer the walk out of the component."""
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd']})

        # b and d are the in-EC pairs and are strictly preferred, so the greedy
        # walk can never take the exit c and is trapped in the component.
        u_new = {1: {'b': 1.0, 'c': 0.9, 'd': 1.0}}
        stage.u_new = u_new
        stage.refresh_best_actions(1)
        self.assertCountEqual(stage.best_actions[1], ['b', 'd'])

        stage.deflate_ec({(1, 'b'), (1, 'd')}, u_new)

        # Deflating to the best exit ties every in-EC pair with that exit, so the
        # exit becomes reachable under the per-visit tie-break. The cache has to
        # be refreshed for the walk to see it.
        self.assertEqual(u_new[1], {'b': 0.9, 'c': 0.9, 'd': 0.9})
        self.assertIn('c', stage.best_actions[1])
        self.assertCountEqual(stage.best_actions[1], ['b', 'c', 'd'])


# ----------------------------------------------------------------------------
# DETECT_EC
# ----------------------------------------------------------------------------

class TestDetectEC(unittest.TestCase):

    def test_finds_the_component_and_stops_well_before_z(self):
        random.seed(11)
        sim = goalless_simulator()
        stage = make_stage(sim)
        seed_tables(stage, {1: ['a'], 2: ['a']})

        calls = []
        inner = sim.step
        sim.step = lambda s, a: (calls.append(1), inner(s, a))[1]

        ec = stage.detect_ec(2)

        self.assertEqual(ec, {(2, 'a')})
        self.assertLess(len(calls), stage.calculate_Z(stage.k),
                        "the patience window should cut the walk short")

    def test_early_exit_finds_the_same_set_as_the_full_walk(self):
        random.seed(13)
        sim = goalless_simulator()

        stage_short = make_stage(sim)
        seed_tables(stage_short, {1: ['a'], 2: ['a']})
        ec_short = stage_short.detect_ec(2)

        stage_full = make_stage(sim)
        seed_tables(stage_full, {1: ['a'], 2: ['a']})
        # min_patience >= Z disables the early exit.
        ec_full = stage_full.detect_ec(2, min_patience=stage_full.calculate_Z(1))

        self.assertEqual(ec_short, ec_full)

    def test_records_the_component_against_its_states(self):
        random.seed(17)
        stage = make_stage(goalless_simulator())
        seed_tables(stage, {1: ['a'], 2: ['a']})

        ec = stage.detect_ec(2)
        self.assertIn(frozenset(ec), stage.ecs)
        self.assertEqual(stage.corresponding_ecs[2], ec)


# ----------------------------------------------------------------------------
# SIMULATE_AND_UPDATE: detection is gated on the purely greedy simulations
# ----------------------------------------------------------------------------

def chain_to_trap_simulator():
    """1 -a-> 2 -a-> 3 -a-> 3, no goal. Three distinct states so that the state the
    walk ENDS in (3) differs from the second state of the episode (2)."""
    return build_simulator(
        states=[1, 2, 3],
        actions=[(1, 'a'), (2, 'a'), (3, 'a')],
        transitions=[(1, 'a', 2, 1.0), (2, 'a', 3, 1.0), (3, 'a', 3, 1.0)],
        goals=set(),
        initial=1,
    )


class TestEpisodeAccounting(unittest.TestCase):
    """
    Regression tests for a duplicated loop advance.

    The update pass consumes the episode with `for (...) in reversed(trajectory)`.
    A `curr_state = next_state; t += 1` pair was left inside that loop as well as
    inside the walk. Two consequences, both after the walk had already finished:

    * `curr_state` was reassigned on the final (reversed) iteration to that
      transition's successor -- the episode's SECOND state -- so detection was
      rooted there instead of at the state the walk ended in. The tests below
      cover this; they need three distinct states to tell the two apart.
    * `t` was advanced a second time per transition, roughly doubling the value
      the detection gate reads. That does not shorten the episode (the walk has
      already exited by then) but it does slacken the confinement test
      `20 * distinct <= t - half_H`, making detection fire more readily than
      intended. No test below covers that; it is observable only through the gate,
      which the episode does not expose.
    """

    def _run_one_episode(self, stage):
        u_new, l_new, u_agg, l_agg, counts = fresh_round_state(stage)
        before = stage.sample_count
        stage.simulate_and_update(stage.u, stage.l, u_new, l_new,
                                  u_agg, l_agg, counts, stage.eps_u, stage.m, mu=0.0)
        return stage.sample_count - before, u_new

    def test_a_trapped_episode_draws_exactly_one_horizon_of_samples(self):
        """Sanity: the walk itself is unaffected -- one sample per step, up to H."""
        random.seed(3)
        stage = make_stage(goalless_simulator())
        H = stage.calculate_H(stage.k)
        drawn, _ = self._run_one_episode(stage)
        self.assertEqual(drawn, H)

    def test_detection_is_rooted_where_the_walk_stopped(self):
        """On 1 -> 2 -> 3(trap), detection must start at 3, not at 2."""
        random.seed(3)
        stage = make_stage(chain_to_trap_simulator())
        rooted_at = []
        real = stage.detect_ec
        stage.detect_ec = lambda s, **kw: (rooted_at.append(s), real(s, **kw))[1]
        self._run_one_episode(stage)

        self.assertTrue(rooted_at, "a stuck greedy walk must trigger detection")
        self.assertEqual(rooted_at[0], 3,
                         "detection must be rooted at the trap the walk ended in (3), "
                         "not at the episode's second state (2)")

    def test_the_component_is_the_trap_alone(self):
        """State 2 is transient and belongs to no end component."""
        random.seed(3)
        stage = make_stage(chain_to_trap_simulator())
        self._run_one_episode(stage)

        self.assertTrue(stage.ecs)
        for ec in stage.ecs:
            self.assertEqual({x for (x, _a) in ec}, {3},
                             "the component must be the absorbing state alone")

    def test_the_trap_is_deflated_to_zero(self):
        random.seed(3)
        stage = make_stage(chain_to_trap_simulator())
        _, u_new = self._run_one_episode(stage)
        self.assertEqual(u_new[3]['a'], 0.0,
                         "an absorbing non-goal state cannot reach the target")

    def test_the_transient_prefix_is_not_deflated(self):
        """Deflating state 2 would be wrong -- it is on the path, not in the trap."""
        random.seed(3)
        stage = make_stage(chain_to_trap_simulator())
        _, u_new = self._run_one_episode(stage)
        self.assertEqual(u_new[1]['a'], 1.0)
        self.assertEqual(u_new[2]['a'], 1.0)


class TestBoundedEvidenceWindow(unittest.TestCase):
    """
    `reset_evidence_each_round` bounds the evidence behind every estimate to one
    round's worth, as the proof's pseudocode does by resetting U_agg, L_agg and
    #(s,a) once per round. Off by default, so nothing else changes.
    """

    def _run(self, windowed, stages=4):
        """Drives the real run() loop, which is where the round boundary lives."""
        random.seed(23)
        sim = six_state_simulator()
        prev = None
        for k in range(1, stages + 1):
            stage = TDQL_PAC_Stage(k=k, error=1 / 2 ** k, confidence_error=1 / 2 ** k,
                                   p_min=1 / 2 ** k, mu=0.01, mdp_simulator=sim,
                                   prev_stage=prev)
            stage.reset_evidence_each_round = windowed
            stage.run()
            prev = stage
        return stage

    def test_off_by_default(self):
        stage = make_stage(six_state_simulator())
        self.assertFalse(stage.reset_evidence_each_round)

    def test_the_setting_is_carried_to_the_next_stage(self):
        first = make_stage(six_state_simulator())
        first.reset_evidence_each_round = True
        second = TDQL_PAC_Stage(k=2, error=1.0, confidence_error=1.0, p_min=1.0,
                                mu=0.01, mdp_simulator=first.mdp_simulator,
                                prev_stage=first)
        self.assertTrue(second.reset_evidence_each_round)

    def test_windowed_evidence_is_bounded_and_cumulative_is_not(self):
        """The whole point: the cached counts must stop growing."""
        cumulative = self._run(windowed=False)
        windowed = self._run(windowed=True)
        cum_max = max((sum(r.values()) for r in cumulative.succ.values()), default=0)
        win_max = max((sum(r.values()) for r in windowed.succ.values()), default=0)
        self.assertGreater(cum_max, 0)
        self.assertLess(win_max, cum_max,
                        "the windowed run should hold strictly less evidence per pair")

    def test_bounds_stay_sound_when_windowed(self):
        stage = self._run(windowed=True, stages=5)
        for st in stage.u:
            for a in stage.u[st]:
                self.assertTrue(0.0 <= stage.u[st][a] <= 1.0)
                self.assertLessEqual(stage.l[st][a], stage.u[st][a],
                                     "sandwich violated at (%r, %r)" % (st, a))

    def test_clearing_does_not_disturb_the_bounds_themselves(self):
        """A reset drops evidence, never a bound that has already been committed."""
        stage = make_stage(six_state_simulator())
        stage.reset_evidence_each_round = True
        run_round(stage, num_simulations=60)
        before = {s: dict(r) for s, r in stage.u.items()}
        stage.succ.clear(); stage.preds.clear(); stage.batch.clear()
        for row in stage.curr_counts.values():
            for a in row:
                row[a] = 0
        self.assertEqual({s: dict(r) for s, r in stage.u.items()}, before)


class TestECPatience(unittest.TestCase):
    """The closure walk's patience is sized by the component, not left at a flat 1000."""

    def test_patience_grows_with_component_size(self):
        stage = make_stage(goalless_simulator())
        small, large = stage.calculate_ec_patience(1), stage.calculate_ec_patience(64)
        self.assertGreater(large, small)
        self.assertGreaterEqual(small, 1000, "there is a floor below which nothing is covered")

    def test_patience_grows_with_the_stage(self):
        """A component miscovered at one stage must get a longer look at the next."""
        early = make_stage(goalless_simulator(), k=1).calculate_ec_patience(16)
        late = make_stage(goalless_simulator(), k=6).calculate_ec_patience(16)
        self.assertGreater(late, early)

    def test_default_patience_is_used_when_none_is_passed(self):
        """The walk must run at least the component-sized budget before giving up."""
        random.seed(5)
        sim = goalless_simulator()
        stage = make_stage(sim)
        seed_tables(stage, {1: ['a'], 2: ['a']})
        calls = []
        inner = sim.step
        sim.step = lambda s, a: (calls.append(1), inner(s, a))[1]

        stage.detect_ec(2)
        self.assertGreaterEqual(len(calls), stage.calculate_ec_patience(1),
                                "the adaptive budget should not stop before its floor")

    def test_an_explicit_patience_still_overrides(self):
        random.seed(5)
        sim = goalless_simulator()
        stage = make_stage(sim)
        seed_tables(stage, {1: ['a'], 2: ['a']})
        calls = []
        inner = sim.step
        sim.step = lambda s, a: (calls.append(1), inner(s, a))[1]

        stage.detect_ec(2, patience_factor=1, min_patience=5)
        self.assertLess(len(calls), stage.calculate_ec_patience(1),
                        "an explicitly passed window must still be honoured")


class TestECMemo(unittest.TestCase):
    """A component is read off the walk once and reused whenever it is re-entered."""

    def _trapped_stage(self):
        """1 -a-> 2 -a-> 2, no goal: state 2 is a trap the greedy walk cannot leave."""
        sim = goalless_simulator()
        stage = make_stage(sim)
        seed_tables(stage, {1: ['a'], 2: ['a']})
        return stage

    def test_component_is_remembered_against_every_state_in_it(self):
        stage = self._trapped_stage()
        ec = stage.detect_ec(2)
        for s in {x for (x, _a) in ec}:
            self.assertIn(s, stage.corresponding_ecs)
            self.assertEqual(stage.corresponding_ecs[s], ec)

    def test_re_entry_reuses_the_stored_component_without_sampling(self):
        """The whole point: re-entering a mapped trap must cost no simulator calls."""
        random.seed(11)
        stage = self._trapped_stage()
        stage.detect_ec(2)
        u = {s: dict(r) for s, r in stage.u.items()}
        stage.u_new = u

        calls = []
        inner = stage.mdp_simulator.step
        stage.mdp_simulator.step = lambda s, a: (calls.append(1), inner(s, a))[1]
        stage.deflate_ec(stage.corresponding_ecs[2], u)

        self.assertEqual(len(calls), 0, "deflating a remembered component must not sample")

    def test_a_stored_component_that_deflates_to_nothing_is_re_read_once(self):
        """A component recorded by a walk that wandered in can be wrong; one re-read fixes it."""
        random.seed(13)
        stage = self._trapped_stage()
        # Pin state 2 to a deliberately useless component: a pair whose deflation
        # cannot lower anything, standing in for a path recorded instead of a trap.
        stage.corresponding_ecs[2] = {(1, 'a')}

        detected = []
        real = stage.detect_ec
        stage.detect_ec = lambda s, **kw: (detected.append(s), real(s, **kw))[1]

        u, l, u_agg, l_agg, counts = fresh_round_state(stage)
        for _ in range(6):
            stage.simulate_and_update(stage.u, stage.l, u, l, u_agg, l_agg, counts,
                                      stage.eps_u, stage.m, mu=0.0)

        self.assertIn(2, stage.ec_rechecked, "the useless entry should have been re-read")
        self.assertLessEqual(detected.count(2), 1,
                             "and re-read at most once, however often it is re-entered")

    def test_deflation_reports_whether_it_lowered_anything(self):
        """last_deflation_lowered is what tells the caller a stored entry was useless."""
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd']})
        stage.u[1] = {'b': 1.0, 'c': 0.4, 'd': 1.0}
        stage.u_new = stage.u

        u = {1: dict(stage.u[1])}
        stage.deflate_ec({(1, 'b'), (1, 'd')}, u)
        self.assertEqual(stage.last_deflation_lowered, 2, "both in-EC pairs came down to 0.4")

        u2 = {1: dict(u[1])}
        stage.deflate_ec({(1, 'b'), (1, 'd')}, u2)
        self.assertEqual(stage.last_deflation_lowered, 0, "nothing left to lower")

    def test_bounds_stay_sound_with_the_memo_in_play(self):
        random.seed(17)
        stage = make_stage(six_state_simulator())
        for _ in range(10):
            run_round(stage, num_simulations=50)
        for s in stage.u:
            for a in stage.u[s]:
                self.assertTrue(0.0 <= stage.u[s][a] <= 1.0)
                self.assertLessEqual(stage.l[s][a], stage.u[s][a],
                                     "sandwich violated at (%r, %r)" % (s, a))


class TestDetectionGating(unittest.TestCase):

    def test_greedy_simulation_detects(self):
        random.seed(19)
        stage = make_stage(goalless_simulator())
        u_new, l_new, u_agg, l_agg, counts = fresh_round_state(stage)

        stage.simulate_and_update(stage.u, stage.l, u_new, l_new,
                                  u_agg, l_agg, counts, stage.eps_u, stage.m, mu=0.0)
        self.assertTrue(stage.ecs, "a stuck greedy walk must detect its component")

    def test_exploring_simulation_does_not_detect(self):
        random.seed(19)
        stage = make_stage(goalless_simulator())
        u_new, l_new, u_agg, l_agg, counts = fresh_round_state(stage)

        stage.simulate_and_update(stage.u, stage.l, u_new, l_new,
                                  u_agg, l_agg, counts, stage.eps_u, stage.m, mu=stage.mu)
        self.assertFalse(stage.ecs, "detection is restricted to mu == 0 simulations")

    def test_a_walk_that_reaches_the_goal_does_not_detect(self):
        random.seed(23)
        sim = build_simulator(states=[1, 2], actions=[(1, 'a'), (2, 'a')],
                              transitions=[(1, 'a', 2, 1.0), (2, 'a', 2, 1.0)],
                              goals={2}, initial=1)
        stage = make_stage(sim)
        u_new, l_new, u_agg, l_agg, counts = fresh_round_state(stage)

        stage.simulate_and_update(stage.u, stage.l, u_new, l_new,
                                  u_agg, l_agg, counts, stage.eps_u, stage.m, mu=0.0)
        self.assertFalse(stage.ecs)


# ----------------------------------------------------------------------------
# Round bookkeeping
# ----------------------------------------------------------------------------

class TestRoundTables(unittest.TestCase):

    def test_round_copy_isolates_every_row(self):
        stage = make_stage(six_state_simulator())
        seed_tables(stage, {1: ['b', 'c', 'd'], 2: ['b']})
        u_new, _, _, _, _ = fresh_round_state(stage)

        self.assertIsNot(u_new, stage.u)
        for s in stage.u:
            self.assertIsNot(u_new[s], stage.u[s])

        u_new[1]['b'] = 0.123
        self.assertEqual(stage.u[1]['b'], 1.0, "a write to u_new must not reach self.u")

    def test_simulation_discovers_states_and_counts_samples(self):
        random.seed(29)
        stage = make_stage(six_state_simulator())
        u_new, l_new, u_agg, l_agg, counts = fresh_round_state(stage)

        stage.simulate_and_update(stage.u, stage.l, u_new, l_new,
                                  u_agg, l_agg, counts, stage.eps_u, stage.m, mu=stage.mu)

        self.assertIn(1, stage.seen_states)
        self.assertGreater(stage.sample_count, 0)
        for s in stage.seen_states:
            self.assertEqual(set(stage.u[s]), set(stage.mdp_simulator.gt_mdp.get_actions(s)))
            self.assertEqual(set(counts[s]), set(stage.u[s]),
                             "curr_counts must stay key-aligned with u")

    def test_goal_states_initialise_l_to_one(self):
        random.seed(31)
        stage = make_stage(six_state_simulator())
        run_round(stage, num_simulations=20)

        for s in stage.seen_states & stage.mdp_simulator.gt_mdp.goal_states:
            for a in stage.l[s]:
                self.assertEqual(stage.l[s][a], 1.0)

    def test_bounds_stay_valid_across_rounds(self):
        random.seed(37)
        stage = make_stage(six_state_simulator())
        for _ in range(10):
            run_round(stage, num_simulations=40)

        for s in stage.u:
            for a in stage.u[s]:
                u, l = stage.u[s][a], stage.l[s][a]
                self.assertTrue(0.0 <= u <= 1.0, "U(%r,%r)=%r out of [0,1]" % (s, a, u))
                self.assertTrue(0.0 <= l <= 1.0, "L(%r,%r)=%r out of [0,1]" % (s, a, l))
                self.assertLessEqual(l, u, "sandwich violated at (%r,%r)" % (s, a))

    def test_summarize_learning_reports_the_initial_state(self):
        random.seed(41)
        stage = make_stage(six_state_simulator())
        run_round(stage, num_simulations=20)

        policy, upper, lower, error, samples = stage.summarize_learning()
        self.assertIn(1, policy)
        self.assertAlmostEqual(error, upper - lower)
        self.assertGreater(samples, 0)


# ----------------------------------------------------------------------------
# MDPSimulator sampling
# ----------------------------------------------------------------------------

class TestSimulatorSampling(unittest.TestCase):

    def test_empirical_distribution_matches_the_true_one(self):
        random.seed(43)
        sim = build_simulator(states=[1, 2, 3], actions=[(1, 'a')],
                              transitions=[(1, 'a', 1, 0.4), (1, 'a', 2, 0.05),
                                           (1, 'a', 3, 0.55)],
                              goals={3}, initial=1)
        n = 200000
        seen = {1: 0, 2: 0, 3: 0}
        for _ in range(n):
            seen[sim.step(1, 'a')[0]] += 1

        for state, truth in [(1, 0.4), (2, 0.05), (3, 0.55)]:
            self.assertAlmostEqual(seen[state] / n, truth, delta=0.01,
                                   msg="state %d" % state)

    def test_zero_probability_successor_is_never_sampled(self):
        random.seed(47)
        sim = build_simulator(states=[1, 2, 3], actions=[(1, 'a')],
                              transitions=[(1, 'a', 1, 0.5), (1, 'a', 2, 0.0),
                                           (1, 'a', 3, 0.5)],
                              goals={3}, initial=1)
        drawn = {sim.step(1, 'a')[0] for _ in range(5000)}
        self.assertEqual(drawn, {1, 3})

    def test_deterministic_transition_is_exact(self):
        sim = build_simulator(states=[1, 2], actions=[(1, 'a')],
                              transitions=[(1, 'a', 2, 1.0)], goals={2}, initial=1)
        for _ in range(1000):
            self.assertEqual(sim.step(1, 'a'), (2, True))

    def test_reward_flags_goal_states_only(self):
        sim = six_state_simulator()
        self.assertTrue(sim.step(6, 'a')[1])
        self.assertFalse(sim.step(5, 'a')[1])

    def test_cache_is_invalidated_when_a_transition_changes(self):
        random.seed(53)
        sim = build_simulator(states=[1, 2, 3], actions=[(1, 'a')],
                              transitions=[(1, 'a', 2, 0.5), (1, 'a', 3, 0.5)],
                              goals={3}, initial=1)
        self.assertEqual({sim.step(1, 'a')[0] for _ in range(200)}, {2, 3})

        sim.add_transition(1, 'a', 2, 0.0)
        sim.add_transition(1, 'a', 3, 1.0)
        self.assertEqual({sim.step(1, 'a')[0] for _ in range(200)}, {3})

    def test_default_constructed_simulators_do_not_share_an_mdp(self):
        """A `mdp=MDP()` default would be evaluated once, at definition time."""
        first, second = MDPSimulator(), MDPSimulator()

        self.assertIsNot(first.gt_mdp, second.gt_mdp)
        first.add_state(42)
        self.assertNotIn(42, second.gt_mdp.states)

    def test_invalid_state_action_pair_is_rejected(self):
        sim = six_state_simulator()
        with self.assertRaises(AssertionError):
            sim.step(1, 'a')          # action 'a' is not enabled at state 1
        with self.assertRaises(AssertionError):
            sim.step(99, 'a')         # state 99 does not exist


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    return unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    result = run_tests()
    exit(0 if result.wasSuccessful() else 1)
