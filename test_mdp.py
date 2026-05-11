"""
Unit tests for MDP class
Tests all functionality of the MDP for correctness.
"""

import unittest
import numpy as np
from mdp import MDP


def setup(confidence_error=0.1, p_min=0.1):
    mdp = MDP(confidence_error=confidence_error, p_min=p_min)
    mdp.add_state(1)
    mdp.add_state(2)
    mdp.add_state(3)

    mdp.set_goal_states(set([3]))
    
    mdp.add_action_to_state(1, 'a')
    mdp.add_action_to_state(1, 'b')
    mdp.add_action_to_state(2, 'b')
    mdp.add_action_to_state(3, 'a')

    mdp.update_transition_count(1, 'a', 2, 2)
    mdp.update_transition_count(1, 'a', 3, 2)
    mdp.update_transition_count(1, 'b', 2)
    mdp.record_sample(1, 'b', 2)
    mdp.update_transition_count(2, 'b', 2, 4)
    mdp.update_transition_count(2, 'b', 3, 6)
    mdp.update_transition_count(3, 'a', 3, 5)

    mdp.set_initial_state(1)
    mdp.initialize_mdp_value_bounds()

    return mdp

def setup_1(confidence_error=0.1, p_min=0.1):
    mdp = MDP(confidence_error=confidence_error, p_min=p_min)
    mdp.add_state(1)
    mdp.add_state(2)
    mdp.add_state(3)
    mdp.add_state(4)
    mdp.add_state(5)
    mdp.add_state(6)

    mdp.set_goal_states(set([3, 6]))

    mdp.add_action_to_state(1, 'b')
    mdp.add_action_to_state(1, 'c')
    mdp.add_action_to_state(2, 'b')
    mdp.add_action_to_state(3, 'a')
    mdp.add_action_to_state(4, 'a')
    mdp.add_action_to_state(5, 'a')
    mdp.add_action_to_state(6, 'a')

    mdp.record_sample(1, 'b', 4)

    mdp.update_transition_count(1, 'c', 0, 1)
    mdp.update_transition_count(1, 'c', 1, 4)
    mdp.update_transition_count(1, 'c', 2, 1)
    mdp.update_transition_count(1, 'c', 4, 1)
    mdp.update_transition_count(1, 'c', 5, 2)
    mdp.update_transition_count(1, 'c', 6, 2)
    mdp.update_transition_count(2, 'b', 2, 5)
    mdp.update_transition_count(2, 'b', 3, 5)
    mdp.update_transition_count(3, 'a', 3, 5)
    mdp.update_transition_count(4, 'a', 3, 5)
    mdp.update_transition_count(5, 'a', 5, 10)
    mdp.update_transition_count(6, 'a', 6, 1)

    mdp.set_initial_state(1)
    mdp.initialize_mdp_value_bounds()

    return mdp
    
def setup_2(confidence_error=0.1, p_min=0.1):
    mdp = MDP(confidence_error=confidence_error, p_min=p_min)
    mdp.add_state(1)
    mdp.add_state(2)
    mdp.add_state(3)

    mdp.set_goal_states(set([3]))
    
    mdp.add_action_to_state(1, 'a')
    mdp.add_action_to_state(1, 'b')
    mdp.add_action_to_state(2, 'b')
    mdp.add_action_to_state(3, 'a')

    mdp.update_transition_count(1, 'a', 2, 20)
    mdp.update_transition_count(1, 'a', 3, 20)
    mdp.update_transition_count(1, 'b', 2, 30)
    mdp.record_sample(1, 'b', 2)
    mdp.update_transition_count(2, 'b', 2, 40)
    mdp.update_transition_count(2, 'b', 3, 60)
    mdp.update_transition_count(3, 'a', 3, 50)

    mdp.set_initial_state(1)
    mdp.initialize_mdp_value_bounds()

    return mdp

def setup_3(confidence_error=0.1, p_min=0.1):
    mdp = MDP(confidence_error=confidence_error, p_min=p_min)
    mdp.add_state(1)
    mdp.add_state(2)

    mdp.set_goal_states(set([2]))
    
    mdp.add_action_to_state(1, 'a')
    mdp.add_action_to_state(1, 'b')
    mdp.add_action_to_state(1, 'c')
    mdp.add_action_to_state(1, 'd')
    mdp.add_action_to_state(1, 'e')
    mdp.add_action_to_state(1, 'f')
    mdp.add_action_to_state(2, 'a')

    mdp.update_transition_count(1, 'a', 1, 20)
    mdp.update_transition_count(1, 'a', 2, 20)
    mdp.update_transition_count(1, 'b', 1, 1)
    mdp.update_transition_count(1, 'b', 2, 19)
    mdp.update_transition_count(1, 'c', 1, 19)
    mdp.update_transition_count(1, 'c', 2, 1)
    mdp.update_transition_count(1, 'd', 1, 5)
    mdp.update_transition_count(1, 'd', 2, 15)
    mdp.update_transition_count(1, 'e', 1, 10)
    mdp.update_transition_count(1, 'e', 2, 20)
    mdp.update_transition_count(1, 'f', 1, 20)
    mdp.update_transition_count(1, 'f', 2, 20)


    mdp.update_transition_count(2, 'a', 2, 5)

    mdp.set_initial_state(1)
    mdp.initialize_mdp_value_bounds()

    return mdp

def setup_loop(confidence_error=0.1, p_min=0.1):
    mdp = MDP(confidence_error=confidence_error, p_min=p_min)
    mdp.add_state(1)
    mdp.add_state(2)
    mdp.add_state(3)
    mdp.add_state(4)
    mdp.add_state(5)
    mdp.add_state(6)

    mdp.set_goal_states(set([3, 6]))

    mdp.add_action_to_state(1, 'b')
    mdp.add_action_to_state(1, 'c')
    mdp.add_action_to_state(2, 'b')
    mdp.add_action_to_state(3, 'a')
    mdp.add_action_to_state(4, 'a')
    mdp.add_action_to_state(5, 'a')
    mdp.add_action_to_state(6, 'a')

    mdp.update_transition_count(1, 'b', 5, 1)
    mdp.update_transition_count(1, 'c', 0, 1)
    mdp.update_transition_count(1, 'c', 1, 4)
    mdp.update_transition_count(1, 'c', 2, 1)
    mdp.update_transition_count(2, 'b', 2, 5)
    mdp.update_transition_count(2, 'b', 3, 5)
    mdp.update_transition_count(3, 'a', 3, 5)
    mdp.update_transition_count(4, 'a', 3, 5)
    mdp.update_transition_count(5, 'a', 1, 10)
    mdp.update_transition_count(5, 'a', 5, 10)
    mdp.update_transition_count(6, 'a', 6, 1)

    mdp.set_initial_state(1)
    mdp.initialize_mdp_value_bounds()

    return mdp

def setup_loop_2(confidence_error=0.1, p_min=0.1):
    mdp = MDP(confidence_error=confidence_error, p_min=p_min)
    mdp.add_state(1)
    mdp.add_state(2)
    mdp.add_state(3)
    mdp.add_state(4)

    mdp.set_goal_states(set([4]))

    mdp.add_action_to_state(1, 'a')
    mdp.add_action_to_state(1, 'b')
    mdp.add_action_to_state(2, 'a')
    mdp.add_action_to_state(2, 'b')
    mdp.add_action_to_state(3, 'a')
    mdp.add_action_to_state(4, 'a')

    mdp.update_transition_count(1, 'a', 2, 1)
    mdp.update_transition_count(1, 'a', 3, 3)
    mdp.update_transition_count(1, 'b', 2, 3)
    mdp.update_transition_count(1, 'b', 3, 1)
    mdp.update_transition_count(2, 'a', 3, 1)
    mdp.update_transition_count(2, 'b', 4, 1)
    mdp.update_transition_count(3, 'a', 2, 1)
    mdp.update_transition_count(4, 'a', 4, 1)

    mdp.set_initial_state(1)
    mdp.initialize_mdp_value_bounds()

    return mdp

class TestMDPFunctionality(unittest.TestCase):
    """Test MDP functionality."""
    
    def test_mdp_setup(self):
        """Test setup functionality (addng states, actions, transitions)"""
        mdp = setup(0.2, 0.2)

        self.assertEqual(mdp.confidence_error, 0.2)
        self.assertEqual(mdp.p_min, 0.2)

        self.assertSetEqual(mdp.states, set([1,2,3]))
        self.assertSetEqual(mdp.goal_states, set([3]))
        self.assertDictEqual(mdp.topology,
                         {
                             1: set(['a', 'b']),
                             2: set(['b']),
                             3: set(['a'])
                         })
        
        self.assertSetEqual(mdp.state_action_pairs, set([(1, 'a'), (1, 'b'), (2, 'b'), (3, 'a')]))
        self.assertDictEqual(mdp.s_value_bounds,
                             {
                                 1: [0, 1],
                                 2: [0, 1],
                                 3: [1, 1]
                             })
        self.assertDictEqual(mdp.sa_value_bounds,
                             {
                                 (1, 'a'): [0, 1],
                                 (1, 'b'): [0, 1],
                                 (2, 'b'): [0, 1],
                                 (3, 'a'): [0, 1]
                             })
        self.assertDictEqual(mdp.sample_counts,
                             {
                                 (1, 'a'): {2: 2, 3: 2},
                                 (1, 'b'): {2: 2},
                                 (2, 'b'): {2: 4, 3: 6},
                                 (3, 'a'): {3: 5}
                             })
        self.assertEqual(mdp.initial_state, 1)

        self.assertEqual(mdp.get_sample_count(1, 'a', 2), 2)
        self.assertEqual(mdp.get_sample_count(1, 'a', 3), 2)
        self.assertEqual(mdp.get_sample_count(1, 'a'), 4)

    def test_remove_state(self):
        mdp = setup(0.2, 0.2)
        self.assertEqual(mdp.states, set([1,2,3]))
        self.assertEqual(mdp.goal_states, set([3]))
        self.assertEqual(mdp.state_action_pairs, set([(1, 'a'), (1, 'b'), (2, 'b'), (3, 'a')]))
        self.assertEqual(mdp.topology,
                         {
                             1: set(['a', 'b']),
                             2: set(['b']),
                             3: set(['a'])
                         })
        self.assertIn(3, mdp.s_value_bounds)
        self.assertIn((3, 'a'), mdp.sa_value_bounds)
        self.assertIn((3, 'a'), mdp.sample_counts)
        self.assertIn((3, 'a'), mdp.transition_probabilities)
        
        mdp.remove_state(3)

        self.assertEqual(mdp.states, set([1,2]))
        self.assertEqual(mdp.goal_states, set())
        self.assertEqual(mdp.state_action_pairs, set([(1, 'a'), (1, 'b'), (2, 'b')]))
        self.assertEqual(mdp.topology,
                         {
                             1: set(['a', 'b']),
                             2: set(['b'])
                         })
        self.assertNotIn(3, mdp.s_value_bounds)
        self.assertNotIn((3, 'a'), mdp.sa_value_bounds)
        self.assertNotIn((3, 'a'), mdp.sample_counts)
        self.assertNotIn((3, 'a'), mdp.transition_probabilities)

    def test_add_super_state(self):
        mdp = setup_loop(0.2, 0.2)

        mdp.add_super_state([set([1,5]), set([(1, 'b'), (5, 'a')])])

        for s, a in mdp.state_action_pairs:
            mdp._update_transition_probabilities(s, a)
        
        

        self.assertIn('mec_1', mdp.states)
        self.assertEqual(mdp.initial_state, 'mec_1')        
        self.assertEqual(mdp.MEC_states['mec_1'], set([1,5]))
        self.assertEqual(mdp.state_MEC[1], 'mec_1')
        self.assertEqual(mdp.state_MEC[5], 'mec_1')
        self.assertEqual(mdp.topology['mec_1'], set([(1, 'b'), (1, 'c'), (5, 'a')]))
        self.assertEqual(mdp.s_value_bounds['mec_1'], [0,1])
        self.assertDictContainsSubset(
            {('mec_1', (1, 'b')): [0,0],
             ('mec_1', (1, 'c')): [0,1],
             ('mec_1', (5, 'a')): [0,0]},
            mdp.sa_value_bounds
        )
        self.assertDictContainsSubset(
            {('mec_1', (1, 'b')): {5: 1},
             ('mec_1', (1, 'c')): {0: 1, 1: 4, 2: 1},
             ('mec_1', (5, 'a')): {1: 10, 5: 10}},
            mdp.sample_counts
        )

    def test_collapse_MEC_transitions(self):
        mdp = setup_loop_2(0.2, 0.2)

        for s, a in mdp.state_action_pairs:
            mdp._update_transition_probabilities(s, a)
        
        mdp.add_super_state([set([2,3]), set([(2, 'a'), (3, 'a')])])
        for s in mdp.states:
            mdp.collapse_state_MEC_transitions(s)

        self.assertDictContainsSubset({
                (1, 'a'): { 'mec_1': 4 },
                (1, 'b'): { 'mec_1': 4 },
                ('mec_1', (2, 'a')): { 'mec_1': 1},
                ('mec_1', (2, 'b')): { 4: 1},
                ('mec_1', (3, 'a')): { 'mec_1': 1 },
                (4, 'a'): {4: 1}
            },
            mdp.sample_counts,
        )

    def test_collapse_mdp(self):
        mdp = setup_loop_2(0.2, 0.2)

        for s, a in mdp.state_action_pairs:
            mdp._update_transition_probabilities(s, a)
        
        mdp.collapse([[set([2,3]), set([(2, 'a'), (3, 'a')])]])

        self.assertDictEqual({
                (1, 'a'): { 'mec_1': 4 },
                (1, 'b'): { 'mec_1': 4 },
                ('mec_1', (2, 'a')): { 'mec_1': 1},
                ('mec_1', (2, 'b')): { 4: 1},
                ('mec_1', (3, 'a')): { 'mec_1': 1 },
                (4, 'a'): {4: 1}
            },
            mdp.sample_counts,
        )
        self.assertDictEqual({
                1: set(['a', 'b']),
                'mec_1': set([(2, 'a'), (2, 'b'), (3, 'a')]),
                4: set(['a'])
            },
            mdp.topology
        )
        self.assertSetEqual(mdp.states, set([1, 'mec_1', 4]))
        self.assertEqual(mdp.initial_state, 1)
        self.assertDictEqual({
                1: [0,1],
                'mec_1': [0,1],
                4: [1,1]
            },
            mdp.s_value_bounds
        )
        self.assertDictEqual({
                (1, 'a'): [0,1],
                (1, 'b'): [0,1],
                ('mec_1', (2, 'a')): [0,0],
                ('mec_1', (2, 'b')): [0,1],
                ('mec_1', (3, 'a')): [0,0],
                (4, 'a'): [0,1]
            },
            mdp.sa_value_bounds
        )
                                    
    def test_calculate_transition_probability_confidence_error(self):
        mdp = setup(0.2, 0.2)
        self.assertAlmostEqual(mdp._calculate_transition_probability_confidence_error(), 0.01, places=8)

    def test_calculate_transition_probability_confidence_width(self):
        mdp = setup(0.2, 0.2)

        self.assertAlmostEqual(mdp._calculate_transition_probability_confidence_width(1, 'a', 0.01), 0.7587135646925731)
        self.assertAlmostEqual(mdp._calculate_transition_probability_confidence_width(1, 'b', 0.01), 1.0729830131446736)
        self.assertAlmostEqual(mdp._calculate_transition_probability_confidence_width(2, 'b', 0.01), 0.47985259121880813)
        self.assertAlmostEqual(mdp._calculate_transition_probability_confidence_width(3, 'a', 0.01), 0.6786140424415111)
        
    def test_transition_probabilities_update(self):
        mdp = setup(0.2, 0.2)

        for s, a in mdp.state_action_pairs:
            mdp._update_transition_probabilities(s, a)


        self.assertDictEqual(mdp.transition_probabilities, 
                         {
                             (1, 'a'): {2: 0, 3: 0},
                             (1, 'b'): {2: 0},
                             (2, 'b'): {2: 0, 3: 0.08530021534160148},
                             (3, 'a'): {3: 0.27210458398558135},
                         })
        
    def test_set_goal_states(self):
        mdp = setup()
        mdp.set_goal_states(set([2]))

        self.assertSetEqual(mdp.goal_states, set([2, 3]))
        self.assertDictEqual(mdp.s_value_bounds,
                             {
                                 1: [0, 1],
                                 2: [1, 1],
                                 3: [1, 1]
                             })

    def test_get_sample_counts(self):
        mdp = setup()

        self.assertEqual(mdp.get_sample_count(1, 'd'), 0)
        self.assertEqual(mdp.get_sample_count(1, 'a'), 4)
        self.assertEqual(mdp.get_sample_count(1, 'a', 2), 2)
    
    def test_initialize_mdp_value_bounds(self):
        mdp = setup()

        mdp.s_value_bounds[1] = [-3, -2]
        mdp.s_value_bounds[2] = [-0.5, -20]
        mdp.s_value_bounds[3] = [1, 4]

        mdp.sa_value_bounds[(1, 'a')] = [-3, -2.9]
        mdp.sa_value_bounds[(1, 'b')] = [0.1, 1]
        mdp.sa_value_bounds[(2, 'b')] = [0, 0.9]
        mdp.sa_value_bounds[(3, 'a')] = [0.2, 0.7]

        mdp.initialize_mdp_value_bounds()

        self.assertDictEqual(mdp.s_value_bounds,
                             {
                                 1: [0, 1],
                                 2: [0, 1],
                                 3: [1, 1]
                             })
        self.assertDictEqual(mdp.sa_value_bounds,
                             {
                                 (1, 'a'): [0, 1],
                                 (1, 'b'): [0, 1],
                                 (2, 'b'): [0, 1],
                                 (3, 'a'): [0, 1]
                             })
    
    def test_mdp_merge(self):
        mdp1 = setup(0.1, 0.1)
        mdp2 = setup(0.2, 0.2)

        mdp1.merge(mdp2)

        self.assertEqual(mdp1.confidence_error, 0.1)
        self.assertEqual(mdp1.p_min, 0.1)
        self.assertSetEqual(mdp1.states, mdp2.states)
        self.assertSetEqual(mdp1.goal_states, mdp2.goal_states)
        self.assertDictEqual(mdp1.topology, mdp2.topology)
        self.assertSetEqual(mdp1.state_action_pairs, mdp2.state_action_pairs)
        
        for sa, tc in mdp1.sample_counts.items():
            for t, c in tc.items():
                self.assertEqual(c, mdp2.sample_counts[sa][t] * 2)
    
    def test_diff_mdp_merge(self):
        mdp1 = setup(0.1, 0.1)
        mdp2 = setup_1(0.2, 0.2)

        mdp1.merge(mdp2)

        self.assertEqual(mdp1.confidence_error, 0.1)
        self.assertEqual(mdp1.p_min, 0.1)
        self.assertSetEqual(mdp1.states, set([1,2,3,4,5,6]))
        self.assertSetEqual(mdp1.goal_states, set([3,6]))
        self.assertDictEqual(mdp1.topology,
                             {
                                 1: set(['a', 'b', 'c']),
                                 2: set(['b']),
                                 3: set(['a']),
                                 4: set(['a']),
                                 5: set(['a']),
                                 6: set(['a']),
                             })
        self.assertSetEqual(mdp1.state_action_pairs,
                            set([
                                (1, 'a'),
                                (1, 'b'),
                                (1, 'c'),
                                (2, 'b'),
                                (3, 'a'),
                                (4, 'a'),
                                (5, 'a'),
                                (6, 'a')
                            ]))
        self.assertDictEqual(mdp1.sample_counts,
                             {
                                (1, 'a'): {2: 2, 3: 2},
                                (1, 'b'): {2: 2, 4: 1},
                                (1, 'c'): {0: 1, 1: 4, 2: 1, 4: 1, 5: 2, 6: 2},
                                (2, 'b'): {2: 9, 3: 11},
                                (3, 'a'): {3: 10},
                                (4, 'a'): {3: 5},
                                (5, 'a'): {5: 10},
                                (6, 'a'): {6: 1},
                             })
        
    def test_update_state_action_value_bounds(self):
        mdp = setup(0.2, 0.2)

        for s, a in mdp.state_action_pairs:
            mdp._update_transition_probabilities(s, a)
            mdp._update_sa_value_bounds(s, a, inplace=True)

        self.assertDictEqual(mdp.sa_value_bounds,
                             {
                                 (1, 'a'): [0, 1],
                                 (1, 'b'): [0, 1],
                                 (2, 'b'): [0.08530021534160148, 1],
                                 (3, 'a'): [1, 1]
                             })
    
    def test_update_state_value_bounds(self):
        mdp = setup(0.2, 0.2)

        for s, a in mdp.state_action_pairs:
            mdp._update_transition_probabilities(s, a)
            mdp._update_sa_value_bounds(s, a, inplace=True)

        new_s_val_bounds = dict()
        for s in mdp.states:
            new_s_val_bounds[s] = mdp._update_s_value_bounds(s)
        mdp.s_value_bounds = new_s_val_bounds

        self.assertDictEqual(mdp.s_value_bounds,
                             {
                                 1: [0, 1],
                                 2: [0.08530021534160148, 1],
                                 3: [1, 1]
                             })
        
    def test_static_bvi_update(self):
        mdp = setup(0.2, 0.2)

        # Iteration 1
        mdp.update_transition_probabilities()
        mdp.bvi_update()
        self.assertDictEqual(mdp.sa_value_bounds,
                             {
                                (1, 'a'): [0, 1],
                                (1, 'b'): [0, 1],
                                (2, 'b'): [0.08530021534160148, 1],
                                (3, 'a'): [1, 1]
                             })
        self.assertDictEqual(mdp.s_value_bounds,
                             {
                                1: [0, 1],
                                2: [0.08530021534160148, 1],
                                3: [1, 1]
                             })
        # Iteration 2 (no change since there are not enough samples)
        mdp.bvi_update()
        self.assertDictEqual(mdp.sa_value_bounds,
                             {
                                 (1, 'a'): [0, 1],
                                 (1, 'b'): [0, 1],
                                 (2, 'b'): [0.08530021534160148, 1],
                                 (3, 'a'): [1, 1]
                             })
        self.assertDictEqual(mdp.s_value_bounds,
                             {
                                 1: [0, 1],
                                 2: [0.08530021534160148, 1],
                                 3: [1, 1]
                             })
                             
        
    def test_bvi_update(self):
        mdp = setup_2(0.2, 0.2)

        # Iteration 1
        mdp.update_transition_probabilities()
        mdp.bvi_update()
        mdp.update_learned_policy()
        self.assertDictEqual(mdp.sa_value_bounds,
                             {
                                (1, 'a'): [0.24265010767080075, 1],
                                (1, 'b'): [0, 1],
                                (2, 'b'): [0.43723763692812706, 1],
                                (3, 'a'): [1, 1]
                             })
        self.assertDictEqual(mdp.s_value_bounds,
                             {
                                1: [0.24265010767080075, 1],
                                2: [0.43723763692812706, 1],
                                3: [1, 1]
                             })
        self.assertDictEqual(mdp.learned_policy,
                             {
                                 1: 'a',
                                 2: 'b'
                             })
        # Iteration 2 (expect change since there are enough samples)
        mdp.bvi_update()
        mdp.update_learned_policy()
        self.assertDictEqual(mdp.sa_value_bounds,
                             {
                                 (1, 'a'): [0.34874586734913726, 1],
                                 (1, 'b'): [0.3094200312020634, 1],
                                 (2, 'b'): [0.5409668606889944, 1],
                                 (3, 'a'): [1, 1]
                             })
        self.assertDictEqual(mdp.s_value_bounds,
                             {
                                 1: [0.34874586734913726, 1],
                                 2: [0.5409668606889944, 1],
                                 3: [1, 1]
                             })
        self.assertDictEqual(mdp.learned_policy,
                             {
                                 1: 'a',
                                 2: 'b'
                             })
        
    def test_sample_best_action_many_best(self):
        mdp = setup_3()
        mdp.sa_value_bounds = {
            (1, 'a'): [0, 1],
            (1, 'b'): [0, 1],
            (1, 'c'): [0, 0.2],
            (1, 'd'): [0, 0.99],
            (1, 'e'): [0, 0.5],
            (1, 'f'): [0, 1],
            (2, 'a'): [1, 1]
        }
        sampled_best_actions = set()
        for _ in range(0, 1000):
            curr_action = mdp.sample_best_action_from_state(1)
            sampled_best_actions.add(curr_action)
        
        self.assertEqual(sampled_best_actions, set(['a', 'b', 'f']))

    def test_sample_best_action_one_best(self):
        mdp = setup_3()
        mdp.sa_value_bounds = {
            (1, 'a'): [0, 1],
            (1, 'b'): [0, 0.9],
            (1, 'c'): [0, 0.2],
            (1, 'd'): [0, 0.99],
            (1, 'e'): [0, 0.5],
            (1, 'f'): [0, 0.9],
            (2, 'a'): [1, 1]
        }
        sampled_best_actions = set()
        for _ in range(0, 1000):
            curr_action = mdp.sample_best_action_from_state(1)
            sampled_best_actions.add(curr_action)
        
        self.assertEqual(sampled_best_actions, set(['a']))




def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMDPFunctionality))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    exit(0 if result.wasSuccessful() else 1)
