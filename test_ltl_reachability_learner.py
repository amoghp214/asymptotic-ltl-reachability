"""
Unit tests for MDP class
Tests all functionality of the MDP for correctness.
"""

import unittest
import numpy as np
from mdp import MDP
from ltl_reachability_learner import LTLReachabilityLearner
from mdp_simulator import MDPSimulator


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

def setup_loop_3(confidence_error=0.1, p_min=0.1):
    # Nested ECs
    mdp = MDP(confidence_error=confidence_error, p_min=p_min)
    mdp.add_state(1)
    mdp.add_state(2)
    mdp.add_state(3)
    mdp.add_state(4)
    mdp.add_state(5)
    mdp.add_state(6)

    mdp.set_goal_states(set([6]))

    mdp.add_action_to_state(1, 'a')
    mdp.add_action_to_state(1, 'b')
    mdp.add_action_to_state(2, 'a')
    mdp.add_action_to_state(3, 'a')
    mdp.add_action_to_state(3, 'b')
    mdp.add_action_to_state(4, 'a')
    mdp.add_action_to_state(4, 'b')
    mdp.add_action_to_state(5, 'a')
    mdp.add_action_to_state(6, 'a')

    mdp.update_transition_count(1, 'a', 2, 1)
    mdp.update_transition_count(1, 'a', 3, 3)
    mdp.update_transition_count(1, 'b', 2, 3)
    mdp.update_transition_count(1, 'b', 3, 1)
    mdp.update_transition_count(2, 'a', 3, 1)
    mdp.update_transition_count(3, 'a', 2, 1)
    mdp.update_transition_count(3, 'b', 4, 1)
    mdp.update_transition_count(4, 'a', 3, 1)
    mdp.update_transition_count(4, 'b', 5, 1)
    mdp.update_transition_count(5, 'a', 4, 1)
    mdp.update_transition_count(6, 'a', 6, 1151286)

    mdp.set_initial_state(1)
    mdp.initialize_mdp_value_bounds()

    return mdp

class TestMDPFunctionality(unittest.TestCase):
    """Test MDP functionality."""
    
    def test_is_staying_state_action_pair(self):
        """Test is_staying_state_action_pair method."""
        mdp = setup_loop_2()
        learner = LTLReachabilityLearner(mdp_simulator=MDPSimulator(mdp))
        self.assertTrue(learner._is_staying_state_action_pair(mdp, set([2,3]), 3, 'a'))
        self.assertFalse(learner._is_staying_state_action_pair(mdp, set([1,2]), 1, 'a'))
        self.assertFalse(learner._is_staying_state_action_pair(mdp, set([2]), 2, 'b'))
        self.assertTrue(learner._is_staying_state_action_pair(mdp, set([2,4]), 2, 'b'))

    def test_get_staying_state_action_pairs(self):
        """Test get_staying_state_action_pairs method."""
        mdp = setup_loop_2()
        learner = LTLReachabilityLearner(mdp_simulator=MDPSimulator(mdp))
        staying_pairs = learner._get_staying_state_action_pairs(mdp, set([2,3]))
        expected_pairs = set([(s, a) for s, a in mdp.state_action_pairs if learner._is_staying_state_action_pair(mdp, set([2,3]), s, a)])
        self.assertEqual(staying_pairs, expected_pairs)
        for s, a in expected_pairs:
            self.assertTrue(learner._is_staying_state_action_pair(mdp, set([2,3]), s, a))

    def test_get_staying_mdp(self):
        """Test get_staying_mdp method."""
        mdp = setup_loop_2()
        learner = LTLReachabilityLearner(mdp_simulator=MDPSimulator(mdp))
        staying_mdp = learner._get_staying_mdp(mdp, set([2,3]))
        self.assertEqual(staying_mdp.states, set([2,3]))
        self.assertDictEqual(staying_mdp.topology,
                             {
                                    2: set('a'),
                                    3: set('a')
                             })
        self.assertDictEqual(staying_mdp.sample_counts,
                             {
                                    (2, 'a'): {3: 1.0},
                                    (3, 'a'): {2: 1.0}
                             })
    
    def test_find_all_possible_end_components(self):
        """Test find_all_possible_end_components method."""
        mdp = setup_loop_2()
        learner = LTLReachabilityLearner(mdp_simulator=MDPSimulator(mdp))
        end_components = learner._find_all_possible_end_components(mdp, mdp.states)
        expected_ecs = [
            set([2,3]),
            set([4])
        ]
        self.assertEqual(len(end_components), len(expected_ecs))
        for ec in expected_ecs:
            self.assertIn(ec, end_components)

    def test_find_all_possible_end_components_2(self):
        """Test find_all_possible_end_components method."""
        mdp = setup_loop()
        learner = LTLReachabilityLearner(mdp_simulator=MDPSimulator(mdp))
        end_components = learner._find_all_possible_end_components(mdp, mdp.states)
        expected_ecs = [
            set([1, 5]),
            set([3]),
            set([6])
        ]
        self.assertEqual(len(end_components), len(expected_ecs))
        for ec in expected_ecs:
            self.assertIn(ec, end_components)

    def test_find_all_possible_end_components_gets_maximal_ecs(self):
        """Test find_all_possible_end_components method."""
        mdp = setup_loop_3()
        learner = LTLReachabilityLearner(mdp_simulator=MDPSimulator(mdp))
        end_components = learner._find_all_possible_end_components(mdp, mdp.states)
        expected_ecs = [
            set([2, 3, 4, 5]),
            set([6])
        ]
        print(end_components)
        self.assertEqual(len(end_components), len(expected_ecs))
        for ec in expected_ecs:
            self.assertIn(ec, end_components)

    def test_end_component_detection_num_required_samples(self):
        mdp = setup_loop_3()
        learner = LTLReachabilityLearner(mdp_simulator=MDPSimulator())
        self.assertEqual(learner._end_component_detection_num_required_samples(0.001, 0.1), int(65.5630359803)+1)
        self.assertEqual(learner._end_component_detection_num_required_samples(0.0001, 0.01), int(916.421153107)+1)
        self.assertEqual(learner._end_component_detection_num_required_samples(0.1, 0.1), int(21.8543453268)+1)
        self.assertEqual(learner._end_component_detection_num_required_samples(0.00001, 0.00001), int(1151286.79003)+1)
    
    def test_confidently_detect_end_component(self):
        mdp = setup_loop_3()
        learner = LTLReachabilityLearner(mdp_simulator=MDPSimulator())
        self.assertTrue(learner._confidently_detect_end_component(mdp, set([2,3,4,5]), 0.51, 0.51))
        self.assertRaises(AssertionError, learner._confidently_detect_end_component, mdp, set([2,3,4,5]), 0.49999, 0.49999)
        self.assertTrue(learner._confidently_detect_end_component(mdp, set([6]), 0.000021, 0.000021))
        # self.assertTrue(learner._confidently_detect_end_component(mdp, set([6]), 0.00001, 0.00001))
        self.assertRaises(AssertionError, learner._confidently_detect_end_component, mdp, set([6]), 0.00001, 0.00001)
        

        learner.discovered_mdp = mdp
        self.assertTrue(learner._confidently_detect_end_component(mdp, set([2,3,4,5]), 0.51, 0.51))
        self.assertRaises(AssertionError, learner._confidently_detect_end_component, mdp, set([2,3,4,5]), 0.49999, 0.49999)
        self.assertTrue(learner._confidently_detect_end_component(mdp, set([6]), 0.000021, 0.000021))
        self.assertRaises(AssertionError, learner._confidently_detect_end_component, mdp, set([6]), 0.00001, 0.00001)

        learner.mdp_sim = MDPSimulator(mdp)
        learner.mdp_sim.gt_mdp.confidence_error = 1 * len(learner.mdp_sim.gt_mdp.state_action_pairs) / learner.mdp_sim.gt_mdp.p_min  # To make confidence width == 1
        learner.mdp_sim.gt_mdp.update_transition_probabilities()
        self.assertTrue(learner._confidently_detect_end_component(mdp, set([2,3,4,5]), 0.51, 0.51))
        self.assertTrue(learner._confidently_detect_end_component(mdp, set([2,3,4,5]), 0.49999, 0.49999))
        self.assertTrue(learner._confidently_detect_end_component(mdp, set([6]), 0.000021, 0.000021))
        self.assertTrue(learner._confidently_detect_end_component(mdp, set([6]), 0.00001, 0.00001))

    def test_calculate_transition_probability_error_tolerance(self):
        mdp = setup_loop_3()
        learner = LTLReachabilityLearner(mdp_simulator=MDPSimulator())
        self.assertAlmostEqual(learner._calculate_transition_probability_error_tolerance(0.1, 0.1, 1000), 0.00001, places=8)
        self.assertAlmostEqual(learner._calculate_transition_probability_error_tolerance(0.5, 0.5, 100), 0.0025, places=8)
        self.assertAlmostEqual(learner._calculate_transition_probability_error_tolerance(0.1, 0.1, 0), 0.01, places=8)
        self.assertAlmostEqual(learner._calculate_transition_probability_error_tolerance(0.1, 0.1, 1), 0.01, places=8)
    
    
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
