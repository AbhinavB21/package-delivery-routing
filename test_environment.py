import unittest
from environment import PackageEnv

class TestPackageEnv(unittest.TestCase):
    def setUp(self):
        self.env = PackageEnv(num_agents=2, num_packages=2)
    
    def test_initialization(self):
        """Test if environment initializes correctly"""
        self.assertEqual(len(self.env.agent_positions), 2)
        self.assertEqual(len(self.env.package_positions), 2)
        self.assertEqual(len(self.env.package_picked), 2)
        self.assertEqual(self.env.fuel_consumed, [1000, 1000])

    def test_reset(self):
        """Test reset functionality"""
        state, _, _ = self.env.reset()
        self.assertIn('agent_positions', state)
        self.assertIn('package_positions', state)
        self.assertIn('package_picked', state)
        self.assertIn('fuel_consumed', state)

    def test_movement(self):
        """Test agent movement"""
        initial_pos = self.env.current_state['agent_positions'][0]
        
        # Test all movement directions
        for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            self.env.reset()
            # Place agent away from borders to test movement
            self.env.current_state['agent_positions'][0] = (5, 5)
            result, reward = self.env.play_turn(action, 0)
            self.assertNotEqual(self.env.current_state['agent_positions'][0], (5, 5))
            self.assertEqual(reward, -1)

    def test_pickup_drop(self):
        """Test package pickup and drop mechanics"""
        # Test pickup
        self.env.reset()
        agent_pos = (5, 5)
        self.env.current_state['agent_positions'][0] = agent_pos
        self.env.current_state['package_positions'][0] = agent_pos
        
        result, reward = self.env.play_turn('PICKUP', 0)
        self.assertTrue(self.env.current_state['package_picked'][0])
        self.assertEqual(reward, 5000)

        # Test drop at goal
        self.env.current_state['agent_positions'][0] = self.env.goal_room
        result, reward = self.env.play_turn('DROP', 0)
        self.assertEqual(reward, 10000)

    def test_terminal_states(self):
        """Test terminal state conditions"""
        # Test winning condition
        self.env.reset()
        self.env.current_state['agent_positions'][0] = self.env.goal_room
        self.env.current_state['package_picked'][0] = True
        self.assertEqual(self.env.is_terminal(0), 'DROP')

        # Test fuel depletion
        self.env.reset()
        self.env.current_state['fuel_consumed'][0] = 0
        self.assertEqual(self.env.is_terminal(0), 'EMPTY')

    def test_step_function(self):
        """Test the main step function"""
        # Test with string action
        state, reward, done, info = self.env.step('UP', 0)
        self.assertIsInstance(state, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

        # Test with numeric action
        state, reward, done, info = self.env.step(0, 0)  # 0 corresponds to 'UP'
        self.assertIsInstance(state, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

if __name__ == '__main__':
    unittest.main() 