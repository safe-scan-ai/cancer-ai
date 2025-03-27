import unittest
import sys
import os
from datetime import datetime, timezone
from unittest.mock import patch

# Add the project root to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from cancer_ai.validator.rewarder import CompetitionResultsStore


class TestCompetitionResultsStore(unittest.TestCase):
    def setUp(self):
        """Set up a fresh CompetitionResultsStore instance for each test."""
        self.store = CompetitionResultsStore()
        
        # Define test data
        self.competition_id_1 = "competition_1"
        self.competition_id_2 = "competition_2"
        self.hotkey_1 = "hotkey_1"
        self.hotkey_2 = "hotkey_2"
        self.hotkey_3 = "hotkey_3"
        self.score_1 = 0.8
        self.score_2 = 0.6
        self.score_3 = 0.9

    @patch('cancer_ai.validator.rewarder.datetime')
    def test_add_score(self, mock_datetime):
        """Test adding scores to the store."""
        # Mock datetime.now() to return a fixed time
        mock_now = datetime(2025, 3, 25, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        
        # Add scores to competition 1
        self.store.add_score(self.competition_id_1, self.hotkey_1, self.score_1)
        self.store.add_score(self.competition_id_1, self.hotkey_2, self.score_2)
        
        # Add scores to competition 2
        self.store.add_score(self.competition_id_2, self.hotkey_1, self.score_2)
        self.store.add_score(self.competition_id_2, self.hotkey_3, self.score_3)
        
        # Verify scores were added correctly
        self.assertEqual(len(self.store.score_map[self.competition_id_1][self.hotkey_1]), 1)
        self.assertEqual(self.store.score_map[self.competition_id_1][self.hotkey_1][0].score, self.score_1)
        self.assertEqual(self.store.score_map[self.competition_id_1][self.hotkey_1][0].date, mock_now)
        
        self.assertEqual(len(self.store.score_map[self.competition_id_1][self.hotkey_2]), 1)
        self.assertEqual(self.store.score_map[self.competition_id_1][self.hotkey_2][0].score, self.score_2)
        
        self.assertEqual(len(self.store.score_map[self.competition_id_2][self.hotkey_1]), 1)
        self.assertEqual(self.store.score_map[self.competition_id_2][self.hotkey_1][0].score, self.score_2)
        
        self.assertEqual(len(self.store.score_map[self.competition_id_2][self.hotkey_3]), 1)
        self.assertEqual(self.store.score_map[self.competition_id_2][self.hotkey_3][0].score, self.score_3)

    def test_update_average_score(self):
        """Test updating average scores."""
        # Add multiple scores for the same hotkey
        self.store.add_score(self.competition_id_1, self.hotkey_1, 0.7)
        self.store.add_score(self.competition_id_1, self.hotkey_1, 0.9)
        self.store.add_score(self.competition_id_1, self.hotkey_1, 0.8)
        
        # Verify average score was calculated correctly (automatically updated by add_score)
        expected_average = (0.7 + 0.9 + 0.8) / 3
        self.assertAlmostEqual(self.store.average_scores[self.competition_id_1][self.hotkey_1], expected_average)

    def test_delete_dead_hotkeys(self):
        """Test deleting hotkeys that are no longer active."""
        # Add scores for multiple hotkeys
        self.store.add_score(self.competition_id_1, self.hotkey_1, self.score_1)
        self.store.add_score(self.competition_id_1, self.hotkey_2, self.score_2)
        self.store.add_score(self.competition_id_1, self.hotkey_3, self.score_3)
        
        # Average scores are automatically updated by add_score
        # No need to call update_average_score explicitly
        
        # Define active hotkeys (excluding hotkey_2)
        active_hotkeys = [self.hotkey_1, self.hotkey_3]
        
        # Delete dead hotkeys
        self.store.delete_dead_hotkeys(self.competition_id_1, active_hotkeys)
        
        # Verify hotkey_2 was deleted
        self.assertIn(self.hotkey_1, self.store.score_map[self.competition_id_1])
        self.assertIn(self.hotkey_3, self.store.score_map[self.competition_id_1])
        self.assertNotIn(self.hotkey_2, self.store.score_map[self.competition_id_1])
        self.assertNotIn(self.hotkey_2, self.store.average_scores[self.competition_id_1])

    def test_get_top_hotkey(self):
        """Test getting the hotkey with the highest average score."""
        # Add scores for multiple hotkeys
        self.store.add_score(self.competition_id_1, self.hotkey_1, self.score_1)
        self.store.add_score(self.competition_id_1, self.hotkey_2, self.score_2)
        self.store.add_score(self.competition_id_1, self.hotkey_3, self.score_3)
        
        # Average scores are automatically updated by add_score
        # No need to call update_average_score explicitly
        
        # Get top hotkey
        top_hotkey = self.store.get_top_hotkey(self.competition_id_1)
        
        # Verify top hotkey is hotkey_3 (with highest score)
        self.assertEqual(top_hotkey, self.hotkey_3)

    def test_get_top_hotkey_empty_competition(self):
        """Test getting top hotkey for a competition with no scores."""
        # Try to get top hotkey for a non-existent competition
        with self.assertRaises(ValueError):
            self.store.get_top_hotkey("non_existent_competition")

    def test_get_competitions(self):
        """Test getting all competition IDs."""
        # Add scores to multiple competitions
        self.store.add_score(self.competition_id_1, self.hotkey_1, self.score_1)
        self.store.add_score(self.competition_id_2, self.hotkey_2, self.score_2)
        
        # Get all competitions
        competitions = self.store.get_competitions()
        
        # Verify both competitions are returned
        self.assertEqual(len(competitions), 2)
        self.assertIn(self.competition_id_1, competitions)
        self.assertIn(self.competition_id_2, competitions)

    @patch('cancer_ai.validator.rewarder.datetime')
    def test_model_dump_and_load(self, mock_datetime):
        """Test serializing and deserializing the store."""
        # Mock datetime.now() to return a fixed time
        mock_now = datetime(2025, 3, 25, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        
        # Add scores to the store
        self.store.add_score(self.competition_id_1, self.hotkey_1, self.score_1)
        self.store.add_score(self.competition_id_2, self.hotkey_2, self.score_2)
        
        # Dump the model to a dict
        dumped = self.store.model_dump()
        
        # Verify the dumped data has the expected structure
        # Note: We're not testing model_load here since it's not implemented in the class
        # Instead we're just checking that model_dump works correctly
        self.assertEqual(len(dumped), 3)  # score_map, average_scores, and current_top_hotkeys
        self.assertIn('score_map', dumped)
        self.assertIn('average_scores', dumped)

    @patch('cancer_ai.validator.rewarder.datetime')
    def test_edge_cases(self, mock_datetime):
        """Test edge cases and boundary conditions."""
        # Mock datetime.now() to return a fixed time
        mock_now = datetime(2025, 3, 25, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        
        # Test adding a score of 0
        self.store.add_score(self.competition_id_1, self.hotkey_1, 0.0)
        self.assertEqual(self.store.score_map[self.competition_id_1][self.hotkey_1][0].score, 0.0)
        
        # Test adding a negative score (should still work, though it might be invalid in real usage)
        self.store.add_score(self.competition_id_1, self.hotkey_2, -0.5)
        self.assertEqual(self.store.score_map[self.competition_id_1][self.hotkey_2][0].score, -0.5)
        
        # Test with empty active_hotkeys list
        self.store.delete_dead_hotkeys(self.competition_id_1, [])
        # All hotkeys should be deleted
        self.assertEqual(len(self.store.score_map[self.competition_id_1]), 0)


if __name__ == "__main__":
    unittest.main()
