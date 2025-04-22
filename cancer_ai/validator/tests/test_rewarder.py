import unittest
import sys
import os
from datetime import datetime, timezone
from unittest.mock import patch
import bittensor as bt

# Add the project root to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from cancer_ai.validator.rewarder import CompetitionResultsStore


class TestCompetitionResultsStore(unittest.TestCase):
    def setUp(self):
        self.store = CompetitionResultsStore()
        self.competition_id = "test_competition"
        self.hotkey = "test_hotkey"
        self.score = 0.5
        self.date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        # Additional test data for extended tests
        self.competition_id_1 = "competition_1"
        self.competition_id_2 = "competition_2"
        self.hotkey_1 = "hotkey_1"
        self.hotkey_2 = "hotkey_2"
        self.hotkey_3 = "hotkey_3"
        self.score_1 = 0.8
        self.score_2 = 0.6
        self.score_3 = 0.9

    def test_add_score(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        self.assertIn(self.competition_id, self.store.score_map)
        self.assertIn(self.hotkey, self.store.score_map[self.competition_id])
        self.assertEqual(len(self.store.get_scores(self.competition_id, self.hotkey)), 1)
        self.assertEqual(
            self.store.get_newest_score(self.competition_id, self.hotkey), self.score
        )
        self.assertEqual(
            self.store.score_map[self.competition_id][self.hotkey][0].date, self.date
        )  # date is not handled by get_newest_score, keep direct access
        # Additional checks for multi-competition/hotkey
        self.store.add_score(self.competition_id_1, self.hotkey_1, self.score_1)
        self.store.add_score(self.competition_id_1, self.hotkey_2, self.score_2)
        self.store.add_score(self.competition_id_2, self.hotkey_1, self.score_2)
        self.store.add_score(self.competition_id_2, self.hotkey_3, self.score_3)
        self.assertEqual(len(self.store.get_scores(self.competition_id_1, self.hotkey_1)), 1)
        self.assertEqual(self.store.get_newest_score(self.competition_id_1, self.hotkey_1), self.score_1)
        self.assertEqual(len(self.store.get_scores(self.competition_id_2, self.hotkey_3)), 1)
        self.assertEqual(self.store.get_newest_score(self.competition_id_2, self.hotkey_3), self.score_3)

    def test_update_average_score(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        self.assertEqual(
            self.store.average_scores[self.competition_id][self.hotkey], self.score / 5
        )
        self.store.add_score(self.competition_id_1, self.hotkey_1, 0.7)
        self.store.add_score(self.competition_id_1, self.hotkey_1, 0.9)
        self.store.add_score(self.competition_id_1, self.hotkey_1, 0.8)
        expected_average = (0.7 + 0.9 + 0.8) / 5
        self.assertAlmostEqual(self.store.average_scores[self.competition_id_1][self.hotkey_1], expected_average)

    def test_delete_dead_hotkeys(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        active_hotkeys = []
        self.store.delete_dead_hotkeys(self.competition_id, active_hotkeys)
        self.assertNotIn(self.hotkey, self.store.score_map[self.competition_id])
        self.assertNotIn(self.hotkey, self.store.average_scores[self.competition_id])
        # Extended test
        self.store.add_score(self.competition_id_1, self.hotkey_1, self.score_1)
        self.store.add_score(self.competition_id_1, self.hotkey_2, self.score_2)
        self.store.add_score(self.competition_id_1, self.hotkey_3, self.score_3)
        active_hotkeys = [self.hotkey_1, self.hotkey_3]
        self.store.delete_dead_hotkeys(self.competition_id_1, active_hotkeys)
        self.assertIn(self.hotkey_1, self.store.score_map[self.competition_id_1])
        self.assertIn(self.hotkey_3, self.store.score_map[self.competition_id_1])
        self.assertNotIn(self.hotkey_2, self.store.score_map[self.competition_id_1])
        self.assertNotIn(self.hotkey_2, self.store.average_scores[self.competition_id_1])

    def test_get_top_hotkey(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        top_hotkey = self.store.get_top_hotkey(self.competition_id)
        self.assertEqual(top_hotkey, self.hotkey)
        # Extended test
        self.store.add_score(self.competition_id_1, self.hotkey_1, self.score_1)
        self.store.add_score(self.competition_id_1, self.hotkey_2, self.score_2)
        self.store.add_score(self.competition_id_1, self.hotkey_3, self.score_3)
        top_hotkey = self.store.get_top_hotkey(self.competition_id_1)
        self.assertEqual(top_hotkey, self.hotkey_3)

    def test_delete_inactive_competitions(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        active_competitions = []
        self.store.delete_inactive_competitions(active_competitions)
        self.assertNotIn(self.competition_id, self.store.score_map)
        self.assertNotIn(self.competition_id, self.store.average_scores)
        self.assertNotIn(self.competition_id, self.store.current_top_hotkeys)

    def test_step_by_step(self):
        scores_sequential = [1, 2, 1.5, 1.5, 7, 8]
        averages_sequential = [1/5, (1+2)/5, (1+2+1.5)/5, (1+2+1.5+1.5)/5, (1+2+1.5+1.5+7)/5, (2+1.5+1.5+7+8)/5]
        for i in range(6):
            self.store.add_score(self.competition_id, self.hotkey, scores_sequential[i])
            self.assertEqual(
                self.store.average_scores[self.competition_id][self.hotkey],
                averages_sequential[i],
            )

    def test_score_history_and_average(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        dates = [datetime(2023, 1, i, tzinfo=timezone.utc) for i in range(1, 13)]
        for score in scores:
            self.store.add_score(
                self.competition_id, self.hotkey, score, dates[scores.index(score)]
            )
        bt.logging.debug(
            f"Scores: {self.store.get_scores(self.competition_id, self.hotkey)}"
        )
        self.assertEqual(
            len(self.store.get_scores(self.competition_id, self.hotkey)), 10
        )
        expected_scores = scores[-10:]
        bt.logging.debug(f"Expected scores: {expected_scores}")
        actual_scores = self.store.get_scores(self.competition_id, self.hotkey)[::-1]  # oldest to newest
        self.assertEqual(actual_scores, expected_scores)

    def test_average_after_history(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for score in scores:
            self.store.add_score(self.competition_id, self.hotkey, score)
        expected_average = sum(scores[-5:]) / 5
        self.assertAlmostEqual(self.store.average_scores[self.competition_id][self.hotkey], expected_average)

        # self.assertEqual(top_hotkey, self.hotkey_3)

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
        self.assertEqual(self.store.get_newest_score(self.competition_id_1, self.hotkey_1), 0.0)
        
        # Test adding a negative score (should still work, though it might be invalid in real usage)
        self.store.add_score(self.competition_id_1, self.hotkey_2, -0.5)
        self.assertEqual(self.store.get_newest_score(self.competition_id_1, self.hotkey_2), -0.5)
        
        # Test with empty active_hotkeys list
        self.store.delete_dead_hotkeys(self.competition_id_1, [])
        # All hotkeys should be deleted
        self.assertEqual(len(self.store.score_map[self.competition_id_1]), 0)


if __name__ == "__main__":
    unittest.main()
