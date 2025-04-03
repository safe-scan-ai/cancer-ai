import unittest
from datetime import datetime, timezone
from cancer_ai.validator.rewarder import CompetitionResultsStore
import bittensor as bt


class TestCompetitionResultsStore(unittest.TestCase):

    def setUp(self):
        self.store = CompetitionResultsStore()
        self.competition_id = "test_competition"
        self.hotkey = "test_hotkey"
        self.score = 0.5
        self.date = datetime(2023, 1, 1, tzinfo=timezone.utc)

    def test_add_score(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        self.assertIn(self.competition_id, self.store.score_map)
        self.assertIn(self.hotkey, self.store.score_map[self.competition_id])
        self.assertEqual(len(self.store.score_map[self.competition_id][self.hotkey]), 1)
        self.assertEqual(
            self.store.score_map[self.competition_id][self.hotkey][0].score, self.score
        )
        self.assertEqual(
            self.store.score_map[self.competition_id][self.hotkey][0].date, self.date
        )

    def test_update_average_score(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        self.assertEqual(
            self.store.average_scores[self.competition_id][self.hotkey], self.score
        )

    def test_delete_dead_hotkeys(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        active_hotkeys = []
        self.store.delete_dead_hotkeys(self.competition_id, active_hotkeys)
        self.assertNotIn(self.hotkey, self.store.score_map[self.competition_id])
        self.assertNotIn(self.hotkey, self.store.average_scores[self.competition_id])

    def test_get_top_hotkey(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        top_hotkey = self.store.get_top_hotkey(self.competition_id)
        self.assertEqual(top_hotkey, self.hotkey)

    def test_delete_inactive_competitions(self):
        self.store.add_score(self.competition_id, self.hotkey, self.score, self.date)
        active_competitions = []
        self.store.delete_inactive_competitions(active_competitions)
        self.assertNotIn(self.competition_id, self.store.score_map)
        self.assertNotIn(self.competition_id, self.store.average_scores)
        self.assertNotIn(self.competition_id, self.store.current_top_hotkeys)

    def test_step_by_step(self):
        scores_sequential = [1, 2, 1.5, 1.5, 7, 8]
        averages_sequential = [1, 1.5, 1.5, 1.5, 2.6, 4.0]
        for i in range(6):
            self.store.add_score(self.competition_id, self.hotkey, scores_sequential[i])
            self.assertEqual(
                self.store.average_scores[self.competition_id][self.hotkey],
                averages_sequential[i],
            )

    def test_add_a_lot_of_runs(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        dates = [datetime(2023, 1, i, tzinfo=timezone.utc) for i in range(1, 11)]
        for score in scores:
            self.store.add_score(
                self.competition_id, self.hotkey, score, dates[scores.index(score)]
            )

        expected_average = sum(scores[-5:]) / 5
        bt.logging.debug(f"Expected average: {expected_average}")
        self.assertAlmostEqual(
            self.store.average_scores[self.competition_id][self.hotkey],
            expected_average,
        )

    def test_add_a_lot_of_runs_history(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        dates = [datetime(2023, 1, i, tzinfo=timezone.utc) for i in range(1, 13)]
        for score in scores:
            self.store.add_score(
                self.competition_id, self.hotkey, score, dates[scores.index(score)]
            )
        bt.logging.debug(
            f"Scores: {self.store.score_map[self.competition_id][self.hotkey]}"
        )
        self.assertEqual(
            len(self.store.score_map[self.competition_id][self.hotkey]), 10
        )
        expected_scores = scores[-10:]
        bt.logging.debug(f"Expected scores: {expected_scores}")
        actual_scores = [
            model_score.score
            for model_score in self.store.score_map[self.competition_id][self.hotkey]
        ]
        self.assertEqual(actual_scores, expected_scores)

    def test_average_after_history(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        dates = [datetime(2023, 1, i, tzinfo=timezone.utc) for i in range(1, 11)]
        for score in scores:
            self.store.add_score(
                self.competition_id, self.hotkey, score, dates[scores.index(score)]
            )

        expected_average = sum(scores[-5:]) / 5  # 1.0, 0.9, 0.8, 0.7, 0.6, 0.5

        self.assertAlmostEqual(
            self.store.average_scores[self.competition_id][self.hotkey],
            expected_average,
        )

        self.store.add_score(
            self.competition_id,
            self.hotkey,
            1.1,
            datetime(2023, 1, 11, tzinfo=timezone.utc),
        )
        expected_average = sum([1.1, 1.0, 0.9, 0.8, 0.7]) / 5
        self.assertAlmostEqual(
            self.store.average_scores[self.competition_id][self.hotkey],
            expected_average,
        )


if __name__ == "__main__":
    unittest.main()
