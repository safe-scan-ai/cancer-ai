from pydantic import BaseModel
import bittensor as bt
from datetime import datetime, timezone

from cancer_ai.validator.competition_handlers.base_handler import ModelEvaluationResult
from cancer_ai.validator.model_db import ModelDBController
from cancer_ai.validator.utils import get_competition_weights

# add type hotkey which is string
Hotkey = str

HISTORY_LENGTH = 10

# how many results should we use for calculating average score
MOVING_AVERAGE_LENGTH = 5


class ModelScore(BaseModel):
    date: datetime
    score: float


class CompetitionResultsStore(BaseModel):
    # Structure: {competition_id: {hotkey: [ModelScore, ...]}}
    score_map: dict[str, dict[Hotkey, list[ModelScore]]] = {}
    # Structure: {competition_id: {hotkey: average_score}}
    average_scores: dict[str, dict[Hotkey, float]] = {}
    # Structure: {competition_id: (hotkey, score)}
    current_top_hotkeys: dict[str, tuple[Hotkey, float]] = {}

    def add_score(self, competition_id: str, hotkey: Hotkey, score: float, date: datetime = None):
        """Add a score for a specific hotkey in a specific competition."""

        if competition_id not in self.score_map:
            self.score_map[competition_id] = {}
        if competition_id not in self.average_scores:
            self.average_scores[competition_id] = {}

        if hotkey not in self.score_map[competition_id]:
            self.score_map[competition_id][hotkey] = []

        score_date = date if date is not None else datetime.now(timezone.utc)
        
        self.score_map[competition_id][hotkey].append(
            ModelScore(date=score_date, score=score)
        )

        # Sort by date and keep only the last HISTORY_LENGTH scores
        self.score_map[competition_id][hotkey].sort(key=lambda x: x.date)
        if len(self.score_map[competition_id][hotkey]) > HISTORY_LENGTH:
            # remove the oldest one
            self.score_map[competition_id][hotkey] = self.score_map[competition_id][hotkey][1:]

        self.update_average_score(competition_id, hotkey)

    def update_average_score(self, competition_id: str, hotkey: Hotkey) -> None:
        """Update the average score for a specific hotkey in a specific competition"""
        if (
            competition_id not in self.score_map
            or hotkey not in self.score_map[competition_id]
        ):
            return 0.0

        scores = self.score_map[competition_id][hotkey][-MOVING_AVERAGE_LENGTH:]
        scores = [score.score for score in scores]
        bt.logging.debug(f"Scores used to calculate average for hotkey {hotkey}: {scores}")
        try:
            result = sum(
                score
                for score in scores
            ) / len(scores)
        except ZeroDivisionError:
            result = 0.0

        if competition_id not in self.average_scores:
            self.average_scores[competition_id] = {}
        self.average_scores[competition_id][hotkey] = result

    def delete_dead_hotkeys(self, competition_id: str, active_hotkeys: list[Hotkey]):
        """Delete hotkeys that are no longer active in a specific competition."""
        if competition_id not in self.score_map:
            return

        hotkeys_to_delete = []
        for hotkey in self.score_map[competition_id].keys():
            if hotkey not in active_hotkeys:
                hotkeys_to_delete.append(hotkey)
        for hotkey in hotkeys_to_delete:
            del self.score_map[competition_id][hotkey]
            if (
                competition_id in self.average_scores
                and hotkey in self.average_scores[competition_id]
            ):
                del self.average_scores[competition_id][hotkey]

    def get_top_hotkey(self, competition_id: str) -> Hotkey:
        if (
            competition_id not in self.average_scores
            or not self.average_scores[competition_id]
        ):
            raise ValueError(
                f"No hotkeys to choose from for competition {competition_id}"
            )
        
        # Find the new top hotkey and score
        new_top_hotkey = max(
            self.average_scores[competition_id],
            key=self.average_scores[competition_id].get,
        )
        new_top_score = self.average_scores[competition_id][new_top_hotkey]
        
        # Check if we have a current top hotkey for this competition
        if competition_id in self.current_top_hotkeys:
            current_top_hotkey, current_top_score = self.current_top_hotkeys[competition_id]
            
            # If the current top hotkey is still active and the new top score
            # is not significantly better (within threshold), keep the current top hotkey
            if (
                current_top_hotkey in self.average_scores[competition_id] and
                abs(new_top_score - current_top_score) <= 0.0001
            ):
                return current_top_hotkey
        
        # Update the current top hotkey and score
        self.current_top_hotkeys[competition_id] = (new_top_hotkey, new_top_score)
        return new_top_hotkey

    def get_competitions(self) -> list[str]:
        return list(self.score_map.keys())
        
    def delete_inactive_competitions(self, active_competitions: list[str]):
        """Delete competitions that are no longer active."""
        competitions_to_delete = []
        for competition_id in self.score_map.keys():
            if competition_id not in active_competitions:
                competitions_to_delete.append(competition_id)
        
        for competition_id in competitions_to_delete:
            bt.logging.info(f"Deleting inactive competition {competition_id} from results store")
            del self.score_map[competition_id]
            if competition_id in self.average_scores:
                del self.average_scores[competition_id]
            if competition_id in self.current_top_hotkeys:
                del self.current_top_hotkeys[competition_id]

    async def update_competition_results(self, competition_id: str, model_results: list[tuple[str, ModelEvaluationResult]], config: bt.config, metagraph_hotkeys:list[Hotkey], hf_api):
        """Update competition results for a specific competition."""

        # Delete hotkeys from competition result score which don't exist anymore
        self.delete_dead_hotkeys(competition_id, metagraph_hotkeys)

        # Get competition weights from the config
        competition_weights = await get_competition_weights(config, hf_api)
        
        # Delete competitions that don't exist in the weights mapping
        self.delete_inactive_competitions(list(competition_weights.keys()))
        
        # Get all hotkeys that have models for this competition from the database
        latest_models = ModelDBController(db_path=config.db_path).get_latest_models(metagraph_hotkeys, competition_id)
        competition_miners = set(latest_models.keys())

        evaluated_miners = set()
        
        evaluation_timestamp = datetime.now(timezone.utc)

        for hotkey, result in model_results:
            self.add_score(competition_id, hotkey, result.score, date=evaluation_timestamp)
            evaluated_miners.add(hotkey)
        
        # Add score of 0 for miners who are in the competition but didn't take part in the evaluation
        # This is necessary to decrease their average score when their model fails or has errors
        failed_miners = competition_miners - evaluated_miners
        for hotkey in failed_miners:
            bt.logging.info(f"Adding score of 0 for hotkey {hotkey} in competition {competition_id} due to model failure or error")
            self.add_score(competition_id, hotkey, 0.0, date=evaluation_timestamp)

        # Get the winner hotkey for this competition
        try:
            winner_hotkey = self.get_top_hotkey(competition_id)
            bt.logging.info(f"Competition result for {competition_id}: {winner_hotkey}")
        except ValueError as e:
            bt.logging.warning(f"Could not determine winner for competition {competition_id}: {e}")
            winner_hotkey = None
        
        return competition_weights
