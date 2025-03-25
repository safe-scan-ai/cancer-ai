from pydantic import BaseModel
from datetime import datetime, timezone
import bittensor as bt


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

    def add_score(self, competition_id: str, hotkey: Hotkey, score: float):
        """Add a score for a specific hotkey in a specific competition."""

        # Initialize competition dictionaries if they don't exist
        if competition_id not in self.score_map:
            self.score_map[competition_id] = {}
        if competition_id not in self.average_scores:
            self.average_scores[competition_id] = {}

        # Initialize hotkey list if it doesn't exist
        if hotkey not in self.score_map[competition_id]:
            self.score_map[competition_id][hotkey] = []

        # Add the score
        self.score_map[competition_id][hotkey].append(
            ModelScore(date=datetime.now(timezone.utc), score=score)
        )

        # Sort by date and keep only the last HISTORY_LENGTH scores
        self.score_map[competition_id][hotkey].sort(key=lambda x: x.date, reverse=True)
        if len(self.score_map[competition_id][hotkey]) > HISTORY_LENGTH:
            self.score_map[competition_id][hotkey] = self.score_map[competition_id][
                hotkey
            ][-HISTORY_LENGTH:]

        # Update the average score
        self.update_average_score(competition_id, hotkey)

    def update_average_score(self, competition_id: str, hotkey: Hotkey):
        """Update the average score for a specific hotkey in a specific competition."""
        if (
            competition_id not in self.score_map
            or hotkey not in self.score_map[competition_id]
        ):
            return 0.0

        try:
            result = sum(
                score.score
                for score in self.score_map[competition_id][hotkey][
                    -MOVING_AVERAGE_LENGTH:
                ]
            ) / len(self.score_map[competition_id][hotkey][-MOVING_AVERAGE_LENGTH:])
        except ZeroDivisionError:
            result = 0.0

        if competition_id not in self.average_scores:
            self.average_scores[competition_id] = {}
        self.average_scores[competition_id][hotkey] = result
        return result

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

        return max(
            self.average_scores[competition_id],
            key=self.average_scores[competition_id].get,
        )

    def get_competitions(self) -> list[str]:
        return list(self.score_map.keys())
