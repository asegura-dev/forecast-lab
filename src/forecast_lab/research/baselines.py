"""The numbers a model has to beat before anything it does counts as a result.

This module exists because of one measurement. The original analysis selected LightGBM
with PCA, reported **51.53% accuracy**, and concluded that machine learning can beat
chance on hourly gold. Its own classification report printed the class support two lines
below the accuracy: 1,547 of 2,983 test bars went up, so **predicting UP every single
time scores 51.86%**. The celebrated model lost to the most trivial rule there is, by
0.33 points, and nothing in the pipeline was arranged to notice.

Three baselines, each answering a different "well, obviously" objection:

- **Majority class**, fitted on train and applied blind. Not the test set's own majority
  - that is the trap the original fell into, and it flatters every model that inherits
  the same drift. The gap between the two is reported, because it *is* the trap.
- **Persistence**: tomorrow repeats today. The cheapest thing that uses the data at all,
  and the one a model claiming to find momentum must beat.
- **Random**, seeded, drawing from train's class frequencies. Establishes what the
  metric looks like when nothing is known, which is not always 50%.

Accuracy alone is not enough to see what a rule is doing - always-UP scores well and has
zero specificity - so precision, recall and specificity come with it. A model with the
same accuracy and a different error profile is a different model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_lab.contracts import ForecastLabError
from forecast_lab.research.labeling import Direction

#: Fixed so a random baseline is a fact about the data rather than about the day it ran.
DEFAULT_SEED = 20260101


class BaselineError(ForecastLabError):
    """A baseline cannot be computed from what was provided."""


@dataclass(frozen=True)
class Score:
    """How one rule did against one block of labels."""

    name: str
    n: int
    correct: int
    predicted_up: int

    # Confusion counts, kept because accuracy alone hides the shape of the errors.
    true_up: int
    false_up: int
    true_down: int
    false_down: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        """Of the bars called UP, how many rose."""
        called = self.true_up + self.false_up
        return self.true_up / called if called else 0.0

    @property
    def recall(self) -> float:
        """Of the bars that rose, how many were called."""
        actual = self.true_up + self.false_down
        return self.true_up / actual if actual else 0.0

    @property
    def specificity(self) -> float:
        """Of the bars that fell, how many were called.

        The number that exposes always-UP: perfect recall, zero specificity.
        """
        actual = self.true_down + self.false_up
        return self.true_down / actual if actual else 0.0


@dataclass(frozen=True)
class BaselineReport:
    """Every baseline against one block, plus the class balance behind them."""

    block: str
    n: int
    up_rate: float
    train_up_rate: float
    scores: tuple[Score, ...]

    @property
    def best(self) -> Score:
        return max(self.scores, key=lambda s: s.accuracy)

    @property
    def oracle_gap(self) -> float:
        """How much a rule gains purely by knowing this block's own class balance.

        The distance between the honest baseline and the trap. Where it is large, an
        accuracy figure says more about which stretch of history was used as test than
        about the model that produced it.
        """
        return abs(self.up_rate - self.train_up_rate)


def score(name: str, predicted: np.ndarray, actual: np.ndarray) -> Score:
    """Compare one rule's predictions against the truth."""
    if predicted.shape != actual.shape:
        raise BaselineError(f"{name}: {predicted.shape} predictions for {actual.shape} labels")

    up = Direction.UP.value
    pred_up, act_up = predicted == up, actual == up
    return Score(
        name=name,
        n=int(actual.size),
        correct=int((predicted == actual).sum()),
        predicted_up=int(pred_up.sum()),
        true_up=int((pred_up & act_up).sum()),
        false_up=int((pred_up & ~act_up).sum()),
        true_down=int((~pred_up & ~act_up).sum()),
        false_down=int((~pred_up & act_up).sum()),
    )


def evaluate_baselines(
    train_labels: pd.Series,
    block_labels: pd.Series,
    *,
    block: str,
    seed: int = DEFAULT_SEED,
) -> BaselineReport:
    """Score every baseline against ``block_labels``, fitting only on ``train_labels``.

    The separation is the whole point: a baseline that peeks at the block it is scored
    on is not a baseline, it is an oracle - and comparing a model against an oracle makes
    the model look worse, while comparing it against the block's own majority makes it
    look better than it is.
    """
    train = _decided(train_labels)
    actual = _decided(block_labels)
    if actual.size == 0:
        raise BaselineError(f"{block}: no decided labels to score against")

    train_up_rate = float((train == Direction.UP.value).mean())
    majority = Direction.UP.value if train_up_rate > 0.5 else Direction.DOWN.value

    rng = np.random.default_rng(seed)
    scores = (
        score(f"majority-class ({'UP' if majority == Direction.UP.value else 'DOWN'}, from train)",
              np.full(actual.size, majority), actual),
        score("persistence", _persistence(block_labels), actual),
        score(
            "random (train frequencies)",
            rng.choice(
                [Direction.DOWN.value, Direction.UP.value],
                size=actual.size,
                p=[1 - train_up_rate, train_up_rate],
            ),
            actual,
        ),
    )

    return BaselineReport(
        block=block,
        n=int(actual.size),
        up_rate=float((actual == Direction.UP.value).mean()),
        train_up_rate=train_up_rate,
        scores=scores,
    )


def _decided(labels: pd.Series) -> np.ndarray:
    """Only UP and DOWN. A FLAT bar has no direction to be right or wrong about."""
    values = labels.dropna().to_numpy(dtype="float64")
    keep: np.ndarray = values[(values == Direction.UP.value) | (values == Direction.DOWN.value)]
    return keep


def _persistence(labels: pd.Series) -> np.ndarray:
    """Predict that this bar repeats the previous one's direction.

    The first bar has no predecessor and is given DOWN, which is arbitrary and
    inconsequential over thousands of rows - but it is stated rather than hidden.
    """
    values = labels.dropna()
    decided = values[
        (values == Direction.UP.value) | (values == Direction.DOWN.value)
    ].to_numpy(dtype="float64")
    if decided.size == 0:
        return decided
    shifted: np.ndarray = np.concatenate([[Direction.DOWN.value], decided[:-1]])
    return shifted
