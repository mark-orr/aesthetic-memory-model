"""Bandit controllers for Algorithm C (literature-notes/main-project-idea.txt).

SlidingWindowUCB implements Algorithm C1: it only ever sees the `evaluation`
reward returned for whichever song it plays. ChangeAwareUCB implements
Algorithm C2: same bandit, but with privileged access to the memory agent's
live time-averaged aesthetic basis `r`, used to detect when the inverted-U
target has moved and reset its statistics accordingly.
"""

import math
from collections import deque


class SlidingWindowUCB:
    """Non-stationary multi-armed bandit (Garivier & Moulines sliding-window
    UCB). Reward distributions per arm are assumed to drift over time, so only
    the last `window` rewards for an arm count towards its estimate."""

    def __init__(self, arms, window=50, c=2.0, rng=None):
        self.arms = list(arms)
        self.window = window
        self.c = c
        self.rng = rng
        self.history = {arm: deque(maxlen=window) for arm in self.arms}
        self.t = 0

    def select(self):
        self.t += 1
        unplayed = [arm for arm in self.arms if len(self.history[arm]) == 0]
        if unplayed:
            return self.rng.choice(unplayed) if self.rng is not None else unplayed[0]

        scores = {}
        for arm in self.arms:
            rewards = self.history[arm]
            n = len(rewards)
            mean = sum(rewards) / n
            bonus = self.c * math.sqrt(math.log(min(self.t, self.window)) / n)
            scores[arm] = mean + bonus
        return max(scores, key=scores.get)

    def update(self, arm, reward):
        self.history[arm].append(reward)


class ChangeAwareUCB(SlidingWindowUCB):
    """SlidingWindowUCB with an explicit change-point reset: when the caller
    reports a new `r` (Algorithm A4's gamma-scaled time-averaged aesthetic
    basis) that differs from the last reported `r` by more than
    `r_change_threshold` (relative), every arm's reward history is cleared,
    forcing immediate re-exploration instead of waiting for stale rewards to
    age out of the sliding window on their own."""

    def __init__(self, arms, window=50, c=2.0, rng=None, r_change_threshold=0.15):
        super().__init__(arms, window=window, c=c, rng=rng)
        self.r_change_threshold = r_change_threshold
        self.last_r = None

    def observe_r(self, r):
        if r is not None and self.last_r is not None:
            relative_change = abs(r - self.last_r) / max(abs(self.last_r), 1e-9)
            if relative_change > self.r_change_threshold:
                for arm in self.arms:
                    self.history[arm].clear()
        self.last_r = r
