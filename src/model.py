"""Aesthetic judgment model: Algorithm A (activation-difference basis) and
Algorithm A2 (its extension, used by Design 3).

Algorithm A: each song is encoded as a pyactup chunk with slots
{song_id, complexity}. "Aesthetic basis" = |activation(predicted chunk) -
activation(actual song chunk)|.

Algorithm A2 (literature-notes/main-project-idea.txt): each song is instead
encoded as {song_id, evaluation}, where evaluation = g(y) * f(x):
  x = predicted_activation (Algorithm A step 1)
  f(x) = 1/e^x
  y = actual_activation_now - actual_activation_previous (signed, this song's
      own last two activation readings)
  g(y) = mirrored sigmoid = 1/(1+e^y)
Since a song's chunks now vary by evaluation value, a song accumulates
multiple distinct chunks over time (one per distinct evaluation encoded) --
each a separate IBL instance of experiencing that song, consistent with the
project's instance-based-learning premise. Retrieving by song_id alone still
returns whichever of that song's chunks currently has the highest activation.

evaluation is undefined for a song's first exposure (no chunk yet) and its
second (only one prior activation reading exists, no diff possible yet); the
caller decides what placeholder to encode during this bootstrap period.
"""

import math

import pyactup


class AestheticMemoryModel:
    def __init__(self, noise=0.25, decay=0.5, temperature=None,
                 threshold=None, mismatch=None, optimized_learning=False):
        self.memory = pyactup.Memory(
            noise=noise,
            decay=decay,
            temperature=temperature,
            threshold=threshold,
            mismatch=mismatch,
            optimized_learning=optimized_learning,
        )
        self._last_actual_activation = {}
        self._pinned_activations = {}
        self._adjustments = {}

    def learn_song(self, song_id, complexity, advance=True):
        chunk = self.memory.learn({"song_id": song_id, "complexity": complexity})
        if advance:
            self.memory.advance()
        return chunk

    def learn_evaluation(self, song_id, evaluation, advance=True):
        chunk = self.memory.learn({"song_id": song_id, "evaluation": evaluation})
        if advance:
            self.memory.advance()
        return chunk

    def _activation_of(self, slots):
        """Runs a retrieve() constrained by slots and reads the matching
        chunk's activation out of activation_history, since pyactup has no
        standalone activation-lookup method."""
        self.memory.activation_history = []
        chunk = self.memory.retrieve(slots)
        if chunk is None:
            return None, None
        chunk_items = set(chunk.items())
        for entry in self.memory.activation_history:
            if chunk_items <= set(entry["attributes"]):
                return chunk, entry["activation"]
        raise RuntimeError("No matching activation_history entry for retrieved chunk")

    def _base_level_activation(self, chunk):
        """Replicates pyactup's own base-level activation formula (the
        non-optimized-learning case) so a correction can be computed fresh at
        any point in time, rather than as a one-off snapshot. See pyactup.py's
        Memory._activations, the `self._optimized_learning is None` branch."""
        if self.memory.optimized_learning:
            raise NotImplementedError(
                "pin_activation doesn't support optimized_learning; the base-level "
                "formula differs and isn't replicated here.")
        now = self.memory.time
        decay = self.memory.decay
        refs = chunk.references[:chunk.reference_count]
        return math.log(sum((now - t) ** -decay for t in refs))

    def pin_activation(self, slots, target):
        """Forces every chunk matching `slots` (subset match, like
        memory.retrieve()) to report activation == target on every future
        retrieve()/blend(), by installing an extra_activation correction that
        is recomputed from the chunk's live base-level activation each time
        it's queried -- so the pin holds even as more time passes or the
        chunk gets reinforced again, not just as a one-time snapshot.

        Exact only when memory.noise == 0 (deterministic); with noise on, each
        query adds a fresh random draw this can't cancel out. Doesn't account
        for partial-matching/similarity mismatch penalties."""
        if self.memory.noise:
            raise ValueError(
                f"pin_activation requires memory.noise == 0 for exact results "
                f"(currently {self.memory.noise}).")
        if self.memory.mismatch is not None:
            raise NotImplementedError(
                "pin_activation doesn't account for partial-matching mismatch "
                "penalties; only supported with mismatch=None.")
        self._pinned_activations[frozenset(slots.items())] = target
        self._install_extra_activation()

    def unpin_activation(self, slots):
        """Removes a previously installed pin for the chunk(s) matching `slots`."""
        self._pinned_activations.pop(frozenset(slots.items()), None)

    def adjust_activation(self, slots, delta):
        """Sets a persistent additive offset on every chunk matching `slots`
        (subset match, like memory.retrieve()): its activation becomes
        natural_activation + delta on every future retrieve()/blend(), still
        tracking decay/reinforcement normally -- unlike pin_activation, which
        forces an exact value regardless of the chunk's natural dynamics.

        Calling this again on the same slots REPLACES the previous delta
        (not cumulative). Unlike pin_activation, this works under any
        noise/optimized_learning/mismatch setting, since it's a raw additive
        offset rather than a computed correction to an exact target."""
        self._adjustments[frozenset(slots.items())] = delta
        self._install_extra_activation()

    def clear_adjustment(self, slots):
        """Removes a previously set adjustment for the chunk(s) matching `slots`."""
        self._adjustments.pop(frozenset(slots.items()), None)

    def _install_extra_activation(self):
        self.memory.extra_activation = [self._extra_activation_callback]

    def _extra_activation_callback(self, chunk):
        """Pins take precedence over adjustments when both match the same
        chunk, since a pin's whole point is an exact, unconditional value."""
        chunk_items = set(chunk.items())
        for pin_slots, target in self._pinned_activations.items():
            if pin_slots <= chunk_items:
                return target - self._base_level_activation(chunk)
        for adj_slots, delta in self._adjustments.items():
            if adj_slots <= chunk_items:
                return delta
        return 0.0

    def predicted_chunk(self):
        """Step 1: highest-activation chunk in memory, unconstrained."""
        return self._activation_of({})

    def song_activation(self, song_id):
        """Step 2: activation of the actual current song's own chunk."""
        return self._activation_of({"song_id": song_id})

    def aesthetic_basis(self, song_id):
        """Steps 1-4: |predicted activation - actual song activation|."""
        predicted_chunk, predicted_activation = self.predicted_chunk()
        _, actual_activation = self.song_activation(song_id)
        return {
            "predicted_song_id": predicted_chunk["song_id"] if predicted_chunk else None,
            "predicted_activation": predicted_activation,
            "actual_song_id": song_id,
            "actual_activation": actual_activation,
            "aesthetic_basis": abs(predicted_activation - actual_activation),
        }

    def evaluate_a2(self, song_id):
        """Algorithm A2. Returns Algorithm A's fields plus x, f_x, y, g_y,
        evaluation. y/g_y/evaluation are None when this song doesn't yet have
        two prior actual_activation readings (bootstrap period)."""
        result = self.aesthetic_basis(song_id)
        x = result["predicted_activation"]
        f_x = math.exp(-x)
        actual_now = result["actual_activation"]

        previous = self._last_actual_activation.get(song_id)
        if previous is not None:
            y = actual_now - previous
            g_y = 1 / (1 + math.exp(y))
            evaluation = g_y * f_x
        else:
            y = None
            g_y = None
            evaluation = None
        self._last_actual_activation[song_id] = actual_now

        result.update({"x": x, "f_x": f_x, "y": y, "g_y": g_y, "evaluation": evaluation})
        return result
