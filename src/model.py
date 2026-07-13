"""Aesthetic judgment model: Algorithm A (activation-difference basis).

Each song is encoded as a pyactup chunk with slots {song_id, complexity}.
"Aesthetic basis" = |activation(predicted chunk) - activation(actual song chunk)|,
per literature-notes/main-project-idea.txt.
"""

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

    def learn_song(self, song_id, complexity, advance=True):
        chunk = self.memory.learn({"song_id": song_id, "complexity": complexity})
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
