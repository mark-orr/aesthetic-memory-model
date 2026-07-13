"""Loads config -> runs the aesthetic memory model over a song sequence -> saves data.

Aesthetic basis (Algorithm A) is only computed on a song's repeat exposures,
using memory as it stood before that trial's re-encoding, since it depends on
the song already having a chunk of its own.
"""

import csv
import random

import pandas as pd
import yaml

from src.model import AestheticMemoryModel

# Toy listening sequence: (song_id, complexity). Replace with real stimulus data.
SONG_SEQUENCE = [
    ("A", 3), ("B", 7), ("C", 5),
    ("A", 3), ("D", 8), ("B", 7),
    ("C", 5), ("A", 3), ("E", 2), ("B", 7),
]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run(song_sequence=SONG_SEQUENCE, config_path="config.yaml",
        output_path="results/data/aesthetic_basis.csv"):
    config = load_config(config_path)
    model = AestheticMemoryModel(**config)

    seen = set()
    rows = []
    for trial, (song_id, complexity) in enumerate(song_sequence):
        if song_id in seen:
            row = model.aesthetic_basis(song_id)
            row["trial"] = trial
            row["complexity"] = complexity
            rows.append(row)
        seen.add(song_id)
        model.learn_song(song_id, complexity)

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


def make_ergodic_environment(num_songs=100, complexity_low=1, complexity_high=10, seed=42):
    """A stationary environment: num_songs songs, each with a fixed complexity
    drawn once. Uniform random draws with replacement from this fixed pool
    make the exposure process i.i.d. (hence ergodic)."""
    rng = random.Random(seed)
    song_ids = [f"song_{i:03d}" for i in range(num_songs)]
    complexities = {sid: rng.randint(complexity_low, complexity_high) for sid in song_ids}
    return song_ids, complexities


def run_design1(num_songs=100, num_exposures=10000, window=20, seed=42,
                 config_path="config.yaml",
                 output_path="results/data/design1_ergodic_timeseries.csv"):
    """Series of Experiments, Design 1 (literature-notes/main-project-idea.txt):
    aesthetic_basis as a time series under a random, ergodic environment,
    summarized by a rolling time-average over `window` exposures."""
    config = load_config(config_path)
    model = AestheticMemoryModel(**config)
    rng = random.Random(seed)
    song_ids, complexities = make_ergodic_environment(num_songs, seed=seed)

    seen = set()
    rows = []
    for trial in range(num_exposures):
        song_id = rng.choice(song_ids)
        complexity = complexities[song_id]
        if song_id in seen:
            row = model.aesthetic_basis(song_id)
            row["trial"] = trial
            row["complexity"] = complexity
            rows.append(row)
        seen.add(song_id)
        model.learn_song(song_id, complexity)

    df = pd.DataFrame(rows)
    df["aesthetic_basis_rolling_mean"] = df["aesthetic_basis"].rolling(window).mean()
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = run_design1()
    print(df.tail())
