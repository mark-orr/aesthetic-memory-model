"""Loads config -> runs the aesthetic memory model over a song sequence -> saves data.

Aesthetic basis (Algorithm A) is only computed on a song's repeat exposures,
using memory as it stood before that trial's re-encoding, since it depends on
the song already having a chunk of its own.
"""

import csv
import math
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
                 noise=None, decay=None,
                 config_path="config.yaml",
                 output_path="results/data/design1_ergodic_timeseries.csv"):
    """Series of Experiments, Design 1 (literature-notes/main-project-idea.txt):
    aesthetic_basis as a time series under a random, ergodic environment,
    summarized by a rolling time-average over `window` exposures.

    noise/decay override the corresponding pyactup Memory parameters from
    config_path when not None, so they can be set directly from a notebook."""
    config = load_config(config_path)
    if noise is not None:
        config["noise"] = noise
    if decay is not None:
        config["decay"] = decay
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


def sweep_center(trial, num_exposures, num_cycles, sweep_type="rotate"):
    """The moving center mu(t) (radians) of the Design 2 probability distribution.

    "rotate": steady one-direction rotation, completing num_cycles full
    revolutions over the run (a spotlight continuously circling the songs).
    "oscillate": center swings back and forth between -pi and +pi instead of
    completing full loops in one direction.
    """
    phase = 2 * math.pi * num_cycles * trial / num_exposures
    if sweep_type == "rotate":
        return phase
    elif sweep_type == "oscillate":
        return math.pi * math.sin(phase)
    raise ValueError(f"unknown sweep_type: {sweep_type!r}")


def von_mises_weights(angles, mu, kappa):
    """Unnormalized von Mises density at each song's circle position, given
    the distribution's current center mu and concentration kappa."""
    return [math.exp(kappa * math.cos(a - mu)) for a in angles]


def run_design2(num_songs=100, num_exposures=10000, window=20, seed=42,
                 kappa=4.0, num_cycles=10, sweep_type="rotate",
                 noise=None, decay=None,
                 config_path="config.yaml",
                 output_path="results/data/design2_cyclic_timeseries.csv"):
    """Series of Experiments, Design 2 (literature-notes/main-project-idea.txt):
    same as Design 1, but songs are equally spaced on a circle and drawn from
    a von Mises distribution whose center sweeps cyclically around the circle,
    instead of Design 1's uniform i.i.d. draws.

    noise/decay override the corresponding pyactup Memory parameters from
    config_path when not None, so they can be set directly from a notebook."""
    config = load_config(config_path)
    if noise is not None:
        config["noise"] = noise
    if decay is not None:
        config["decay"] = decay
    model = AestheticMemoryModel(**config)
    rng = random.Random(seed)
    song_ids, complexities = make_ergodic_environment(num_songs, seed=seed)
    angles = [2 * math.pi * i / num_songs for i in range(num_songs)]

    seen = set()
    rows = []
    for trial in range(num_exposures):
        mu = sweep_center(trial, num_exposures, num_cycles, sweep_type)
        weights = von_mises_weights(angles, mu, kappa)
        song_id = rng.choices(song_ids, weights=weights, k=1)[0]
        complexity = complexities[song_id]
        if song_id in seen:
            row = model.aesthetic_basis(song_id)
            row["trial"] = trial
            row["complexity"] = complexity
            row["mu"] = mu
            rows.append(row)
        seen.add(song_id)
        model.learn_song(song_id, complexity)

    df = pd.DataFrame(rows)
    df["aesthetic_basis_rolling_mean"] = df["aesthetic_basis"].rolling(window).mean()
    df.to_csv(output_path, index=False)
    return df


def run_design3(num_songs=100, num_exposures=10000, window=20, seed=42,
                 kappa=4.0, num_cycles=10, sweep_type="rotate",
                 noise=None, decay=None,
                 config_path="config.yaml",
                 output_path="results/data/design3_evaluation_timeseries.csv"):
    """Series of Experiments, Design 3 (literature-notes/main-project-idea.txt):
    an expansion of Design 2 -- same cyclic von Mises environment -- but using
    Algorithm A2 instead of Algorithm A, and chunks encoded as
    {song_id, evaluation} instead of {song_id, complexity}. complexity is
    still tracked at the environment level (for e.g. inverted-U analysis)
    even though it's no longer stored in the memory chunk itself.

    A song's evaluation is undefined on its first exposure (no chunk yet) and
    second (no prior activation reading to diff against yet); 0.0 is encoded
    as a bootstrap placeholder for those two exposures.

    noise/decay override the corresponding pyactup Memory parameters from
    config_path when not None, so they can be set directly from a notebook."""
    config = load_config(config_path)
    if noise is not None:
        config["noise"] = noise
    if decay is not None:
        config["decay"] = decay
    model = AestheticMemoryModel(**config)
    rng = random.Random(seed)
    song_ids, complexities = make_ergodic_environment(num_songs, seed=seed)
    angles = [2 * math.pi * i / num_songs for i in range(num_songs)]

    seen = set()
    rows = []
    for trial in range(num_exposures):
        mu = sweep_center(trial, num_exposures, num_cycles, sweep_type)
        weights = von_mises_weights(angles, mu, kappa)
        song_id = rng.choices(song_ids, weights=weights, k=1)[0]
        complexity = complexities[song_id]
        if song_id in seen:
            row = model.evaluate_a2(song_id)
            row["trial"] = trial
            row["complexity"] = complexity
            row["mu"] = mu
            rows.append(row)
            evaluation_to_learn = row["evaluation"] if row["evaluation"] is not None else 0.0
        else:
            evaluation_to_learn = 0.0
        seen.add(song_id)
        model.learn_evaluation(song_id, evaluation_to_learn)

    df = pd.DataFrame(rows)
    df["evaluation_rolling_mean"] = df["evaluation"].rolling(window).mean()
    df.to_csv(output_path, index=False)
    return df


def run_design4(num_songs=100, num_exposures=10000, window=20, seed=42,
                 time_average_mode="cumulative", time_average_window=20,
                 noise=None, decay=None,
                 config_path="config.yaml",
                 output_path="results/data/design4_baseline_a3_timeseries.csv"):
    """Series of Experiments, Design 4 (literature-notes/main-project-idea.txt):
    a baseline study -- same random, ergodic environment as Design 1 (not
    Design 2/3's cyclic one) -- but using Algorithm A3 instead of Algorithm A,
    with chunks encoded as {song_id, evaluation} as in Design 3. Examines how
    both `evaluation` and Algorithm A3's own running `time_averaged_aesthetic_basis`
    vary over time.

    evaluate_a3() learns its own chunk internally as a final step (unlike
    evaluate_a2()), so -- unlike Designs 2/3's loop -- this does NOT also call
    learn_evaluation() after it on repeat exposures; that would double-learn.
    Only a song's first exposure gets an explicit bootstrap
    learn_evaluation(song_id, 0.0) call, same placeholder convention as
    Design 3.

    noise/decay override the corresponding pyactup Memory parameters from
    config_path when not None, so they can be set directly from a notebook."""
    config = load_config(config_path)
    if noise is not None:
        config["noise"] = noise
    if decay is not None:
        config["decay"] = decay
    model = AestheticMemoryModel(
        **config, time_average_mode=time_average_mode, time_average_window=time_average_window)
    rng = random.Random(seed)
    song_ids, complexities = make_ergodic_environment(num_songs, seed=seed)

    seen = set()
    rows = []
    for trial in range(num_exposures):
        song_id = rng.choice(song_ids)
        complexity = complexities[song_id]
        if song_id in seen:
            row = model.evaluate_a3(song_id)  # computes AND learns its own chunk
            row["trial"] = trial
            row["complexity"] = complexity
            rows.append(row)
        else:
            model.learn_evaluation(song_id, 0.0)  # first-exposure bootstrap only
        seen.add(song_id)

    df = pd.DataFrame(rows)
    df["evaluation_rolling_mean"] = df["evaluation"].rolling(window).mean()
    df.to_csv(output_path, index=False)
    return df


def run_design5(num_songs=100, num_exposures=10000, window=20, seed=42,
                 time_average_mode="cumulative", time_average_window=20, gamma=1.0,
                 noise=None, decay=None,
                 config_path="config.yaml",
                 output_path="results/data/design5_baseline_a4_timeseries.csv"):
    """Series of Experiments, Design 5 (literature-notes/main-project-idea.txt):
    identical to Design 4, but using Algorithm A4 instead of A3 -- same
    random ergodic environment, same {song_id, evaluation} chunk schema,
    same two goals (evaluation over time, time_averaged_aesthetic_basis over
    time), but r = gamma * time_averaged_aesthetic_basis. gamma=1.0 reduces
    exactly to Design 4/Algorithm A3.

    Same double-learning caveat as Design 4: evaluate_a4() learns its own
    chunk internally, so this does NOT separately call learn_evaluation() on
    repeat exposures.

    noise/decay override the corresponding pyactup Memory parameters from
    config_path when not None, so they can be set directly from a notebook."""
    config = load_config(config_path)
    if noise is not None:
        config["noise"] = noise
    if decay is not None:
        config["decay"] = decay
    model = AestheticMemoryModel(
        **config, time_average_mode=time_average_mode, time_average_window=time_average_window,
        gamma=gamma)
    rng = random.Random(seed)
    song_ids, complexities = make_ergodic_environment(num_songs, seed=seed)

    seen = set()
    rows = []
    for trial in range(num_exposures):
        song_id = rng.choice(song_ids)
        complexity = complexities[song_id]
        if song_id in seen:
            row = model.evaluate_a4(song_id)  # computes AND learns its own chunk
            row["trial"] = trial
            row["complexity"] = complexity
            rows.append(row)
        else:
            model.learn_evaluation(song_id, 0.0)  # first-exposure bootstrap only
        seen.add(song_id)

    df = pd.DataFrame(rows)
    df["evaluation_rolling_mean"] = df["evaluation"].rolling(window).mean()
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = run_design1()
    print(df.tail())
