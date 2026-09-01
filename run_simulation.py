"""Loads config -> runs the aesthetic memory model over a song sequence -> saves data.

Aesthetic basis (Algorithm A) is only computed on a song's repeat exposures,
using memory as it stood before that trial's re-encoding, since it depends on
the song already having a chunk of its own.
"""

import csv
import glob
import math
import os
import random

import pandas as pd
import yaml

from src.controller import ChangeAwareUCB, SlidingWindowUCB
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


def run_design6(controller="c1", num_songs=100, num_exposures=10000, window=20, seed=42,
                 time_average_mode="cumulative", time_average_window=20, gamma=1.0,
                 bandit_window=50, bandit_c=2.0, r_change_threshold=0.15,
                 noise=None, decay=None,
                 config_path="config.yaml",
                 output_path=None):
    """Series of Experiments, Design 6 (literature-notes/main-project-idea.txt):
    Algorithm C control study. Same random ergodic song pool as Designs 1/4/5,
    same Algorithm A4 basis and {song_id, evaluation} chunk schema as Design 5,
    but the environment's uniform i.i.d. song draw is replaced by a bandit
    controller choosing which song to present next, trying to maximize
    `evaluation`.

    controller="c1": Algorithm C1 -- SlidingWindowUCB, reward = observed
    `evaluation` only, no access to the memory agent's internal state.

    controller="c2": Algorithm C2 -- same bandit, plus privileged access to
    the memory agent's live gamma-scaled time-averaged aesthetic basis `r`
    (the target peak is at aesthetic_basis = r/2). When `r` shifts by more
    than `r_change_threshold` (relative) between trials, the bandit's reward
    history is reset, letting it re-adapt immediately instead of waiting for
    stale rewards to age out of its sliding window.

    Same double-learning caveat as Design 4/5: evaluate_a4() learns its own
    chunk internally, so this does NOT separately call learn_evaluation() on
    repeat exposures. A song's bootstrap play (learn_evaluation(song_id, 0.0))
    is also reported to the bandit as a reward of 0.0, marking the arm as
    played so select()'s forced-exploration pass doesn't pick it forever.

    noise/decay override the corresponding pyactup Memory parameters from
    config_path when not None, so they can be set directly from a notebook."""
    if output_path is None:
        output_path = f"results/data/design6_{controller}_timeseries.csv"

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

    if controller == "c1":
        bandit = SlidingWindowUCB(song_ids, window=bandit_window, c=bandit_c, rng=rng)
    elif controller == "c2":
        bandit = ChangeAwareUCB(song_ids, window=bandit_window, c=bandit_c, rng=rng,
                                 r_change_threshold=r_change_threshold)
    else:
        raise ValueError(f"unknown controller: {controller!r}")

    seen = set()
    rows = []
    last_r = None
    for trial in range(num_exposures):
        if controller == "c2":
            bandit.observe_r(last_r)
        song_id = bandit.select()
        complexity = complexities[song_id]
        if song_id not in seen:
            model.learn_evaluation(song_id, 0.0)  # first-exposure bootstrap only
            bandit.update(song_id, 0.0)
        else:
            row = model.evaluate_a4(song_id)  # computes AND learns its own chunk
            if row["evaluation"] is not None:
                bandit.update(song_id, row["evaluation"])
                last_r = row["r"]
                row["trial"] = trial
                row["complexity"] = complexity
                rows.append(row)
        seen.add(song_id)

    df = pd.DataFrame(rows)
    df["evaluation_rolling_mean"] = df["evaluation"].rolling(window).mean()
    df.to_csv(output_path, index=False)
    return df


def load_session_from_log(csv_path, session_id):
    """Reads just one session's rows out of one day's raw log_0_YYYYMMDD_*.csv
    file (Spotify Sequential Skip Prediction Challenge format -- WSDM Cup
    2019/AIcrowd -- columns: session_id, session_position, session_length,
    track_id_clean, skip_1/2/3, not_skipped, hour_of_day, date, ...), without
    loading the whole multi-hundred-MB file into memory. Sessions are stored
    as contiguous row blocks in these files (confirmed empirically against
    this data), so a chunked scan can stop as soon as a session's block ends.

    Returns rows sorted by session_position, with track_id_clean renamed to
    song_id (the model's own vocabulary), plus:
      - hour_timestamp: date + hour_of_day, for time_mode="elapsed" -- note
        this dataset has no finer per-play timestamp than hour-of-day, so
        within-hour gaps read as 0; it's a coarse approximation, not true
        elapsed time between plays.
      - real_skip: Mark's binary skip coding (confirmed 2026-08-27) derived
        from skip_1/skip_2/skip_3/not_skipped -- see add_real_skip()."""
    chunks = []
    found = False
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        matched = chunk[chunk["session_id"] == session_id]
        if len(matched):
            chunks.append(matched)
            found = True
        elif found:
            break  # past the session's contiguous block
    if not chunks:
        raise ValueError(f"session_id {session_id!r} not found in {csv_path}")
    df = pd.concat(chunks).sort_values("session_position").reset_index(drop=True)
    df = df.rename(columns={"track_id_clean": "song_id"})
    df["hour_timestamp"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour_of_day"], unit="h")
    df = add_real_skip(df)
    return df


def add_real_skip(df):
    """Derives real_skip, Mark's binary skip coding (confirmed 2026-08-27),
    from the raw skip_1/skip_2/skip_3/not_skipped columns:
      - skip_1 or skip_2 True -> real_skip = 1 (an early/mid skip)
      - otherwise, skip_3 or not_skipped True -> real_skip = 0 (a late
        partial-skip, or a full listen)
      - otherwise (all four columns False, ~1.7% of rows empirically --
        confirmed with Mark this is a real, not-rare category, not just an
        edge case) -> real_skip = NaN, meant to be excluded from analysis
        rather than silently coded either way.
    skip_1/2/3 nest almost perfectly in this data (skip_1 implies skip_2
    implies skip_3), so the first two conditions overlap on roughly half of
    all rows (wherever skip_2 is True) -- skip_1-or-skip_2 wins by design in
    that overlap, per Mark's explicit tie-break choice."""
    df = df.copy()
    df["real_skip"] = float("nan")
    df.loc[df["skip_3"] | df["not_skipped"], "real_skip"] = 0
    df.loc[df["skip_1"] | df["skip_2"], "real_skip"] = 1
    return df


def run_real_session(session_df, time_mode="tick", window=20, min_elapsed_hours=1e-6,
                      time_average_mode="cumulative", time_average_window=20, gamma=1.0,
                      noise=None, decay=None,
                      config_path="config.yaml",
                      output_path="results/data/real_session_timeseries.csv",
                      model=None, seen=None):
    """Real-data validation study: feeds one real Spotify session (see
    load_session_from_log()), in session_position order, through Algorithm
    A4 (same algorithm as Design 5) in place of Design 5's synthetic random
    ergodic environment. All four of skip_1/skip_2/skip_3/not_skipped ride
    along on each output row for later comparison against the model's own
    computed `evaluation` only -- confirmed with Mark this is a VALIDATION
    study (real behavior compared against the model's output), not a
    supervision study (real behavior is never learned into memory). Which of
    the four columns operationalizes "skipped" is left for analysis time,
    not decided here.

    Analysis unit is one session at a time (confirmed with Mark) -- this
    dataset has no person-identifier linking sessions to the same listener,
    so cross-session pooling isn't meaningful here the way it would be for
    a real per-person history.

    time_mode="tick": advance() by 1 per exposure, same convention as
    Designs 1-6.
    time_mode="elapsed": advance() by real elapsed hours between plays
    (from `hour_timestamp`) instead of a uniform tick -- see
    load_session_from_log()'s docstring for this dataset's hour-level (not
    per-play) time resolution caveat. As in the tick case, this must land
    BEFORE the trial's retrieval, not after -- evaluate_a4() only advances
    as its own final step, so advancing by the upcoming gap ahead of time is
    what keeps a trial's retrieval computed at the right point in decay,
    rather than one gap "too fresh". The very first exposure has no prior
    timestamp to diff against, so nothing advances before it either way
    (nothing is in memory yet, so this doesn't affect anything).

    min_elapsed_hours floors every nonzero-or-later elapsed gap to at least
    this value (default a negligible fraction of an hour): consecutive plays
    sharing the same hour_of_day bucket produce a delta of exactly 0, and
    pyactup's base-level activation formula divides by
    (now - creation_time)**-decay -- a retrieval at literally the same time
    a chunk was created is a divide-by-zero, not just an approximation
    error. Confirmed empirically against this dataset (same-hour repeats are
    common given only hour-level time resolution).

    model/seen let a caller hand in an already-primed model and its
    seen-song set instead of starting fresh (e.g.
    run_real_session_with_burn_in priming the model before this function's
    loop ever runs) -- when omitted (the default), behavior is unchanged: a
    fresh model and empty seen set, as before."""
    if time_mode not in ("tick", "elapsed"):
        raise ValueError(f"time_mode must be 'tick' or 'elapsed' (got {time_mode!r})")

    if model is None:
        config = load_config(config_path)
        if noise is not None:
            config["noise"] = noise
        if decay is not None:
            config["decay"] = decay
        model = AestheticMemoryModel(
            **config, time_average_mode=time_average_mode, time_average_window=time_average_window,
            gamma=gamma)
    if seen is None:
        seen = set()

    rows = []
    prev_timestamp = None
    first_position = {}
    for trial, record in enumerate(session_df.itertuples(index=False)):
        song_id = record.song_id
        if time_mode == "elapsed":
            if prev_timestamp is not None:
                delta_hours = (record.hour_timestamp - prev_timestamp).total_seconds() / 3600
                model.memory.advance(max(delta_hours, min_elapsed_hours))
            advance = 0
        else:
            advance = 1
        prev_timestamp = record.hour_timestamp

        if song_id in seen:
            row = model.evaluate_a4(song_id, advance=advance)  # computes AND learns its own chunk
            row["trial"] = trial
            row["session_position"] = record.session_position
            # None when the song's real first occurrence happened outside session_df entirely
            # (e.g. during a burn-in priming pass, which doesn't feed through this loop/dict).
            row["first_occurrence_position"] = first_position.get(song_id)
            if row["first_occurrence_position"] is not None:
                row["repeat_gap"] = record.session_position - row["first_occurrence_position"]
            else:
                row["repeat_gap"] = None
            row["skip_1"] = record.skip_1
            row["skip_2"] = record.skip_2
            row["skip_3"] = record.skip_3
            row["not_skipped"] = record.not_skipped
            row["real_skip"] = record.real_skip
            rows.append(row)
        else:
            model.learn_evaluation(song_id, 0.0, advance=advance)  # first-exposure bootstrap only
            first_position[song_id] = record.session_position
        seen.add(song_id)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["evaluation_rolling_mean"] = df["evaluation"].rolling(window).mean()
    if output_path is not None:
        df.to_csv(output_path, index=False)
    return df


FAKE_SONG_CORPUS = [f"fake_{i:03d}" for i in range(1, 21)]
"""Shared, fixed 20-song corpus of fabricated (never-really-heard) song_ids
used by run_real_session_with_burn_in's burn-in phase 1 -- confirmed with
Mark (2026-08-28) as a fixed corpus reused identically across every
session, rather than freshly generated per session, so the "background"
context is constant across the whole study."""


def _expose_sequence(model, song_ids, seen, advance=1):
    """Feeds a plain sequence of song_ids through model (mutating seen and
    model state) -- the same seen/evaluate-or-bootstrap logic as
    run_real_session's main loop (evaluate_a4 whenever the song already has
    a chunk, contributing to the running time_averaged_aesthetic_basis;
    a placeholder bootstrap learn otherwise), but for burn-in exposures
    with no real skip data and no output collected -- burn-in's whole point
    is that its own results are discarded, only the model state it leaves
    behind matters."""
    for song_id in song_ids:
        if song_id in seen:
            model.evaluate_a4(song_id, advance=advance)
        else:
            model.learn_evaluation(song_id, 0.0, advance=advance)
        seen.add(song_id)


def run_real_session_with_burn_in(session_df, rng, phase0_repeats=3, phase1=True, window=20,
                                   time_average_mode="cumulative", time_average_window=20, gamma=1.0,
                                   noise=None, decay=None, config_path="config.yaml", output_path=None):
    """Primes a fresh model with a burn-in procedure (confirmed with Mark,
    2026-08-28, replacing the two-pass technique entirely) before playing
    session_df's actual sequence once -- this is the mechanism that lets a
    no-repeat session produce real evaluation values on every trial
    (evaluate_a4 requires a song to already have its own chunk -- confirmed
    directly: calling it on a never-before-seen song raises TypeError,
    since song_activation()'s retrieve() finds nothing -- which burn-in
    phase 0 guarantees for every song in the session before the real pass
    even starts) while also maturing the running
    time_averaged_aesthetic_basis well before the real pass, so (unlike the
    old single-pass/two-pass approaches) no forced-evaluation==0 artifact
    should land on the real pass at all -- confirmed empirically, not just
    assumed; see the memory notes for the check.

    Burn-in phase 0: the session's own unique song_ids (its "song set") are
    exposed phase0_repeats times each (default 3), via phase0_repeats
    independent random permutations (no replacement within a permutation)
    drawn from rng. Each song's 2nd and later exposures are genuine repeats
    -- evaluate_a4 runs and updates the running average -- while its 1st
    exposure (whichever permutation it falls in first) is a bootstrap.
    NOTE: with phase0_repeats=1 (and phase1=False), no song ever repeats
    during burn-in at all -- every phase-0 exposure is a first-time
    bootstrap, so the model's very first-ever evaluate_a4 call happens on
    the real pass's first evaluable row, same forced-evaluation==0 artifact
    as the old single-pass/two-pass approaches (see drop_first_evaluable_trial).
    With phase0_repeats>=2 or phase1=True, burn-in itself absorbs that
    artifact before the real pass starts, as usual.

    Burn-in phase 1 (skip with phase1=False): 100 exposures sampled with
    replacement from FAKE_SONG_CORPUS (a fixed, shared 20-song fabricated
    corpus -- see its own docstring), drawn from the same rng. This
    represents generic "background" listening experience competing for the
    unconstrained predicted_chunk() slot, rather than leaving the model's
    whole world be just the session's own tiny song set. Confirmed with
    Mark (2026-08-29): phase1=False isolates phase 0's effect alone, to
    check whether phase 1's fake-corpus reinforcement is what diluted the
    evaluation-vs-real_skip signal found under the full (phase 0 + 1)
    burn-in relative to the earlier two-pass/single-pass results.

    Real pass: session_df's actual rows, once, in session_position order --
    via run_real_session(..., model=model, seen=seen), reusing the already
    -primed model/seen state. This is the only data returned/kept.

    rng is a random.Random instance the caller owns and threads across
    sessions (matching this project's existing seeding convention) so a
    whole batch run is reproducible from one seed while each session still
    gets distinct burn-in randomization.

    time_mode is always "tick" here (not exposed as a parameter) -- burn-in
    exposures have no real timestamps, so "elapsed" has no meaning across
    them.

    Returns a DataFrame (session_id NOT added -- callers insert it, as in
    run_real_sessions_batch) and optionally writes it to output_path."""
    config = load_config(config_path)
    if noise is not None:
        config["noise"] = noise
    if decay is not None:
        config["decay"] = decay
    model = AestheticMemoryModel(
        **config, time_average_mode=time_average_mode, time_average_window=time_average_window,
        gamma=gamma)
    seen = set()

    song_set = list(session_df["song_id"].unique())
    for _ in range(phase0_repeats):
        _expose_sequence(model, rng.sample(song_set, len(song_set)), seen)

    if phase1:
        fake_sample = rng.choices(FAKE_SONG_CORPUS, k=100)
        _expose_sequence(model, fake_sample, seen)

    return run_real_session(
        session_df, time_mode="tick", window=window,
        time_average_mode=time_average_mode, time_average_window=time_average_window, gamma=gamma,
        noise=noise, decay=decay, config_path=config_path, output_path=output_path,
        model=model, seen=seen)


def run_real_sessions_burn_in_sample(data_dir="data_all", num_files=5, sessions_per_file=1000, seed=42,
                                      sessions_source="qualifying", phase0_repeats=3, phase1=True,
                                      session_length=20,
                                      min_repeat_position=11, min_gap=3, window=20,
                                      time_average_mode="cumulative", time_average_window=20, gamma=1.0,
                                      noise=None, decay=None, config_path="config.yaml",
                                      output_path=None):
    """Burn-in version of run_real_sessions_sample (see
    run_real_session_with_burn_in's docstring for the burn-in procedure).
    sessions_source picks which session population to draw from --
    "qualifying" (qualifying_sessions: repeat at position >=
    min_repeat_position with a gap >= min_gap) or "no_repeat"
    (no_repeat_sessions: zero in-session repeats) -- so the same function
    builds both sides of the comparison (2026-08-28: "we should do this for
    both repeat and non-repeat sessions"). phase1=False skips burn-in phase
    1 (the fake-corpus exposures) entirely -- confirmed with Mark
    (2026-08-29) as an ablation to isolate phase 1's effect on the
    evaluation-vs-real_skip signal. phase0_repeats controls how many times
    each song in phase 0 gets exposed (default 3, matching the original
    design) -- see run_real_session_with_burn_in's docstring for the
    phase0_repeats=1 special case (no burn-in repeat at all, so the
    forced-evaluation==0 artifact resurfaces on the real pass's first row).

    Picks num_files files evenly spaced across data_dir (see
    _evenly_spaced_valid_files), then a random sample of sessions_per_file
    sessions from that population in each file. One random.Random(seed) is
    built here and threaded across every session in the run (matching this
    project's existing seeding convention), so burn-in randomization
    differs session to session but the whole run is reproducible from one
    seed.

    Returns one combined DataFrame across all sampled files (adds
    source_file and session_id columns) and writes it to output_path
    (defaults to a name reflecting sessions_source and phase1 if not
    given)."""
    if output_path is None:
        suffix = "" if phase1 else "_no_phase1"
        if phase0_repeats != 3:
            suffix += f"_phase0x{phase0_repeats}"
        output_path = f"results/data/real_sessions_burn_in_{sessions_source}{suffix}_sample.csv"

    chosen_files = _evenly_spaced_valid_files(data_dir, num_files)
    rng = random.Random(seed)

    results = []
    for csv_path in chosen_files:
        df = pd.read_csv(csv_path)
        if sessions_source == "qualifying":
            session_ids = qualifying_sessions(df, session_length=session_length,
                                               min_repeat_position=min_repeat_position, min_gap=min_gap)
        elif sessions_source == "no_repeat":
            session_ids = no_repeat_sessions(df, session_length=session_length)
        else:
            raise ValueError(f"unknown sessions_source: {sessions_source!r}")

        session_ids = pd.Series(session_ids).sample(n=min(sessions_per_file, len(session_ids)),
                                                      random_state=seed).values

        file_results = []
        for session_id in session_ids:
            session_df = _prepare_session_df(df, session_id)
            result = run_real_session_with_burn_in(
                session_df, rng, phase0_repeats=phase0_repeats, phase1=phase1, window=window,
                time_average_mode=time_average_mode, time_average_window=time_average_window,
                gamma=gamma, noise=noise, decay=decay, config_path=config_path, output_path=None)
            if not result.empty:
                result.insert(0, "session_id", session_id)
                file_results.append(result)
        if file_results:
            file_combined = pd.concat(file_results, ignore_index=True)
            file_combined.insert(0, "source_file", os.path.basename(csv_path))
            results.append(file_combined)

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    combined.to_csv(output_path, index=False)
    return combined


def _prepare_session_df(df, session_id):
    """Builds one session's model-ready DataFrame (song_id rename,
    hour_timestamp, real_skip) from an already-loaded multi-session log
    DataFrame -- the common prep step shared by run_real_sessions_batch and
    run_real_sessions_burn_in_sample."""
    session_df = df[df["session_id"] == session_id].sort_values("session_position").reset_index(drop=True)
    session_df = session_df.rename(columns={"track_id_clean": "song_id"})
    session_df["hour_timestamp"] = (
        pd.to_datetime(session_df["date"]) + pd.to_timedelta(session_df["hour_of_day"], unit="h"))
    return add_real_skip(session_df)


def sessions_with_repeats(df):
    """session_ids (from an already-loaded, possibly multi-session log
    DataFrame) that have at least one in-session repeated track_id_clean --
    the only sessions run_real_session() can ever produce an evaluable row
    from (Algorithm A/A2/A3/A4 only compute anything on a repeat exposure).
    Superseded as the default session filter by qualifying_sessions() (see
    its docstring) but kept as the more permissive building block."""
    has_repeat = df.groupby("session_id")["track_id_clean"].apply(lambda s: s.duplicated().any())
    return has_repeat[has_repeat].index


def qualifying_sessions(df, session_length=20, min_repeat_position=11, min_gap=3):
    """session_ids (from an already-loaded, possibly multi-session log
    DataFrame) meeting Mark's three-stage filter (confirmed 2026-08-28) for
    the real-data validation study:
      1. session_length == 20 (full-length sessions only)
      2. has a repeated track_id_clean whose repeat occurs at
         session_position >= min_repeat_position (11 by default -- i.e.
         strictly after the 10th song)
      3. that repeat is separated from the track's first play by
         min_gap=3 or more positions (repeat_position - first_position >= 3
         -- i.e. at least 2 other tracks actually sit between them)

    A session qualifies if ANY repeated track satisfies both (2) and (3) --
    confirmed with Mark this is a session-inclusion gate only: a qualifying
    session is then run through run_real_session() in full as usual, so
    every repeat exposure in it (qualifying pair or not) still produces an
    evaluable row, not just the pair that triggered inclusion."""
    candidates = df[df["session_length"] == session_length]
    if candidates.empty:
        return pd.Index([], name="session_id")

    def session_qualifies(positions_by_track):
        for positions in positions_by_track:
            if len(positions) < 2:
                continue
            positions = sorted(positions)
            first = positions[0]
            if any(p >= min_repeat_position and (p - first) >= min_gap for p in positions[1:]):
                return True
        return False

    grouped = candidates.groupby(["session_id", "track_id_clean"])["session_position"].apply(list)
    qualifies = grouped.groupby("session_id").apply(session_qualifies)
    return qualifies[qualifies].index


def run_real_sessions_batch(csv_path, output_path, max_sessions=None, sample_n=None, seed=42,
                             session_length=20, min_repeat_position=11, min_gap=3,
                             time_mode="tick", window=20,
                             time_average_mode="cumulative", time_average_window=20, gamma=1.0,
                             noise=None, decay=None, config_path="config.yaml"):
    """Runs run_real_session() independently over every session in one day's
    log file that qualifies per qualifying_sessions() (session_length==20,
    a repeat at position >= min_repeat_position with a gap >= min_gap from
    its first play -- confirmed with Mark, see that function's docstring).
    Each session still gets its own fresh AestheticMemoryModel -- no state
    carries between sessions, same as calling run_real_session() by hand
    per session (confirmed earlier: this dataset has no person-identifier
    to justify pooling across sessions).

    Reads the whole file once (~3.4M rows/day, ~2s to load) rather than
    re-scanning per session the way load_session_from_log() does for a
    single one-off session -- much cheaper when processing many sessions
    out of the same file.

    Measured cost is ~112ms/session (mostly pyactup call overhead), so a
    full day's ~100k+ repeat-having sessions is a multi-hour undertaking --
    max_sessions caps how many qualifying sessions get processed (first N,
    unrandomized -- for timing/sanity checks) and sample_n instead draws a
    random sample of that size (seeded by `seed`) -- confirmed with Mark:
    default real-data runs should sample rather than process exhaustively.
    Only one of max_sessions/sample_n should be set.

    Returns one combined DataFrame (adds a session_id column identifying
    which session each row came from) and also writes it to output_path."""
    df = pd.read_csv(csv_path)
    session_ids = qualifying_sessions(df, session_length=session_length,
                                       min_repeat_position=min_repeat_position, min_gap=min_gap)
    if sample_n is not None:
        session_ids = pd.Series(session_ids).sample(n=min(sample_n, len(session_ids)),
                                                      random_state=seed).values
    elif max_sessions is not None:
        session_ids = session_ids[:max_sessions]

    results = []
    for session_id in session_ids:
        session_df = _prepare_session_df(df, session_id)

        result = run_real_session(
            session_df, time_mode=time_mode, window=window,
            time_average_mode=time_average_mode, time_average_window=time_average_window,
            gamma=gamma, noise=noise, decay=decay, config_path=config_path,
            output_path=None)  # aggregate write only, not one file per session
        if not result.empty:
            result.insert(0, "session_id", session_id)
            results.append(result)

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    combined.to_csv(output_path, index=False)
    return combined


def _is_valid_log_file(csv_path):
    """Cheap corruption check: reads just the first 64KB and confirms it
    starts with the expected header and contains a newline. Found 4 of 60
    files in data_all/ (20180913/14/16/17) that are entirely null bytes
    throughout (confirmed by sampling multiple offsets, not just truncated
    at the start) -- a real data-integrity issue distinct from the
    Clean_data_*.RData corruption found earlier, so this filters both kinds
    of bad file out rather than letting them silently produce zero rows or
    crash a multi-file run partway through."""
    with open(csv_path, "rb") as f:
        chunk = f.read(65536)
    return chunk.startswith(b"session_id") and b"\n" in chunk


def _evenly_spaced_valid_files(data_dir, num_files):
    """num_files log_0_*.csv files evenly spaced across the full sorted
    (chronological-by-filename) list of valid files in data_dir, skipping
    corrupted ones (see _is_valid_log_file). Shared by run_real_sessions_sample
    and sample_no_repeat_sessions_multi so both draw from the same file
    selection logic for comparable multi-file samples."""
    all_files = [f for f in sorted(glob.glob(f"{data_dir}/log_0_*.csv")) if _is_valid_log_file(f)]
    if num_files > len(all_files):
        raise ValueError(f"num_files={num_files} exceeds {len(all_files)} valid files found in {data_dir}")
    indices = [round(i * (len(all_files) - 1) / (num_files - 1)) for i in range(num_files)] \
        if num_files > 1 else [0]
    return [all_files[i] for i in sorted(set(indices))]


def run_real_sessions_sample(data_dir="data_all", num_files=5, sessions_per_file=1000, seed=42,
                              output_path="results/data/real_sessions_sample_timeseries.csv",
                              **kwargs):
    """Draws a real-data sample spread across several days rather than one
    file or an exhaustive run (confirmed with Mark: ~112ms/session makes a
    full day's ~100k+ repeat-having sessions a multi-hour undertaking, all
    65 files multi-day -- sampling is the intended default).

    Picks num_files files evenly spaced across data_dir (see
    _evenly_spaced_valid_files), then a random sample of sessions_per_file
    qualifying sessions from each (via run_real_sessions_batch's
    sample_n/seed) for broader date coverage than a single file alone.
    **kwargs forwards to run_real_sessions_batch (time_mode, noise, decay,
    gamma, session_length, min_repeat_position, min_gap, etc.).

    Returns one combined DataFrame across all sampled files (adds a
    source_file column) and writes it to output_path."""
    chosen_files = _evenly_spaced_valid_files(data_dir, num_files)

    results = []
    for csv_path in chosen_files:
        result = run_real_sessions_batch(
            csv_path, output_path=None, sample_n=sessions_per_file, seed=seed, **kwargs)
        if not result.empty:
            result.insert(0, "source_file", os.path.basename(csv_path))
            results.append(result)

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    combined.to_csv(output_path, index=False)
    return combined


def no_repeat_sessions(df, session_length=20):
    """session_ids (from an already-loaded, possibly multi-session log
    DataFrame) with session_length == session_length and NO in-session
    repeated track_id_clean at all.

    This is the control population for checking a confound Mark spotted
    (2026-08-28) in qualifying_sessions()'s repeat-based filter: since a
    qualifying session's evaluated (repeat) rows are, by construction,
    always at session_position >= min_repeat_position (late in the
    session), any evaluation-vs-real_skip relationship found among them
    could just be a generic "skip rate varies with position in the
    session" effect, confounded with the repeat requirement, rather than a
    real memory-based signal. Comparing skip rate by session_position in
    this repeat-free population (where no evaluation is even possible)
    against the same in the qualifying/evaluated population isolates that."""
    candidates = df[df["session_length"] == session_length]
    counts = candidates.groupby("session_id").agg(
        n_rows=("track_id_clean", "size"), n_unique=("track_id_clean", "nunique"))
    clean = counts[counts["n_rows"] == counts["n_unique"]]
    return clean.index


def sample_no_repeat_sessions(csv_path, sample_n=None, seed=42, session_length=20):
    """All rows (with real_skip added) for a random sample of sample_n
    no-repeat, session_length==20 sessions from one log file -- see
    no_repeat_sessions()'s docstring. No model is run here (there's nothing
    to evaluate without a repeat), so this is cheap compared to
    run_real_sessions_batch -- every row of every sampled session is kept,
    not just repeat-exposure rows."""
    df = pd.read_csv(csv_path)
    session_ids = no_repeat_sessions(df, session_length=session_length)
    if sample_n is not None:
        session_ids = pd.Series(session_ids).sample(n=min(sample_n, len(session_ids)),
                                                      random_state=seed).values
    result = df[df["session_id"].isin(session_ids)].copy()
    return add_real_skip(result)


def sample_no_repeat_sessions_multi(data_dir="data_all", num_files=5, sessions_per_file=2000, seed=42,
                                     session_length=20,
                                     output_path="results/data/no_repeat_sessions_sample.csv"):
    """Multi-file version of sample_no_repeat_sessions, mirroring
    run_real_sessions_sample's file-selection logic (same
    _evenly_spaced_valid_files call, so both draws can use the same files
    for a fair comparison when num_files/data_dir match). sessions_per_file
    defaults higher than run_real_sessions_sample's (2000 vs 1000) since
    this is much cheaper per session (no model run) -- a tighter control
    estimate is nearly free here.

    Returns one combined DataFrame (adds source_file) and writes it to
    output_path."""
    chosen_files = _evenly_spaced_valid_files(data_dir, num_files)

    results = []
    for csv_path in chosen_files:
        result = sample_no_repeat_sessions(
            csv_path, sample_n=sessions_per_file, seed=seed, session_length=session_length)
        if not result.empty:
            result.insert(0, "source_file", os.path.basename(csv_path))
            results.append(result)

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    combined.to_csv(output_path, index=False)
    return combined


def drop_first_evaluable_trial(df):
    """Drops each session's first evaluable trial from a multi-session
    output DataFrame (one with a session_id column, e.g. from
    run_real_sessions_batch/run_real_sessions_sample) before analysis.

    On a fresh AestheticMemoryModel, Algorithm A4's running
    time_averaged_aesthetic_basis has only one data point on its very first
    update, which forces aesthetic_basis == r exactly on that trial --
    plugging x == r into the parabola always gives evaluation == 0,
    regardless of what actually happened (confirmed empirically: 100%
    correspondence between evaluation==0 rows and each session's first
    evaluable trial). Since every real-data session gets its own fresh
    model, this hits every session once -- confirmed with Mark (2026-08-28):
    exclude these from analysis (they carry no real signal) while still
    running the full session through the model as usual -- this is a
    post-hoc analysis filter, not a change to run_real_session's own
    output, which is left untouched.

    NOT needed for run_real_sessions_burn_in_sample's output: burn-in
    matures the running average well before the real pass starts, so this
    artifact doesn't occur there (confirmed empirically: zero exactly-0.0
    evaluations in burn-in results, unlike single-pass results)."""
    df = df.sort_values(["session_id", "trial"])
    return df[df["session_id"].duplicated(keep="first")]


if __name__ == "__main__":
    df = run_design1()
    print(df.tail())
