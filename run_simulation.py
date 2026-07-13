"""Loads config -> runs the aesthetic memory model over a song sequence -> saves data.

Aesthetic basis (Algorithm A) is only computed on a song's repeat exposures,
using memory as it stood before that trial's re-encoding, since it depends on
the song already having a chunk of its own.
"""

import csv
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


if __name__ == "__main__":
    for row in run():
        print(row)
