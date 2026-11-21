import csv
from collections import defaultdict
import numpy as np

KEYS = 10
ENROLLMENT_WINDOWS = (2, 5, 10, 20)
EVALUATION_WINDOWS = 10
TARGET_EER = 0.05


def features(holds, intervals):
    return [list(holds), list(intervals)]


def load_windows(path, mode, audit):
    """Read timing windows and record excluded data for the report."""
    sessions = defaultdict(list)
    counts = defaultdict(int)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.reader(stream)
        header = [name.strip() for name in next(rows)]
        if mode == "fixed":
            hold_columns = [i for i, name in enumerate(header)
                            if name.startswith("DU.") and name.split(".")[1] == name.split(".")[2]]
            interval_columns = [i for i, name in enumerate(header) if name.startswith("DD.")]
            if (len(hold_columns), len(interval_columns)) != (KEYS, KEYS - 1):
                raise ValueError("Unexpected fixed-text timing columns")
            repetitions = defaultdict(list)
            for row in rows:
                counts["rows"] += 1
                if row:
                    values = np.asarray([float(row[i]) for i in hold_columns + interval_columns])
                    if not np.isfinite(values).all() or np.any(values < 0):
                        counts["invalid_rows"] += 1
                        continue
                    repetitions[row[0], row[1]].append(
                        (int(row[2]), features([float(row[i]) for i in hold_columns[:-1]],
                                               [float(row[i]) for i in interval_columns]))
                    )
            for group, records in repetitions.items():
                sessions[group] = [values for _, values in sorted(records)]
        else:
            segment = []
            previous_group = None
            previous_key = None
            for row in rows:
                counts["rows"] += 1
                if not row:
                    continue
                group = (row[0], row[1])
                if group != previous_group or row[2] != previous_key:
                    counts["discarded_partial_pairs"] += len(segment)
                    counts["sequence_breaks"] += 1
                    segment.clear()
                try:
                    hold, interval = float(row[4]), float(row[5])
                except ValueError:
                    counts["invalid_rows"] += 1
                    counts["discarded_partial_pairs"] += len(segment)
                    segment.clear()
                    previous_group, previous_key = None, None
                    continue
                if not (np.isfinite(hold) and np.isfinite(interval) and hold >= 0 and interval >= 0):
                    counts["invalid_rows"] += 1
                    counts["discarded_partial_pairs"] += len(segment)
                    segment.clear()
                    previous_group, previous_key = None, None
                    continue
                segment.append((hold, interval))
                if len(segment) == KEYS - 1:
                    timings = np.asarray(segment)
                    sessions[group].append(features(timings[:, 0], timings[:, 1]))
                elif len(segment) == KEYS:
                    segment.clear()  # Skip the pair that bridges two windows.
                previous_group, previous_key = group, row[3]
            counts["discarded_partial_pairs"] += min(len(segment), KEYS - 2)
    counts["windows"] = sum(map(len, sessions.values()))
    audit[mode] = dict(counts)
    return {group: np.asarray(windows) for group, windows in sessions.items()}


def write_results(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


