"""Run RQ1, RQ2 and RQ3 on the original KeyRecs CSV files."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from data import load_windows, write_results
from fixed_text import FixedMatcher, load_features
from matching import combine_observations, describe, evaluate, match
from performance import Evaluator, NUM_THRESHOLDS
from takeovers import simulate, summarize


def split_recording(recording):
    # Reserve the first 20 base windows/password entries for enrollment.
    # Validation and test use separate halves of everything that follows.
    middle = 20 + (len(recording) - 20) // 2
    return recording[:20], recording[20:middle], recording[middle:]


def free_experiment(data, users):
    """RQ1 compares enrollment amounts and free-text observation lengths."""
    trials = []
    for budget, block, scaling in product((5, 10, 20), (1, 2, 5), ("global", "user")):
        enrollment, validation, testing = [], [], []
        for user in users:
            reserved, val, test = split_recording(data[user, "2"])
            enrollment.append(describe(reserved[:budget], "free"))
            validation.append(combine_observations(describe(val[-2*block:], "free"), block, "mean"))
            testing.append(combine_observations(describe(test[-2*block:], "free"), block, "mean"))
        enrollment = np.array(enrollment)
        scores = match(enrollment, np.array(validation), scaling)
        eer, threshold = evaluate(scores)
        setting = dict(enrollment_key_events=budget*10, observation_key_events=block*10,
                       scaling=scaling, validation_eer=eer, validation_threshold=threshold)
        trials.append((setting, enrollment, np.array(testing)))
    setting, enrollment, queries = min(trials, key=lambda trial: trial[0]["validation_eer"])
    scores = match(enrollment, queries, setting["scaling"])
    return setting, scores, enrollment, [trial[0] for trial in trials]


def fixed_experiment(data, users):
    """RQ2 compares enrollment amounts with the retained fixed-text method."""
    trials = []
    for budget in (5, 10, 20):
        enrollment, validation, testing = [], [], []
        for user in users:
            reserved, val, test = split_recording(data[user, "2"])
            enrollment.append(reserved[:budget])
            # One fixed-text observation summarizes 36 password entries.
            validation.append(np.median(val[-36:], axis=0))
            testing.append(np.median(test[-36:], axis=0))
        matcher = FixedMatcher().fit(np.array(enrollment))
        eer, threshold = evaluate(matcher.scores(np.array(validation)[:, None]))
        setting = dict(enrollment_key_events=budget*10, observation_key_events=360,
                       scaling="PCA and GaussianNB", validation_eer=eer,
                       validation_threshold=threshold)
        trials.append((setting, matcher, np.array(testing)[:, None]))
    setting, matcher, testing = min(trials, key=lambda trial: trial[0]["validation_eer"])
    return setting, matcher.scores(testing), matcher, [trial[0] for trial in trials]


def save_evaluation(mode, setting, scores, output):
    # The diagonal compares each query with its actual identity.
    # Off-diagonal scores represent claims to other users' identities.
    genuine = np.eye(len(scores), dtype=bool)
    evaluator = Evaluator(NUM_THRESHOLDS, scores[genuine].ravel(),
                          scores[~genuine].ravel(), mode, output_dir=output)
    fpr, fnr, tpr = evaluator.compute_rates()
    eer, _ = evaluator.get_EER(fpr, fnr)
    evaluator.plot_score_distribution()
    evaluator.plot_det_curve(fpr, fnr)
    evaluator.plot_roc_curve(fpr, tpr)
    threshold = setting["validation_threshold"]
    return dict(mode=mode, users=len(scores), **setting, test_eer=eer,
                dprime=evaluator.get_dprime(),
                genuine_test_queries=int(genuine.sum()*scores.shape[-1]),
                test_frr=float(np.mean(scores[genuine] < threshold)),
                test_far=float(np.mean(scores[~genuine] >= threshold)),
                test_eer_below_5_percent=eer < .05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {}
    free = load_windows(args.data_dir / "free-text.csv", "free", audit)
    fixed = load_features(args.data_dir / "fixed-text.csv", audit)

    # Retain the same eligibility rule used in the development experiments.
    # Selecting users never uses their matching scores or classification errors.
    users = sorted({user for user, session in free if session == "1"})
    users = [user for user in users if all(
        len(data.get((user, "1"), [])) >= 20 and len(data.get((user, "2"), [])) >= 20
        for data in (free, fixed)
    )]
    if len(users) < 2:
        raise ValueError("At least two users need enough data in both CSV files.")
    if any(len(split_recording(fixed[user, "2"])[1]) < 36 or
           len(split_recording(fixed[user, "2"])[2]) < 36 for user in users):
        raise ValueError("The retained fixed-text setting needs 36 validation and 36 test entries.")

    settings_free, scores_free, enrollment, trials_free = free_experiment(free, users)
    settings_fixed, scores_fixed, matcher, trials_fixed = fixed_experiment(fixed, users)
    results = [
        save_evaluation("free", settings_free, scores_free, args.output_dir),
        save_evaluation("fixed", settings_fixed, scores_fixed, args.output_dir),
    ]
    write_results(args.output_dir / "rq1_validation.csv", trials_free)
    write_results(args.output_dir / "rq2_validation.csv", trials_fixed)
    write_results(args.output_dir / "summary.csv", results)

    # RQ3 uses these same selected models and validation thresholds.
    takeover_rows = []
    for mode, data, setting, model in (
            ("free", free, settings_free, enrollment),
            ("fixed", fixed, settings_fixed, matcher)):
        test = [split_recording(data[user, "2"])[2] for user in users]
        if mode == "free":
            window = setting["observation_key_events"] // 10
        else:
            window = 36
        streams, starts, horizon = simulate(mode, test, model, setting, window)
        for increase in (0, .025, .05, .1):
            threshold = min(1., setting["validation_threshold"] + increase)
            reference_scores = scores_free if mode == "free" else scores_fixed
            mask = np.eye(len(users), dtype=bool)
            takeover_rows.append(dict(
                mode=mode, threshold_increase=increase, threshold=threshold,
                genuine_query_frr=float(np.mean(reference_scores[mask] < threshold)),
                **summarize(streams, starts, threshold, horizon)
            ))
    write_results(args.output_dir / "rq3_takeovers.csv", takeover_rows)
    audit["users"] = users
    audit["protocol"] = "within_session"
    audit["fixed_representation"] = "10 hold times and 9 press-to-press intervals"
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2))
    for result in results:
        print(f"{result['mode']}: validation EER {result['validation_eer']:.2%}, "
              f"test EER {result['test_eer']:.2%}")


if __name__ == "__main__":
    main()
