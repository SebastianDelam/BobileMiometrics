"""Simulate a typist change using the retained authentication models."""

import numpy as np

from matching import describe


def simulate(mode, test, model, setting, window):
    # Only held-out test data enter these streams.
    # A stream starts with genuine history, then receives an impostor's data.
    # One decision unit is a password entry or a free-text base window.
    horizon = min(36, min(map(len, test)))
    query_streams, genuine_starts = [], []
    count = len(test)
    if mode == "free":
        centers = np.median(model, axis=1)
        pooled = model.reshape(-1, *model.shape[2:]).std(axis=0)
        scales = np.maximum(model.std(axis=1), pooled*.25)
        if setting["scaling"] == "global":
            scales = np.broadcast_to(pooled, centers.shape)
        scales = np.maximum(scales, 1e-8)

    for owner in range(count):
        attackers = [identity for identity in range(count) if identity != owner]
        history = test[owner][-window:]
        incoming = np.stack([test[attacker][-horizon:] for attacker in attackers])
        owned = np.broadcast_to(history, (len(attackers), *history.shape))
        combined = np.concatenate((owned, incoming), axis=1)
        decisions = []

        # Step 0 checks the genuine user before the simulated takeover.
        # Subsequent decisions advance by one base window/password entry.
        for step in range(horizon + 1):
            raw = combined[:, step:step+window]
            if mode == "fixed":
                features = np.median(raw, axis=1)
                scores = model.score_identity(features, owner)
            else:
                described = describe(raw.reshape(-1, 2, 9), "free")
                features = described.reshape(len(attackers), window, 2, -1).mean(axis=1)
                distance = np.abs((features-centers[owner])/scales[owner]).mean(axis=-1)
                scores = (1/(1+distance)).mean(axis=-1)
            decisions.append(scores)
        genuine_starts.extend(decisions[0])
        query_streams.append(np.stack(decisions[1:], axis=1))
    return np.concatenate(query_streams), np.array(genuine_starts), horizon


def summarize(scores, initial_scores, threshold, horizon):
    # Count a takeover only if its genuine starting state was accepted.
    # Report excluded starting states so false alarms are visible.
    accepted = initial_scores >= threshold
    rejected = scores[accepted] < threshold
    detected = rejected.any(axis=1)
    delay = rejected[detected].argmax(axis=1) + 1
    return dict(
        total_streams=len(scores),
        initially_rejected_streams=int((~accepted).sum()),
        eligible_takeovers=int(accepted.sum()),
        detected=int(detected.sum()),
        missed=int((~detected).sum()),
        horizon_decision_units=horizon,
        median_units_if_detected=float(np.median(delay)) if len(delay) else "",
        p90_units_if_detected=float(np.quantile(delay, .9)) if len(delay) else "",
    )
