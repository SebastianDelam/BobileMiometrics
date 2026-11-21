"""Free-text features and matching scores."""
import numpy as np
from performance import Evaluator, NUM_THRESHOLDS

def describe(raw, mode):
    """Keep password positions, or summarize free-text timing distributions."""
    if mode == "fixed":
        return raw
    percentiles = np.quantile(raw, (.1, .25, .5, .75, .9), axis=-1)
    # Put the five summary values in the last dimension.
    return np.moveaxis(percentiles, 0, -1)


def combine_observations(values, block, aggregation):
    grouped = values.reshape(-1, block, *values.shape[1:])
    if aggregation == "median":
        return np.median(grouped, axis=1)
    return grouped.mean(axis=1)


def match(enrollment, queries, scaling, hold_weight=.5, distance_rule="absolute"):
    # Input dimensions: user, observation, timing family, feature.
    # The two timing families are hold time and time between key presses.
    template = np.median(enrollment, axis=1)
    global_scale = enrollment.reshape(-1, *enrollment.shape[2:]).std(axis=0)
    if scaling == "user":
        # A variance floor prevents an almost constant feature from dominating.
        scale = np.maximum(enrollment.std(axis=1), global_scale * .25)
    else:
        scale = global_scale
    scale = np.maximum(scale, 1e-8)
    # Compare every query with every claimed user's enrollment template.
    # Result dimensions: actual user, claimed user, observation, family, feature.
    differences = queries[:, None] - template[None, :, None]
    if scaling == "user":
        differences = differences / scale[None, :, None]
    else:
        differences = differences / scale
    if distance_rule == "euclidean":
        distance = np.sqrt(np.mean(differences ** 2, axis=-1))
    else:
        distance = np.abs(differences).mean(axis=-1)
    family_scores = 1 / (1 + distance)
    # Weighted score fusion: hold weight plus the complementary interval weight.
    return hold_weight * family_scores[..., 0] + (1-hold_weight) * family_scores[..., 1]


def evaluate(scores):
    # Matching row and column IDs are genuine attempts; other pairs are impostors.
    mask = np.eye(len(scores), dtype=bool)
    evaluator = Evaluator(NUM_THRESHOLDS, scores[mask].ravel(),
                          scores[~mask].ravel(), "comparison")
    fpr, fnr, _ = evaluator.compute_rates()
    eer, threshold = evaluator.get_EER(fpr, fnr)
    return eer, threshold


