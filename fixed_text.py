"""Fixed-text feature fusion, PCA and per-user Naive Bayes verification."""

import csv
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB


def load_features(path, audit):
    """Keep all ten hold times and nine press-to-press intervals per entry."""
    grouped = defaultdict(list)
    invalid = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != 50 or header[48] != "DU.b.b":
            raise ValueError("The fixed-text CSV has an unexpected format.")
        for row in reader:
            try:
                hold = [float(row[3 + 5 * i]) for i in range(9)] + [float(row[48])]
                interval = [float(row[4 + 5 * i]) for i in range(9)]
                values = np.array(hold + interval)
                repetition = int(row[2])
            except (ValueError, IndexError):
                invalid += 1
                continue
            if not np.isfinite(values).all() or np.any(values < 0):
                invalid += 1
                continue
            grouped[row[0], row[1]].append((repetition, values))
    result = {}
    for identity, records in grouped.items():
        numbers = [number for number, _ in records]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"Duplicate password repetition for {identity}.")
        result[identity] = np.stack([values for _, values in sorted(records)])
    audit["fixed_features"] = {
        "invalid_entries": invalid,
        "valid_entries": sum(map(len, result.values())),
    }
    return result


class FixedMatcher:
    """Fit one genuine-versus-impostor classifier for each enrolled user."""

    def fit(self, enrollment, block=3, components=18):
        # enrollment has dimensions [user, password entry, 19 timing features].
        # Each group averages three consecutive password entries.
        # Groups overlap, but they all stay inside the enrollment portion.
        groups = [
            enrollment[:, start:start + block].mean(axis=1)
            for start in range(enrollment.shape[1] - block + 1)
        ]
        training = np.stack(groups, axis=1)
        labels = np.repeat(np.arange(len(enrollment)), training.shape[1])
        training = training.reshape(-1, training.shape[-1])

        # Lecture 6 feature fusion concatenates hold times and intervals.
        # Lecture 8 PCA maps those 19 features onto 18 enrollment-learned axes.
        self.pca = PCA(n_components=components, svd_solver="full")
        training = self.pca.fit_transform(training)
        self.models = []
        for identity in range(len(enrollment)):
            # 1 means this user. 0 means every other enrolled user.
            binary_labels = (labels == identity).astype(int)
            model = GaussianNB(priors=[.5, .5], var_smoothing=1e-9)
            model.fit(training, binary_labels)
            self.models.append(model)
        return self

    def scores(self, queries):
        # queries has dimensions [actual user, observation, 19 features].
        shape = queries.shape
        projected = self.pca.transform(queries.reshape(-1, shape[-1]))
        probabilities = np.stack([
            model.predict_proba(projected)[:, 1] for model in self.models
        ], axis=1)
        # Return [actual user, claimed user, observation], like the free matcher.
        return probabilities.reshape(shape[0], shape[1], -1).transpose(0, 2, 1)

    def score_identity(self, features, identity):
        """Score a batch against one claimed identity during a takeover."""
        projected = self.pca.transform(features)
        return self.models[identity].predict_proba(projected)[:, 1]
