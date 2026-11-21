"""Performance plots adapted from the HW3 evaluator."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM_THRESHOLDS = 500


class Evaluator:
    def __init__(self, num_thresholds, genuine_scores, impostor_scores,
                 plot_title, epsilon=1e-12, output_dir=None):
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = np.asarray(genuine_scores)
        self.impostor_scores = np.asarray(impostor_scores)
        self.plot_title = plot_title
        self.epsilon = epsilon
        self.output_dir = Path(output_dir or Path(__file__).parent / "plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_dprime(self):
        difference = abs(np.mean(self.genuine_scores) - np.mean(self.impostor_scores))
        pooled_sd = np.sqrt((np.var(self.genuine_scores) + np.var(self.impostor_scores)) / 2)
        return float(difference / (pooled_sd + self.epsilon))

    def compute_rates(self):
        # Similarity scores: accept a comparison when score >= threshold.
        FPR, FNR, TPR = [], [], []
        for threshold in self.thresholds:
            TP = np.count_nonzero(self.genuine_scores >= threshold)
            FN = len(self.genuine_scores) - TP
            FP = np.count_nonzero(self.impostor_scores >= threshold)
            TN = len(self.impostor_scores) - FP
            FPR.append(FP / (FP + TN))
            FNR.append(FN / (FN + TP))
            TPR.append(TP / (TP + FN))
        return np.array(FPR), np.array(FNR), np.array(TPR)

    def get_EER(self, FPR, FNR):
        # Use the sampled threshold where the two rates are closest.
        index = np.argmin(np.abs(FPR - FNR))
        eer = (FPR[index] + FNR[index]) / 2
        return float(eer), float(self.thresholds[index])

    def _finish(self, fig, ax, name):
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{name}_{self.plot_title}.png", dpi=300)
        plt.close(fig)

    def plot_score_distribution(self):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        bins = np.linspace(0, 1, 41)
        ax.hist(self.genuine_scores, bins=bins, density=True, color="green",
                histtype="step", linewidth=1.7, label="Genuine")
        ax.hist(self.impostor_scores, bins=bins, density=True, color="red",
                histtype="step", linewidth=1.7, label="Impostor")
        ax.set(xlim=(-0.05, 1.05), xlabel="Similarity score", ylabel="Probability density",
               title=f"System {self.plot_title}: score distributions\nd-prime = {self.get_dprime():.5f}")
        ax.legend(loc="upper left")
        self._finish(fig, ax, "score_distribution")

    def plot_det_curve(self, FPR, FNR):
        eer, threshold = self.get_EER(FPR, FNR)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot(FPR, FNR, color="black", linewidth=1.7)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, label="FPR = FNR")
        index = np.argmin(np.abs(FPR - FNR))
        ax.scatter([FPR[index]], [FNR[index]], color="black", zorder=3, label="Closest error rates")
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel="False Positive Rate (FPR)",
               ylabel="False Negative Rate (FNR)",
               title=f"System {self.plot_title}: DET curve\nEER = {eer:.5f} at threshold = {threshold:.5f}")
        ax.legend()
        self._finish(fig, ax, "det")

    def plot_roc_curve(self, FPR, TPR):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot(FPR, TPR, color="black", linewidth=1.7)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, label="Chance reference")
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel="False Positive Rate (FPR)",
               ylabel="True Positive Rate (TPR)", title=f"System {self.plot_title}: ROC curve")
        ax.legend(loc="lower right")
        self._finish(fig, ax, "roc")
