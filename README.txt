MOBILE BIOMETRICS KEYRECS PROJECT

START HERE

main.py runs the three research questions. The other Python files are
imported by main.py, so keep them together in this folder.

Install the libraries:
python -m pip install -r requirements.txt

Run from PowerShell:
python main.py --data-dir "$env:USERPROFILE/Downloads"

Run with another data directory:
python main.py --data-dir "C:/path/to/data"

The data directory must contain the original free-text.csv and fixed-text.csv.
Data files are downloaded separately from https://zenodo.org/records/7886743
Results are written to the results folder. Override with --output-dir if needed.

WHAT EACH FILE DOES

main.py connects the experiments and saves tables and figures.
data.py reads free-text windows and writes CSV tables.
matching.py builds free-text features and matching scores.
fixed_text.py loads fixed-text features and trains PCA/Naive Bayes models.
takeovers.py simulates a change of typist for RQ3.
performance.py reuses the HW3 error-rate and plotting code.

HOW THE DATA IS SPLIT

Both text modes use the same eligible users. The supplied files give 98.
The first 20 base windows of free text, or 20 fixed password entries, from
session 2 are reserved for enrollment. The remainder is split into two
separate halves: validation and test. A smaller enrollment budget uses
only part of the reserved enrollment data; the unused part stays unused.
Enrollment, validation and test never overlap.

This is the within-session condition. It does not measure generalization
to a later recording session. Previous cross-session results are separate.
The final method and its PCA dimension were chosen during earlier exploration
of these same recording portions. Treat the results as exploratory.
An untouched recording session is needed for independent confirmation.

RQ1 FREE TEXT

Compare enrollment budgets of 50, 100 and 200 observed key events, observation
sizes of 10, 20 and 50 events, and global versus per-user timing scales.
The features summarize the distribution of hold times and press intervals.
The matcher compares each observation with a median enrollment template.
The final score averages the hold and interval similarities (lecture 6).
Two nonoverlapping observations per user are taken from each evaluation half.
Select the setting with the smallest validation EER, then evaluate it on test.

RQ2 FIXED TEXT

Use 19 timing features: 10 hold times plus 9 press-to-press intervals.
Their concatenation is feature-level fusion from lecture 6.
An enrollment example averages three successive password entries.
These training groups overlap, so they are dependent. They do not represent
additional collected data.
PCA retains 18 coefficients learned from enrollment (lecture 8).
A separate Gaussian Naive Bayes classifier checks each claimed identity,
using genuine versus other-user labels and equal binary priors (lecture 4,
and the GaussianNB implementation used in HW2).

Compare 5, 10 and 20 enrollment entries (50, 100 and 200 observed key events).
Select using validation EER. Each query summarizes 36 password entries by
their median, requiring 360 observed key events. There is one genuine query
per user in validation and test. The 98 genuine test queries limit precision.
The target of 5% EER was not confirmed on test in the development experiments.

The modes use different features, observation lengths and query counts.
Their performance difference cannot be attributed to text type alone.

RQ3 THRESHOLDS AND TAKEOVERS

Use exactly the selected RQ1/RQ2 models, without fitting on test data.
Compare the validation threshold and increases of 0.025, 0.05 and 0.10.
Higher thresholds reject more low-scoring comparisons.

A synthetic stream starts with genuine held-out history. At the takeover,
an impostor supplies new test observations. Keep a rolling history of the
selected observation length, and check after each incoming decision unit.
One fixed-text unit is a complete password entry. One free-text unit is a
base timing window. The common horizon is at most 36 units.

For each threshold, streams whose genuine starting history is already rejected
are reported separately. Only streams starting in an accepted state contribute
to takeover detection and miss counts. Also report genuine_query_frr, which
uses the held-out genuine queries. It is not a continuous-session false-alarm rate.

Delay units are observed password entries/base windows, not elapsed seconds.
Each corresponds approximately to ten observed key events. Invalid entries,
free-text bridge pairs and recording gaps prevent an exact physical-keypress
count. Fixed-text repetitions are concatenated without observed between-entry
pauses. These are simulated takeovers, not real recordings of a session change.
Overlapping decision windows are dependent.
Always report missed takeovers and initially rejected streams alongside
median/p90 delay. Conditional delay alone can hide failures.

RESULT FILES

rq1_validation.csv and rq2_validation.csv list compared enrollment settings.
summary.csv gives the selected settings, validation and test EER, d-prime,
and test FAR/FRR at the threshold chosen on validation.
rq3_takeovers.csv gives thresholds, false rejections and takeover outcomes.
data_audit.json gives excluded entries, the cohort and the protocol.
Six PNG files show ROC, DET and score distributions for both text modes.
An EER of 0.0530 means approximately 5.30%.
The HW3 code uses 500 sampled thresholds and averages FAR and FRR at their
closest sampled point, following the TA feedback on Homework 3.

Dataset DOI: https://doi.org/10.5281/zenodo.7886743
Dataset article: https://doi.org/10.1016/j.dib.2023.109509
