# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Maryum Akram
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/MaryumAkram16/flyrank-ml-internship
- **Date:** 2026-08

## 0. Abstract

Which pages should a content team review first, given limited weekly review capacity? This
project builds and validates a content-prioritization pipeline on FlyRank's anonymized
search-performance dataset, using a hand-tuned baseline rule and two machine-learning models
(Logistic Regression, Random Forest) evaluated under a client-holdout split. The central,
counter-intuitive result: the simple baseline rule (Precision@50 = 0.90) outperformed Random
Forest (0.76) and matched Logistic Regression (0.92) — traced, via error analysis, to the
model over-weighting high-traffic pages rather than the specific CTR-vs-position signal the
baseline targeted directly. The output is a three-tier, human-reviewed action playbook
(monitor / review-for-refresh / flag-for-tracking-audit) with explicit no-go rules, intended
as decision-support for a content reviewer, not an automated system.

## 1. Problem framing

**Decision this supports:** a content strategist or SEO analyst with a fixed weekly review
budget (roughly 20–50 pages) needs a defensible, ranked shortlist of pages to review first,
rather than reviewing the full inventory or relying on gut feel.

**Unit of analysis:** one row = one page (`content_id`), for one client (`client_id`),
summarizing that page's trailing 90-day observed search signals.

**Output:** a ranked score (`baseline_action_score`) with a reason code and an action label.

**Action a human takes:** review the flagged page's specific reason code, decide on a
concrete fix (rewrite title/meta, verify tracking, or leave as-is), and act within the
team's normal editorial workflow.

**Cost of a wrong call:** a false positive wastes a reviewer's limited time; a false negative
lets a real problem page continue losing visibility unnoticed. Because review capacity is the
scarce resource, precision at the top of the ranked list (Precision@K) matters more than
catching every possible case.

**Why ML/data helps here:** the signals that separate underperforming pages from healthy ones
interact — a CTR gap means something different at different position tiers, and staleness
means something different at different traffic volumes. A fixed if/else rule can only combine
a small, hand-picked set of thresholds; a model can, in principle, discover these interactions
automatically. Whether it actually does so better than a well-targeted rule was tested
directly (see Section 5).

## 2. Data safety

**Dataset used:** the anonymized starter dataset (`data/raw/content_refresh_anonymized.csv`)
— 30,000 pages, 44 columns, one row per pseudonymized content item, filtered to 16,726 pages
after a 500-impression volume floor (55.8% retained).

**Also explored:** the full warehouse release (`FlyRank/internship-warehouse` on Hugging
Face), specifically `fact_content_daily_performance` for a mid-panel development month
(`month=2026-03`: 9,841,378 rows, 55 clients, 331,437 content items) — used to validate the
data contract, table grain, and field availability before modeling on the smaller starter
slice.

**Deliberately excluded:**
- Pages below the 500-impression volume floor — CTR is unreliable to measure on near-zero
  traffic; without this floor, top-ranked pages were dominated by 1–4 impression pages with a
  mechanical 0% CTR (noise, not signal).
- Any FlyRank product decision flags (`health_score`, `priority_score`, `action_type`) — never
  shipped in this data by design, and never reconstructed, to avoid circular results.
- Warehouse rows without `ga4_data_available == TRUE` when GA4-derived fields were considered
  — to avoid mistaking "not yet tracked" for "no traffic."

**Leakage risks considered:** `trend_direction` and `trend_pct` (the fields the label is
built from) were confirmed absent from every feature set used, in two independent checks (one
on warehouse data, one on the final starter-data feature set — see Section 5). Pseudonymous
IDs (`client_id`, `content_id`) were used only for grouping and joins, never as model features.
No client names, domains, URLs, or raw queries appear anywhere in this repository.

## 3. Baseline

**The rule:** `baseline_action_score = 0.7 × ctr_gap_score + 0.3 × staleness_score`, applied
only to pages clearing the 500-impression volume floor.

**Why these weights:** two candidate signals were checked against real bucketed data before
building the rule. CTR vs. position tier showed a clean, monotonic drop (1.484 → 0.652 →
0.323 → 0.222 → 0.150 from `top_3` to `deep`) — a **confirmed** signal, matching the logic
behind FlyRank's own `low_ctr_visible_page` flag. Staleness vs. decline rate was
**not** monotonic (51.2% → 61.1% → 46.7% → 60.0% across increasing staleness buckets, with the
two oldest buckets carrying very small n) — a **mixed** result. The weights (0.7 / 0.3)
reflect this: the confirmed, stronger signal is weighted more heavily than the mixed one.

**Reason code:** `underperforming_ctr_and_stale`. **Action label:** `review_for_refresh` in
the top quartile of score, else `monitor`.

**Fairness of the comparison:** the baseline is evaluated on the identical client-holdout test
split used for the models (Section 5), using the identical metric (Precision@50), with the
expected-CTR-by-tier reference computed from training data only to avoid leaking test-set
information into the rule itself.

## 4. Model / analysis

**Method:** Random Forest Classifier (200 trees, max depth 6, class-balanced), with Logistic
Regression fit alongside as a simple-model sanity check — if a plain linear model matched the
Random Forest, the added complexity would not be justified.

**Features:** `impressions_90d`, `avg_position`, `ctr`, `content_age_days`,
`days_since_last_update`, `word_count` (median-imputed — 4,511 of ~15,466 training rows,
~29%, were missing this field), and one-hot-encoded `position_tier`. Numeric features were
standardized for Logistic Regression only (tree-based models do not require scaling).

**Deliberately left out:** `trend_direction`, `trend_pct` (label-derived), and any FlyRank
product decision flag (not present in this dataset by design).

**Target / proxy, in one sentence:** `is_declining = (trend_direction == "down")` — a
current-90-day-window bucket, not a forward-validated future outcome.

## 5. Evaluation

**Split:** client-holdout (`GroupShuffleSplit`, grouped by `client_id`, 80/20 — 22 train
clients / 6 test clients, 0 client overlap confirmed). A plain random row split was
deliberately avoided: pages from the same client likely share templates and baseline quality,
so a random split risks the model memorizing client-specific patterns rather than learning a
generalizable rule. A direct before/after check confirmed this risk was real: re-running the
identical Random Forest under a naive random split produced 27 overlapping clients and an
inflated Precision@50 of 0.88 — higher than the grouped result (0.76) precisely because it
partly reflected memorization rather than generalization.

**Base rate:** 59.6% of eligible pages carry the positive label (`is_declining`) — a high
base rate, which is one reason Precision@50 (evaluated at the very top of the ranked list) is
reported alongside the full comparison rather than accuracy alone.

**Results — model vs. baseline, same split, same metric:**

| Method | Precision@50 |
|---|---|
| Baseline rule (`underperforming_ctr_and_stale`) | **0.90** |
| Logistic Regression | 0.92 |
| Random Forest | 0.76 |

**Error analysis:** where the Random Forest and the baseline disagreed in their top-50 picks
(40 unique to each), the baseline's unique picks had a 90.0% actual decline rate at an
average of 2,958 impressions, while the Random Forest's unique picks had only a 72.5% decline
rate despite nearly 3× the average impressions (8,763). The Random Forest's own feature
importances confirm the likely cause: `content_age_days` (0.26) outranked `ctr` (0.19),
meaning the model leaned on a general traffic/age signal rather than the specific,
independently confirmed CTR-vs-position gap the baseline targets directly.

**Leakage audit (final feature set):** confirmed clean — no label-derived or future-window
field among the six numeric features or the one categorical feature; feature-to-label
correlations ranged from −0.18 to +0.03, well below the 0.9 threshold that would suggest a
feature secretly encodes the label.

## 6. Interpretation

The honest finding here is a negative result for model complexity: a two-signal, hand-tuned
rule outperformed a 200-tree Random Forest on this task, on this data, under a
leakage-checked, client-grouped evaluation. This is the reverse of the starter pipeline's own
demonstration (baseline 0.240 → Random Forest 0.740 in the original ML-01 walkthrough),
which makes it worth stating plainly rather than smoothing over: **more complexity is not
automatically better**, and a model that spreads attention across a broad feature set can
under-perform a rule built around one well-validated, targeted signal.

A secondary finding surfaced during manual review of the top-ranked pages: a cluster of pages
in the top position tier, with 500+ impressions, reported **exactly 0% CTR** — statistically
unusual for a `top_3` search result. The likely explanation is a tracking or indexing issue
(broken GSC-to-page link, redirect, or canonical problem), not a genuine content quality
problem. This distinction was built directly into the action playbook (Section 7) as its own
category, rather than folded into generic "needs refresh" recommendations.

## 7. Recommendation

The validated baseline rule feeds a three-tier action playbook:

| Action | Count | Meaning |
|---|---|---|
| `monitor` | 12,177 | Below the top quartile of score — no action this review cycle |
| `review_for_refresh` | 4,003 | Real CTR gap for its position tier, plus moderate staleness — a genuine content-review candidate |
| `flag_for_tracking_audit` | 546 | Top-tier position, 500+ impressions, exactly 0% CTR — likely a tracking/indexing problem, **not** a content issue |

**How a reviewer would use this tomorrow:** open the ranked queue, work top-down within
weekly capacity, and for each page read its reason code before acting. For
`flag_for_tracking_audit` rows specifically, verify indexing, canonical tags, and redirect
chains *before* any content work — a content rewrite is the wrong first action if the real
problem is broken tracking.

**What should never be automated:** publishing AI-rewritten content from the score alone;
deprioritizing or removing low-scoring pages (a `monitor` label means "not flagged this
cycle," not "worthless"); bulk-actioning the large tie cluster found during manual review
(24 pages sharing an identical score) without spot-checking, since identical scores do not
guarantee an identical root cause; and treating any `flag_for_tracking_audit` row as a content
problem before a human confirms the tracking issue.

**Confidence and limits, stated explicitly:** this is decision-support, not a production
system. The proxy label is a current-state bucket, not a validated future outcome. The
baseline-vs-model comparison rests on a small held-out test set (6 clients, ~1,260 rows) and
should be treated as directional pending a larger or repeated evaluation. No experiment was
run to test whether refreshing a flagged page actually causes recovery — this pipeline
identifies *candidates worth a human's attention*, nothing more.

## 8. Reproducibility

**To re-run from a fresh clone:**
```bash
git clone https://github.com/MaryumAkram16/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt
```
Then open and run, in order: `work/notebooks/w01_research_question.ipynb` through
`work/notebooks/w07_action_playbook.ipynb`, and finally `work/notebooks/capstone.ipynb`
(each is Colab-ready via the badge in its first cell, or runs locally with Jupyter).

**Random seeds:** `random_state=42` fixed throughout (`GroupShuffleSplit`,
`RandomForestClassifier`, `DecisionTreeClassifier`).

**Evaluated once, on a committed split:** the client-holdout split (22 train / 6 test
clients) is regenerated deterministically from the fixed seed in
`work/notebooks/w05_model.ipynb`; the resulting comparison numbers are committed as receipts
in `work/outputs/playbook_metrics.json`, so this report's figures can be checked against that
file directly rather than taken on faith.

**Data source:** `data/raw/content_refresh_anonymized.csv` (ships in this repo); the
warehouse exploration used `FlyRank/internship-warehouse` on Hugging Face (gated, free,
instant approval).

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).

---

> **Claims checklist:** every claim above uses observed / measured / directional /
> decision-support language. No causal claims are made without an experiment. No claim is
> made about Google's ranking algorithm. No client-identifying detail appears anywhere in
> this report or the underlying repo. The base rate (59.6%) is reported alongside
> Precision@50 so the metric is not read in isolation.
