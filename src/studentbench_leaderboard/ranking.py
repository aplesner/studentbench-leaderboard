"""Model-level pairwise rankings and subject-clustered alignment uncertainty.

``value`` is the probability of model A winning: wins/ties/losses map to
1/.5/0. ``fit_bradley_terry`` consumes already aggregated pair observations;
``aggregate_pairs`` first gives repeated reviews and criteria equal weights.
No global ranks are assigned to disconnected comparison graphs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import isfinite, sqrt
import re
from typing import Iterable, Mapping

import numpy as np

from .data import Dataset, digest


TIE_TOLERANCE = 1e-8


def judgment_rows(dataset: Dataset, records: Iterable[Mapping]) -> list[dict]:
    """Adapt one judge version's SSOT comparisons to model-ranking criteria.

    ``preference_probabilities`` is authoritative when present, even if its
    value is None while the categorical score has a majority. Otherwise the
    signed categorical score maps to win/tie/loss share. Values are normalized
    to sorted *plan* IDs, then ``aggregate_pairs`` handles model orientation.
    Missing observations retain ``value=None`` for coverage accounting and
    are excluded by the ranking aggregation and fitter.
    """
    rows = []
    versions = set()
    requests = set()
    for record in records:
        if record.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported judgment record schema")
        if record.get("instrument") != dataset.instrument or record.get("dataset_sha256") != dataset.fingerprint:
            raise ValueError("Judgment instrument or dataset fingerprint differs from SSOT")
        if record.get("mode") != "pairwise":
            raise ValueError("Model ranking requires pairwise judgments")
        version = record.get("judge_version", record.get("judge_id"))
        if not isinstance(version, str) or not version:
            raise ValueError("Judgment needs a judge version or legacy judge ID")
        versions.add(version)
        if len(versions) > 1:
            raise ValueError("Adapt one judge version at a time; do not pool different judges")
        request = record.get("request_id")
        if request is not None:
            if request in requests:
                raise ValueError(f"Duplicate judgment request: {request}")
            requests.add(request)
        original = record.get("plan_ids")
        if (not isinstance(original, (list, tuple)) or len(original) != 2
                or not all(isinstance(plan, str) and plan in dataset.plans for plan in original)
                or original[0] == original[1]):
            raise ValueError("Judgment needs two distinct actual SSOT plan IDs")
        original = list(original)
        subject = record.get("subject")
        if subject not in dataset.pretests or any(subject not in dataset.plans[plan]["review_subject_ids"] for plan in original):
            raise ValueError("Judgment plans do not belong to the supplied SSOT subject")
        for slot, plan in zip(("model_a", "model_b"), original):
            if slot in record and record[slot] != dataset.plans[plan]["ai_model_preset_id"]:
                raise ValueError(f"Judgment {slot} disagrees with its actual SSOT plan")
        if record.get("grading_id") is not None:
            session = dataset.sessions.get(record["grading_id"])
            if session is None or session["review_subject_id"] != subject:
                raise ValueError("Judgment grading context belongs to a different SSOT subject")
        plans = sorted(original)
        criterion_ids = [criterion["id"] for criterion in record["criteria"]]
        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("Judgment has duplicate criterion IDs")
        probabilities = record.get("preference_probabilities")
        soft = "preference_probabilities" in record
        if soft and (not isinstance(probabilities, dict) or set(probabilities) != set(criterion_ids)):
            raise ValueError("Preference probabilities must cover exactly the requested criteria")
        for criterion in record["criteria"]:
            identifier, scale = criterion["id"], criterion["scale_id"]
            if identifier not in dataset.questions or scale not in dataset.scales:
                raise ValueError("Judgment criterion or rating scale is absent from SSOT")
            if soft:
                value = probabilities[identifier]
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not isfinite(value) or not 0 <= value <= 1):
                    raise ValueError("Preference probabilities must be finite numbers in [0, 1] or None")
                value = float(value) if value is not None else None
            else:
                score = record["ratings"][identifier]["score"]
                if score is not None and (type(score) is not int or not -2 <= score <= 2):
                    raise ValueError("Pairwise score must be an integer in -2..2 or None")
                value = None if score is None else 1.0 if score > 0 else 0.0 if score < 0 else .5
            if value is not None and original != plans:
                value = 1 - value
            match = re.search(r"_q(\d{2})(?:_|$)", identifier)
            family = "q" + match.group(1) if match else dataset.questions[identifier]["rubric_question_id"]
            rows.append({"subject": subject, "pair_id": digest([subject, *plans])[:20],
                         "plan_a": plans[0], "plan_b": plans[1],
                         "model_a": dataset.plans[plans[0]]["ai_model_preset_id"],
                         "model_b": dataset.plans[plans[1]]["ai_model_preset_id"],
                         "criterion": identifier + "|" + scale, "criterion_family": family,
                         "value": value, "judge_version": version,
                         "value_source": "preference_probabilities" if soft else "categorical_score"})
    return rows


def _validated(rows: Iterable[Mapping]) -> list[dict]:
    result = []
    for original in rows:
        row = dict(original)
        for key in ("subject", "pair_id", "model_a", "model_b"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Comparison needs a nonempty string {key}")
        if row.get("value") is None:
            continue  # An abstention is missing data, never an inferred tie.
        value, weight = float(row["value"]), float(row.get("weight", 1.0))
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Comparison value must be finite and in [0, 1]")
        if not isfinite(weight) or weight <= 0:
            raise ValueError("Comparison weight must be finite and positive")
        row.update(value=value, weight=weight)
        result.append(row)
    return result


def aggregate_pairs(rows: Iterable[Mapping]) -> list[dict]:
    """Average reviews within each criterion, then criteria within each pair.

    A pair has unit weight regardless of its number of raters or criteria.
    Pass an exact question-variant/scale identifier as ``criterion`` to keep
    distinct human questions separate. Optional ``criterion_family`` groups
    variants before equal-weight averaging over rubric families. Reversed
    model slots are normalized.
    This helper must run *before* cluster resampling, not after it.
    """
    criteria = defaultdict(list)
    identities = {}
    criterion_families = {}
    for row in _validated(rows):
        a, b = row["model_a"], row["model_b"]
        value = row["value"]
        if b < a:
            a, b, value = b, a, 1 - value
        pair = (row["subject"], row["pair_id"])
        if pair in identities and identities[pair] != (a, b):
            raise ValueError(f"Inconsistent model identities for pair {pair}")
        identities[pair] = (a, b)
        criterion = row.get("criterion", "overall")
        if isinstance(criterion, list):
            criterion = tuple(criterion)
        key = (*pair, criterion)
        family = row.get("criterion_family", criterion)
        if isinstance(family, list):
            family = tuple(family)
        if key in criterion_families and criterion_families[key] != family:
            raise ValueError(f"Inconsistent criterion family for {key}")
        criterion_families[key] = family
        criteria[key].append((value, row["weight"]))
    families = defaultdict(list)
    pairs = defaultdict(list)
    review_counts = Counter()
    exact_counts = Counter()
    for key, values in criteria.items():
        subject, pair_id, _criterion = key
        total_weight = sum(weight for _, weight in values)
        families[(subject, pair_id, criterion_families[key])].append(
            sum(value * weight for value, weight in values) / total_weight)
        review_counts[(subject, pair_id)] += len(values)
        exact_counts[(subject, pair_id)] += 1
    for (subject, pair_id, _family), values in families.items():
        pairs[(subject, pair_id)].append(float(np.mean(values)))
    return [{"subject": subject, "pair_id": pair_id,
             "model_a": identities[(subject, pair_id)][0],
             "model_b": identities[(subject, pair_id)][1],
             "value": float(np.mean(values)), "weight": 1.0,
             "n_criteria": len(values), "n_exact_criteria": exact_counts[(subject, pair_id)],
             "n_criterion_ratings": review_counts[(subject, pair_id)]}
            for (subject, pair_id), values in sorted(pairs.items())]


def _components(models: list[str], rows: list[dict]) -> list[list[str]]:
    adjacency = {model: set() for model in models}
    for row in rows:
        a, b = row["model_a"], row["model_b"]
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    unseen = set(models)
    result = []
    while unseen:
        pending = [min(unseen)]
        component = set()
        while pending:
            model = pending.pop()
            if model in component:
                continue
            component.add(model)
            pending.extend(adjacency[model] - component)
        unseen -= component
        result.append(sorted(component))
    return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -values))


def _ranks(values: list[float], *, descending: bool = True) -> list[float]:
    """Midranks for numerical ties; rank 1 is the highest ability by default."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=descending)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and abs(values[order[stop]] - values[order[start]]) <= TIE_TOLERANCE:
            stop += 1
        rank = (start + 1 + stop) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def fit_bradley_terry(rows: Iterable[Mapping], models: Iterable[str] | None = None,
                     ridge: float = 1.0, max_iter: int = 100,
                     tolerance: float = 1e-10) -> dict:
    """Fit fractional-win Bradley–Terry with an L2 ability penalty.

    The objective is sum(weight * binomial cross-entropy) + ridge/2 * ||s||².
    A tie contributes half a win to both sides; no tie-propensity parameter is
    estimated. Ridge keeps separated results finite, but cannot justify ranks
    between disconnected graph components. Scores are centered per component.
    """
    if not isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive")
    observations = _validated(rows)
    observed_models = {row[key] for row in observations for key in ("model_a", "model_b")}
    model_names = sorted(set(models) if models is not None else observed_models)
    if any(not isinstance(model, str) or not model for model in model_names):
        raise ValueError("Model identifiers must be nonempty strings")
    if not observed_models.issubset(model_names):
        raise ValueError("An observation contains a model outside the requested model universe")
    observations = [r for r in observations if r["model_a"] != r["model_b"]]
    groups = _components(model_names, observations)
    connected = len(model_names) >= 2 and len(groups) == 1
    index = {model: position for position, model in enumerate(model_names)}
    design = np.zeros((len(observations), len(model_names)))
    outcomes = np.asarray([row["value"] for row in observations])
    weights = np.asarray([row["weight"] for row in observations])
    for position, row in enumerate(observations):
        design[position, index[row["model_a"]]] = 1
        design[position, index[row["model_b"]]] = -1
    abilities = np.zeros(len(model_names))

    def objective(candidate):
        difference = design @ candidate
        return float(np.sum(weights * (np.logaddexp(0, difference) - outcomes * difference))
                     + ridge / 2 * (candidate @ candidate))

    converged = not observations
    iterations = 0
    for iteration in range(max_iter if observations else 0):
        iterations = iteration + 1
        probabilities = _sigmoid(design @ abilities)
        gradient = design.T @ (weights * (probabilities - outcomes)) + ridge * abilities
        if np.max(np.abs(gradient)) <= tolerance:
            converged = True
            break
        curvature = weights * probabilities * (1 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) + ridge * np.eye(len(model_names))
        step = np.linalg.solve(hessian, gradient)
        current = objective(abilities)
        step_size = 1.0
        while step_size > 1e-8:
            candidate = abilities - step_size * step
            if objective(candidate) <= current - 1e-4 * step_size * float(gradient @ step):
                break
            step_size *= .5
        abilities = candidate
        if np.max(np.abs(step_size * step)) <= tolerance:
            converged = True
            break
    # Remove numerical offsets independently; no cross-component interpretation.
    component_for = {}
    component_ranks = {}
    for number, group in enumerate(groups):
        indices = [index[model] for model in group]
        abilities[indices] -= np.mean(abilities[indices])
        ranks = _ranks(abilities[indices].tolist())
        for model, rank in zip(group, ranks):
            component_for[model] = number
            component_ranks[model] = rank
    counts, total_weights, subjects = Counter(), Counter(), defaultdict(set)
    for row in observations:
        for model in (row["model_a"], row["model_b"]):
            counts[model] += 1
            total_weights[model] += row["weight"]
            subjects[model].add(row["subject"])
    leaderboard = [{"model": model, "score": float(abilities[index[model]]),
                    "rank": component_ranks[model] if connected else None,
                    "component": component_for[model],
                    "component_rank": component_ranks[model] if len(groups[component_for[model]]) > 1 else None,
                    "n_comparisons": counts[model], "comparison_weight": total_weights[model],
                    "n_subjects": len(subjects[model])}
                   for model in model_names]
    leaderboard.sort(key=lambda row: (row["component"], -row["score"], row["model"]))
    pairwise = [{"model_a": a, "model_b": b,
                 "probability_a": float(_sigmoid(np.asarray([abilities[index[a]] - abilities[index[b]]]))[0])}
                for a, b in combinations(model_names, 2) if component_for[a] == component_for[b]]
    observed_cross_model = {model for model, count in counts.items() if count}
    return {"method": "fractional_bradley_terry", "ridge": ridge, "models": model_names,
            "connected": connected, "components": groups, "converged": converged,
            "iterations": iterations, "objective": objective(abilities),
            "n_comparisons": len(observations), "n_subjects": len({r["subject"] for r in observations}),
            "model_coverage": len(observed_cross_model) / len(model_names) if model_names else 0.0,
            "models_without_comparisons": sorted(set(model_names) - observed_cross_model),
            "unrankable_models": [] if connected else model_names,
            "leaderboard": leaderboard, "pairwise_probabilities": pairwise,
            "warning": None if connected else "Comparison graph is disconnected or has fewer than two models; global ranks are undefined."}


def _sign(value: float) -> int:
    return 0 if abs(value) <= TIE_TOLERANCE else (1 if value > 0 else -1)


def compare_rankings(reference: dict, predicted: dict) -> dict:
    """Compare rankings only when they cover the same connected model universe.

    Top-1 overlap is the Jaccard overlap of tied winner sets. Pair-order
    agreement requires exactly matching directions, including tied ranks.
    Probability discrepancies compare BT-implied probabilities for every
    common unordered model pair, equally weighted regardless of schedule.
    """
    metrics = ("spearman", "kendall_tau_b", "pair_order_agreement", "top1_overlap",
               "rank_mae", "normalized_rank_mae", "probability_mae", "probability_rmse")
    result = {key: None for key in metrics}
    left = {row["model"]: row["score"] for row in reference["leaderboard"]}
    right = {row["model"]: row["score"] for row in predicted["leaderboard"]}
    models = sorted(set(left) & set(right))
    universe = set(left) | set(right)
    result.update(models=models, n_models=len(models), n_model_pairs=len(models) * (len(models) - 1) // 2,
                  reference_connected=reference["connected"], predicted_connected=predicted["connected"],
                  same_model_universe=set(left) == set(right),
                  reference_model_coverage=len(left) / len(universe) if universe else 0.0,
                  predicted_model_coverage=len(right) / len(universe) if universe else 0.0,
                  missing_reference_models=sorted(set(right) - set(left)),
                  missing_predicted_models=sorted(set(left) - set(right)))
    if (not reference["connected"] or not predicted["connected"] or len(models) < 2
            or set(left) != set(right) or not reference.get("converged", True)
            or not predicted.get("converged", True)):
        result["warning"] = "Global rank alignment requires connected, converged rankings over the same model universe (at least two models)."
        return result
    x, y = [left[m] for m in models], [right[m] for m in models]
    rx, ry = np.asarray(_ranks(x)), np.asarray(_ranks(y))
    if np.std(rx) > 0 and np.std(ry) > 0:
        result["spearman"] = float(np.corrcoef(rx, ry)[0, 1])
    concordant = discordant = tied_left = tied_right = agreement = 0
    errors = []
    for i, j in combinations(range(len(models)), 2):
        sx, sy = _sign(x[i] - x[j]), _sign(y[i] - y[j])
        agreement += sx == sy
        if sx and sy:
            concordant += sx == sy
            discordant += sx != sy
        elif sy:
            tied_left += 1
        elif sx:
            tied_right += 1
        px, py = _sigmoid(np.asarray([x[i] - x[j], y[i] - y[j]]))
        errors.append(float(px - py))
    denominator = sqrt((concordant + discordant + tied_left) * (concordant + discordant + tied_right))
    if denominator:
        result["kendall_tau_b"] = (concordant - discordant) / denominator
    top_left = {models[i] for i, rank in enumerate(rx) if rank == min(rx)}
    top_right = {models[i] for i, rank in enumerate(ry) if rank == min(ry)}
    result.update(pair_order_agreement=agreement / len(errors),
                  top1_overlap=len(top_left & top_right) / len(top_left | top_right),
                  rank_mae=float(np.mean(np.abs(rx - ry))),
                  normalized_rank_mae=float(np.mean(np.abs(rx - ry))) / (len(models) - 1),
                  probability_mae=float(np.mean(np.abs(errors))),
                  probability_rmse=float(np.sqrt(np.mean(np.square(errors)))),
                  warning=None)
    return result


def _interval(values: list[float], confidence: float) -> dict:
    if not values:
        return {"low": None, "high": None, "n_valid": 0}
    low, high = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {"low": float(low), "high": float(high), "n_valid": len(values)}


def bootstrap_alignment(human_rows: Iterable[Mapping], judge_rows: Iterable[Mapping],
                        n_bootstrap: int = 500, seed: int = 0,
                        models: Iterable[str] | None = None, ridge: float = 1.0,
                        confidence: float = .95) -> dict:
    """Paired cluster bootstrap on common subjects, with fixed model universe.

    Inputs must already be aggregated per pair. A sampled subject's entire
    human and AI comparison sets receive the same multiplicity. Densities
    may differ. Subjects absent from either side are excluded and reported.
    Disconnected bootstrap fits do not contribute global rank statistics;
    their count is explicit, since conditional intervals can be optimistic.
    """
    if n_bootstrap < 0 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be nonnegative and confidence in (0, 1)")
    human, judge = _validated(human_rows), _validated(judge_rows)
    human_subjects = {r["subject"] for r in human}
    judge_subjects = {r["subject"] for r in judge}
    common = sorted(human_subjects & judge_subjects)
    if not common:
        raise ValueError("Human and judge comparisons have no common subjects")
    human = [r for r in human if r["subject"] in common]
    judge = [r for r in judge if r["subject"] in common]
    universe = sorted(set(models) if models is not None else
                      {r[k] for r in human + judge for k in ("model_a", "model_b")})
    reference = fit_bradley_terry(human, models=universe, ridge=ridge)
    predicted = fit_bradley_terry(judge, models=universe, ridge=ridge)
    point = compare_rankings(reference, predicted)
    metric_values = defaultdict(list)
    rank_values = {source: defaultdict(list) for source in ("human", "judge")}
    top_counts = {source: Counter() for source in ("human", "judge")}
    valid_fits = Counter()
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        multiplicities = Counter(rng.choice(common, size=len(common), replace=True).tolist())
        fits = {}
        for source, observations in (("human", human), ("judge", judge)):
            sampled = [{**r, "weight": r["weight"] * multiplicities[r["subject"]]}
                       for r in observations if multiplicities[r["subject"]]]
            fit = fit_bradley_terry(sampled, models=universe, ridge=ridge)
            fits[source] = fit
            if fit["connected"] and fit["converged"]:
                valid_fits[source] += 1
                best_rank = min(r["rank"] for r in fit["leaderboard"])
                for row in fit["leaderboard"]:
                    rank_values[source][row["model"]].append(row["rank"])
                    top_counts[source][row["model"]] += row["rank"] == best_rank
        if all(fit["connected"] and fit["converged"] for fit in fits.values()):
            valid_fits["paired"] += 1
            for key, value in compare_rankings(fits["human"], fits["judge"]).items():
                if key in point and isinstance(value, (int, float)) and not isinstance(value, bool) and not key.startswith("n_"):
                    metric_values[key].append(value)
    metric_keys = ("spearman", "kendall_tau_b", "pair_order_agreement", "top1_overlap",
                   "rank_mae", "normalized_rank_mae", "probability_mae", "probability_rmse")
    return {"method": "paired_subject_percentile_bootstrap", "seed": seed,
            "n_bootstrap": n_bootstrap, "confidence": confidence,
            "n_common_subjects": len(common),
            "excluded_human_subjects": sorted(human_subjects - set(common)),
            "excluded_judge_subjects": sorted(judge_subjects - set(common)),
            "human": reference, "judge": predicted, "metrics": point,
            "metric_intervals": {key: {"estimate": point[key], **_interval(metric_values[key], confidence)}
                                 for key in metric_keys},
            "rank_intervals": {source: {model: {**_interval(rank_values[source][model], confidence),
                                                  "top1_frequency": top_counts[source][model] / valid_fits[source]
                                                  if valid_fits[source] else None}
                                         for model in universe} for source in ("human", "judge")},
            "valid_bootstrap_fits": {source: valid_fits[source] for source in ("human", "judge", "paired")},
            "interval_warning": "Intervals condition on connected, converged resamples; inspect valid counts. Subject clustering does not capture all repeated-reviewer dependence."}
