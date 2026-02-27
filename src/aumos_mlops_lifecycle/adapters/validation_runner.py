"""Model validation runner adapter for aumos-mlops-lifecycle.

Executes cross-validation, holdout evaluation, classification/regression/
ranking metric computation, statistical significance testing via paired
t-test, data leakage detection, and baseline comparison.

All CPU-intensive computations are offloaded to a thread-pool executor so
the FastAPI event loop is not blocked during long validation runs.
"""

import asyncio
from functools import partial
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ModelValidationRunner:
    """Comprehensive model validation with statistical significance testing.

    Provides cross-validation (k-fold, stratified), holdout evaluation,
    multi-task metric computation, paired t-test for A/B significance,
    baseline comparison, and a data leakage detection heuristic.

    All computationally expensive operations are offloaded to a thread
    pool to keep the async event loop responsive.
    """

    # ------------------------------------------------------------------ #
    # Cross-validation                                                     #
    # ------------------------------------------------------------------ #

    async def cross_validate(
        self,
        model: Any,
        features: Any,
        labels: Any,
        n_folds: int = 5,
        stratified: bool = True,
        scoring: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run k-fold or stratified k-fold cross-validation.

        Args:
            model: Scikit-learn compatible estimator with fit/predict methods.
            features: Feature matrix (numpy array or pandas DataFrame).
            labels: Target vector (numpy array or pandas Series).
            n_folds: Number of cross-validation folds. Defaults to 5.
            stratified: Use StratifiedKFold if True, KFold otherwise.
            scoring: List of sklearn scoring metric names. Defaults to
                     ["accuracy", "f1_weighted", "roc_auc"] for classification.

        Returns:
            Validation result dict with per-metric mean, std, and per-fold scores.
        """
        scoring = scoring or ["accuracy", "f1_weighted", "roc_auc"]
        logger.info("Starting cross-validation", n_folds=n_folds, stratified=stratified, metrics=scoring)

        loop = asyncio.get_event_loop()
        results: dict[str, Any] = await loop.run_in_executor(
            None,
            partial(
                self._cross_validate_sync,
                model=model,
                features=features,
                labels=labels,
                n_folds=n_folds,
                stratified=stratified,
                scoring=scoring,
            ),
        )
        return results

    def _cross_validate_sync(
        self,
        model: Any,
        features: Any,
        labels: Any,
        n_folds: int,
        stratified: bool,
        scoring: list[str],
    ) -> dict[str, Any]:
        from sklearn import model_selection  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        if stratified:
            cv = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            cv = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=42)

        cv_results = model_selection.cross_validate(
            model,
            features,
            labels,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        output: dict[str, Any] = {
            "n_folds": n_folds,
            "stratified": stratified,
            "metrics": {},
        }

        for metric in scoring:
            test_key = f"test_{metric}"
            train_key = f"train_{metric}"
            if test_key in cv_results:
                fold_scores = cv_results[test_key].tolist()
                output["metrics"][metric] = {
                    "mean": round(float(np.mean(fold_scores)), 4),
                    "std": round(float(np.std(fold_scores)), 4),
                    "per_fold": [round(s, 4) for s in fold_scores],
                    "train_mean": round(float(np.mean(cv_results[train_key])), 4) if train_key in cv_results else None,
                }

        logger.info("Cross-validation complete", n_folds=n_folds)
        return output

    # ------------------------------------------------------------------ #
    # Holdout evaluation                                                   #
    # ------------------------------------------------------------------ #

    async def evaluate_holdout(
        self,
        model: Any,
        test_features: Any,
        test_labels: Any,
        task_type: str = "classification",
    ) -> dict[str, Any]:
        """Evaluate a trained model on a held-out test set.

        Args:
            model: Fitted model with predict (and predict_proba for classification).
            test_features: Held-out feature matrix.
            test_labels: Held-out true labels or targets.
            task_type: "classification" | "regression" | "ranking".

        Returns:
            Metric dict appropriate for the task type.
        """
        logger.info("Evaluating on holdout set", task_type=task_type)

        loop = asyncio.get_event_loop()
        results: dict[str, Any] = await loop.run_in_executor(
            None,
            partial(
                self._evaluate_holdout_sync,
                model=model,
                test_features=test_features,
                test_labels=test_labels,
                task_type=task_type,
            ),
        )
        return results

    def _evaluate_holdout_sync(
        self,
        model: Any,
        test_features: Any,
        test_labels: Any,
        task_type: str,
    ) -> dict[str, Any]:
        if task_type == "classification":
            return self._classification_metrics_sync(model, test_features, test_labels)
        if task_type == "regression":
            return self._regression_metrics_sync(model, test_features, test_labels)
        if task_type == "ranking":
            return self._ranking_metrics_sync(model, test_features, test_labels)
        raise ValueError(f"Unsupported task_type: {task_type}")

    @staticmethod
    def _classification_metrics_sync(model: Any, features: Any, labels: Any) -> dict[str, Any]:
        from sklearn import metrics as skmetrics  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        predictions = model.predict(features)
        result: dict[str, Any] = {
            "task_type": "classification",
            "accuracy": round(float(skmetrics.accuracy_score(labels, predictions)), 4),
            "f1_weighted": round(float(skmetrics.f1_score(labels, predictions, average="weighted", zero_division=0)), 4),
            "precision_weighted": round(
                float(skmetrics.precision_score(labels, predictions, average="weighted", zero_division=0)), 4
            ),
            "recall_weighted": round(
                float(skmetrics.recall_score(labels, predictions, average="weighted", zero_division=0)), 4
            ),
        }

        try:
            proba = model.predict_proba(features)
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                result["roc_auc"] = round(float(skmetrics.roc_auc_score(labels, proba[:, 1])), 4)
            else:
                result["roc_auc_ovr"] = round(
                    float(skmetrics.roc_auc_score(labels, proba, multi_class="ovr", average="weighted")), 4
                )
        except (AttributeError, ValueError):
            pass  # Model does not support predict_proba

        return result

    @staticmethod
    def _regression_metrics_sync(model: Any, features: Any, labels: Any) -> dict[str, Any]:
        from sklearn import metrics as skmetrics  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        predictions = model.predict(features)
        return {
            "task_type": "regression",
            "mae": round(float(skmetrics.mean_absolute_error(labels, predictions)), 4),
            "mse": round(float(skmetrics.mean_squared_error(labels, predictions)), 4),
            "rmse": round(float(np.sqrt(skmetrics.mean_squared_error(labels, predictions))), 4),
            "r2": round(float(skmetrics.r2_score(labels, predictions)), 4),
            "mape": round(
                float(skmetrics.mean_absolute_percentage_error(labels, predictions)) * 100.0, 4
            ),
        }

    @staticmethod
    def _ranking_metrics_sync(model: Any, features: Any, labels: Any) -> dict[str, Any]:
        from sklearn import metrics as skmetrics  # type: ignore[import-untyped]

        scores = model.predict(features)
        return {
            "task_type": "ranking",
            "ndcg_at_10": round(float(skmetrics.ndcg_score([labels], [scores], k=10)), 4),
            "ndcg_at_5": round(float(skmetrics.ndcg_score([labels], [scores], k=5)), 4),
        }

    # ------------------------------------------------------------------ #
    # Statistical significance testing                                     #
    # ------------------------------------------------------------------ #

    async def significance_test(
        self,
        scores_model_a: list[float],
        scores_model_b: list[float],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Paired t-test to determine if model B is significantly better than A.

        Args:
            scores_model_a: Per-sample or per-fold metric scores for model A.
            scores_model_b: Per-sample or per-fold metric scores for model B.
            alpha: Significance level. Defaults to 0.05.

        Returns:
            Dict with t_statistic, p_value, is_significant, and winner.
        """
        loop = asyncio.get_event_loop()
        result: dict[str, Any] = await loop.run_in_executor(
            None,
            partial(
                self._t_test_sync,
                scores_a=scores_model_a,
                scores_b=scores_model_b,
                alpha=alpha,
            ),
        )
        return result

    @staticmethod
    def _t_test_sync(
        scores_a: list[float],
        scores_b: list[float],
        alpha: float,
    ) -> dict[str, Any]:
        from scipy import stats  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        is_significant = float(p_value) < alpha
        mean_a = float(np.mean(scores_a))
        mean_b = float(np.mean(scores_b))

        if is_significant:
            winner = "model_b" if mean_b > mean_a else "model_a"
        else:
            winner = "no_significant_difference"

        return {
            "t_statistic": round(float(t_stat), 6),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "is_significant": is_significant,
            "mean_score_a": round(mean_a, 4),
            "mean_score_b": round(mean_b, 4),
            "winner": winner,
        }

    # ------------------------------------------------------------------ #
    # Baseline comparison                                                  #
    # ------------------------------------------------------------------ #

    async def compare_to_baseline(
        self,
        model: Any,
        test_features: Any,
        test_labels: Any,
        baseline_metrics: dict[str, float],
        task_type: str = "classification",
    ) -> dict[str, Any]:
        """Compare model performance against a stored baseline.

        Args:
            model: Trained model to evaluate.
            test_features: Held-out feature matrix.
            test_labels: Held-out true labels.
            baseline_metrics: Previously recorded metric dict to compare against.
            task_type: "classification" | "regression" | "ranking".

        Returns:
            Comparison dict with current_metrics, baseline_metrics, deltas,
            and is_improvement flag.
        """
        current = await self.evaluate_holdout(model, test_features, test_labels, task_type)

        deltas: dict[str, float] = {}
        for key, current_val in current.items():
            if isinstance(current_val, float) and key in baseline_metrics:
                deltas[key] = round(current_val - baseline_metrics[key], 4)

        # Determine improvement: for regression (lower is better for MAE/MSE)
        # For classification higher is better. We use a heuristic:
        positive_metrics = {"accuracy", "f1_weighted", "roc_auc", "r2", "ndcg_at_10", "ndcg_at_5"}
        improvement_scores = [
            delta if metric in positive_metrics else -delta
            for metric, delta in deltas.items()
        ]
        is_improvement = (sum(improvement_scores) > 0) if improvement_scores else False

        return {
            "task_type": task_type,
            "current_metrics": current,
            "baseline_metrics": baseline_metrics,
            "deltas": deltas,
            "is_improvement": is_improvement,
        }

    # ------------------------------------------------------------------ #
    # Data leakage detection                                               #
    # ------------------------------------------------------------------ #

    async def detect_data_leakage(
        self,
        train_features: Any,
        test_features: Any,
        threshold_duplicate_pct: float = 1.0,
    ) -> dict[str, Any]:
        """Detect potential data leakage between train and test sets.

        Uses a hash-based row deduplication check to find rows that appear
        in both the training and test sets, which indicates label leakage.

        Args:
            train_features: Training feature matrix (pandas DataFrame or numpy array).
            test_features: Test feature matrix.
            threshold_duplicate_pct: Percentage of duplicates above which leakage
                                     is flagged as "high risk". Defaults to 1.0%.

        Returns:
            Leakage report dict with overlap_count, overlap_pct, risk_level,
            and leakage_detected flag.
        """
        loop = asyncio.get_event_loop()
        report: dict[str, Any] = await loop.run_in_executor(
            None,
            partial(
                self._detect_leakage_sync,
                train_features=train_features,
                test_features=test_features,
                threshold_pct=threshold_duplicate_pct,
            ),
        )
        return report

    @staticmethod
    def _detect_leakage_sync(
        train_features: Any,
        test_features: Any,
        threshold_pct: float,
    ) -> dict[str, Any]:
        import hashlib

        def row_hashes(matrix: Any) -> set[str]:
            hashes: set[str] = set()
            try:
                import pandas as pd  # type: ignore[import-untyped]
                import numpy as np  # type: ignore[import-untyped]

                if isinstance(matrix, pd.DataFrame):
                    rows = matrix.values
                else:
                    rows = matrix
                for row in rows:
                    row_bytes = np.asarray(row, dtype=float).tobytes()
                    hashes.add(hashlib.md5(row_bytes).hexdigest())
            except Exception:
                logger.exception("Row hashing failed")
            return hashes

        train_hashes = row_hashes(train_features)
        test_hashes = row_hashes(test_features)
        overlap = train_hashes & test_hashes
        overlap_count = len(overlap)
        test_count = len(test_hashes) or 1
        overlap_pct = (overlap_count / test_count) * 100.0

        if overlap_pct >= threshold_pct:
            risk_level = "high"
        elif overlap_pct > 0:
            risk_level = "low"
        else:
            risk_level = "none"

        return {
            "train_rows": len(train_hashes),
            "test_rows": len(test_hashes),
            "overlap_count": overlap_count,
            "overlap_pct": round(overlap_pct, 4),
            "risk_level": risk_level,
            "leakage_detected": overlap_count > 0,
        }

    # ------------------------------------------------------------------ #
    # Report generation                                                    #
    # ------------------------------------------------------------------ #

    async def generate_validation_report(
        self,
        model_name: str,
        model_version: str,
        cv_results: dict[str, Any] | None,
        holdout_results: dict[str, Any] | None,
        significance_results: dict[str, Any] | None,
        baseline_comparison: dict[str, Any] | None,
        leakage_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Assemble a structured validation report from component results.

        Args:
            model_name: Model name for the report header.
            model_version: Model version string.
            cv_results: Output from cross_validate (may be None).
            holdout_results: Output from evaluate_holdout (may be None).
            significance_results: Output from significance_test (may be None).
            baseline_comparison: Output from compare_to_baseline (may be None).
            leakage_report: Output from detect_data_leakage (may be None).

        Returns:
            Complete validation report dict ready for storage or display.
        """
        from datetime import datetime, timezone

        sections: dict[str, Any] = {}
        passed_gates = []
        failed_gates = []

        if cv_results is not None:
            sections["cross_validation"] = cv_results

        if holdout_results is not None:
            sections["holdout_evaluation"] = holdout_results

        if significance_results is not None:
            sections["significance_test"] = significance_results
            if significance_results.get("is_significant"):
                passed_gates.append("significance")
            else:
                failed_gates.append("significance")

        if baseline_comparison is not None:
            sections["baseline_comparison"] = baseline_comparison
            if baseline_comparison.get("is_improvement"):
                passed_gates.append("baseline_improvement")
            else:
                failed_gates.append("baseline_improvement")

        if leakage_report is not None:
            sections["leakage_detection"] = leakage_report
            if not leakage_report.get("leakage_detected"):
                passed_gates.append("no_leakage")
            else:
                failed_gates.append("leakage_detected")

        return {
            "model_name": model_name,
            "model_version": model_version,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "validation_passed": len(failed_gates) == 0,
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "sections": sections,
        }
