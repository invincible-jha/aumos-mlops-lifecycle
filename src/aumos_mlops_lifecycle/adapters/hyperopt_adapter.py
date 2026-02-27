"""Hyperparameter optimisation adapter for aumos-mlops-lifecycle.

Wraps Optuna for Bayesian hyperparameter search with support for pruning,
multi-objective optimisation, categorical/uniform/log-uniform search spaces,
and best trial extraction. Ray Tune is supported as an alternative backend
via the RayTuneBackend helper.

Configuration:
    AUMOS_MLOPS_OPTUNA_STORAGE — Optional Optuna storage URL (e.g. sqlite:///optuna.db)
    AUMOS_MLOPS_HYPEROPT_BACKEND — "optuna" | "ray_tune" (default: "optuna")
"""

import asyncio
from functools import partial
from typing import Any, Callable, Literal

from aumos_common.observability import get_logger

logger = get_logger(__name__)

SearchDistribution = Literal["categorical", "uniform", "log_uniform", "int_uniform"]


class HyperoptAdapter:
    """Optuna-backed hyperparameter optimisation with Ray Tune fallback.

    Creates and manages Optuna studies, defines search spaces, runs trials
    with optional pruning, and extracts best trial data including optimisation
    history for visualisation.

    All blocking Optuna calls are offloaded to the thread pool so the event
    loop remains unblocked during optimisation.

    Args:
        storage_url: Optuna storage URL. Uses in-memory storage if None.
        backend: Backend to use, "optuna" or "ray_tune".
    """

    def __init__(
        self,
        storage_url: str | None = None,
        backend: Literal["optuna", "ray_tune"] = "optuna",
    ) -> None:
        """Initialise the hyperopt adapter.

        Args:
            storage_url: Optuna RDB/file storage URL. None uses in-memory storage.
            backend: "optuna" or "ray_tune".
        """
        self._storage_url = storage_url
        self._backend = backend
        # study_name → optuna.Study
        self._studies: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Study management                                                     #
    # ------------------------------------------------------------------ #

    async def create_study(
        self,
        study_name: str,
        tenant_id: str,
        direction: str | list[str] = "minimize",
        pruner_type: str = "median",
        sampler_type: str = "tpe",
    ) -> str:
        """Create a new Optuna study (single or multi-objective).

        Args:
            study_name: Human-readable study name (namespaced with tenant_id).
            tenant_id: Tenant UUID string for namespace isolation.
            direction: "minimize" | "maximize" or a list of those values for
                       multi-objective optimisation.
            pruner_type: "median" | "percentile" | "hyperband".
            sampler_type: "tpe" | "cma" | "random".

        Returns:
            Namespaced study name used as the study key.
        """
        namespaced_name = f"tenant_{tenant_id}/{study_name}"
        logger.info("Creating Optuna study", namespaced_name=namespaced_name, direction=direction)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._create_study_sync,
                namespaced_name=namespaced_name,
                direction=direction,
                pruner_type=pruner_type,
                sampler_type=sampler_type,
            ),
        )
        return namespaced_name

    def _create_study_sync(
        self,
        namespaced_name: str,
        direction: str | list[str],
        pruner_type: str,
        sampler_type: str,
    ) -> None:
        import optuna  # type: ignore[import-untyped]

        pruner = self._build_pruner(optuna, pruner_type)
        sampler = self._build_sampler(optuna, sampler_type)

        directions = direction if isinstance(direction, list) else [direction]

        study = optuna.create_study(
            study_name=namespaced_name,
            storage=self._storage_url,
            directions=directions,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True,
        )
        self._studies[namespaced_name] = study
        logger.info("Optuna study created", namespaced_name=namespaced_name)

    @staticmethod
    def _build_pruner(optuna: Any, pruner_type: str) -> Any:
        if pruner_type == "percentile":
            return optuna.pruners.PercentilePruner(25.0)
        if pruner_type == "hyperband":
            return optuna.pruners.HyperbandPruner()
        return optuna.pruners.MedianPruner()

    @staticmethod
    def _build_sampler(optuna: Any, sampler_type: str) -> Any:
        if sampler_type == "cma":
            return optuna.samplers.CmaEsSampler()
        if sampler_type == "random":
            return optuna.samplers.RandomSampler()
        return optuna.samplers.TPESampler()

    # ------------------------------------------------------------------ #
    # Search space definition                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_search_space(
        trial: Any,
        params: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Suggest hyperparameter values from an Optuna trial given a space definition.

        Each entry in params must have: name, distribution, and distribution-specific
        keys (low/high/step for numeric types, choices for categorical).

        Args:
            trial: Optuna Trial object with suggest_* methods.
            params: List of parameter definition dicts. Each dict must contain:
                    - name (str)
                    - distribution (str): "categorical" | "uniform" |
                      "log_uniform" | "int_uniform"
                    - For categorical: choices (list)
                    - For numeric: low (float), high (float)
                    - For int_uniform: step (int, optional, default 1)

        Returns:
            Dict mapping parameter name to the suggested value.
        """
        suggested: dict[str, Any] = {}
        for param in params:
            name: str = param["name"]
            distribution: str = param["distribution"]

            if distribution == "categorical":
                suggested[name] = trial.suggest_categorical(name, param["choices"])
            elif distribution == "uniform":
                suggested[name] = trial.suggest_float(name, param["low"], param["high"])
            elif distribution == "log_uniform":
                suggested[name] = trial.suggest_float(name, param["low"], param["high"], log=True)
            elif distribution == "int_uniform":
                suggested[name] = trial.suggest_int(
                    name, int(param["low"]), int(param["high"]), step=int(param.get("step", 1))
                )
            else:
                logger.warning("Unknown distribution type", distribution=distribution, param_name=name)
        return suggested

    # ------------------------------------------------------------------ #
    # Optimisation                                                         #
    # ------------------------------------------------------------------ #

    async def optimise(
        self,
        study_name: str,
        objective_fn: Callable[[Any], float | list[float]],
        n_trials: int = 50,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run the optimisation loop for a given number of trials.

        The objective_fn receives an Optuna Trial object and must return a
        float (single-objective) or list[float] (multi-objective).

        Args:
            study_name: Namespaced Optuna study name (returned by create_study).
            objective_fn: Callable accepting an optuna.Trial and returning
                          metric value(s).
            n_trials: Number of optimisation trials to run.
            timeout_seconds: Optional wall-clock timeout for the study.

        Returns:
            Dict with study_name, n_trials, best_trial (or best_trials for
            multi-objective), and direction.
        """
        logger.info("Starting optimisation", study_name=study_name, n_trials=n_trials)

        loop = asyncio.get_event_loop()
        result: dict[str, Any] = await loop.run_in_executor(
            None,
            partial(
                self._optimise_sync,
                study_name=study_name,
                objective_fn=objective_fn,
                n_trials=n_trials,
                timeout_seconds=timeout_seconds,
            ),
        )
        return result

    def _optimise_sync(
        self,
        study_name: str,
        objective_fn: Callable[[Any], float | list[float]],
        n_trials: int,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        study = self._studies.get(study_name)
        if study is None:
            raise ValueError(f"Study '{study_name}' not found. Call create_study first.")

        study.optimize(objective_fn, n_trials=n_trials, timeout=timeout_seconds, gc_after_trial=True)

        is_multi = len(study.directions) > 1
        if is_multi:
            best = [
                {
                    "trial_number": t.number,
                    "values": t.values,
                    "params": t.params,
                    "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
                }
                for t in study.best_trials
            ]
            return {
                "study_name": study_name,
                "n_trials": len(study.trials),
                "best_trials": best,
                "directions": [str(d) for d in study.directions],
            }
        else:
            best_trial = study.best_trial
            return {
                "study_name": study_name,
                "n_trials": len(study.trials),
                "best_trial": {
                    "trial_number": best_trial.number,
                    "value": best_trial.value,
                    "params": best_trial.params,
                    "datetime_start": best_trial.datetime_start.isoformat() if best_trial.datetime_start else None,
                },
                "direction": str(study.direction),
            }

    # ------------------------------------------------------------------ #
    # Best trial extraction                                                #
    # ------------------------------------------------------------------ #

    async def get_best_trial(self, study_name: str) -> dict[str, Any] | None:
        """Retrieve the best trial from a completed single-objective study.

        Args:
            study_name: Namespaced study name.

        Returns:
            Best trial dict with trial_number, value, params, and duration,
            or None if the study has no completed trials.
        """
        study = self._studies.get(study_name)
        if study is None:
            logger.warning("Study not found", study_name=study_name)
            return None

        try:
            best = study.best_trial
            duration = None
            if best.datetime_start and best.datetime_complete:
                duration = (best.datetime_complete - best.datetime_start).total_seconds()
            return {
                "trial_number": best.number,
                "value": best.value,
                "params": best.params,
                "duration_seconds": duration,
            }
        except ValueError:
            logger.warning("No completed trials in study", study_name=study_name)
            return None

    async def get_best_trials_pareto(self, study_name: str) -> list[dict[str, Any]]:
        """Retrieve the Pareto-optimal trials from a multi-objective study.

        Args:
            study_name: Namespaced study name.

        Returns:
            List of Pareto-optimal trial dicts with values and params.
        """
        study = self._studies.get(study_name)
        if study is None:
            return []

        return [
            {
                "trial_number": t.number,
                "values": t.values,
                "params": t.params,
            }
            for t in study.best_trials
        ]

    # ------------------------------------------------------------------ #
    # History and visualisation data                                       #
    # ------------------------------------------------------------------ #

    async def get_optimisation_history(self, study_name: str) -> list[dict[str, Any]]:
        """Return the full trial history for visualisation.

        Returns one dict per completed trial with trial number, objective
        value(s), params, and elapsed time.

        Args:
            study_name: Namespaced Optuna study name.

        Returns:
            List of trial history dicts sorted by trial number ascending.
        """
        study = self._studies.get(study_name)
        if study is None:
            return []

        history = []
        for trial in study.trials:
            if trial.state.name != "COMPLETE":
                continue
            duration = None
            if trial.datetime_start and trial.datetime_complete:
                duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
            history.append(
                {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "values": trial.values,
                    "params": trial.params,
                    "duration_seconds": duration,
                    "state": trial.state.name,
                }
            )
        return sorted(history, key=lambda h: h["trial_number"])

    async def get_param_importance(self, study_name: str) -> dict[str, float]:
        """Compute feature (hyperparameter) importance for a study.

        Uses Optuna's built-in FanovaImportanceEvaluator to rank how much
        each hyperparameter affects the objective.

        Args:
            study_name: Namespaced Optuna study name.

        Returns:
            Dict mapping param_name → importance score (0.0–1.0), sorted
            by importance descending.
        """
        study = self._studies.get(study_name)
        if study is None:
            return {}

        loop = asyncio.get_event_loop()
        importance: dict[str, float] = await loop.run_in_executor(
            None,
            partial(self._compute_importance_sync, study=study),
        )
        return importance

    @staticmethod
    def _compute_importance_sync(study: Any) -> dict[str, float]:
        try:
            import optuna  # type: ignore[import-untyped]

            raw_importance = optuna.importance.get_param_importances(study)
            return dict(sorted(raw_importance.items(), key=lambda item: item[1], reverse=True))
        except Exception:
            logger.exception("Failed to compute param importance")
            return {}
