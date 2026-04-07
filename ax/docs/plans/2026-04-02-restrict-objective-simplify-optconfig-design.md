# Design: Restrict Objective to Single/Scalarized & Simplify OptimizationConfig

**Date:** 2026-04-02
**Author:** Sait Cakmak
**Design Doc:** https://docs.google.com/document/d/1EGQYmBjiNGtYapXu1RLHEBdA5Yz2c7q17acX3es0yV8/edit
**Selected Option:** Option 3 -- Restrict Objective to single, possibly scalarized objective
**Prerequisite:** D98837790 (Move metric_name_to_signature from Adapter to Objective/OutcomeConstraint)

## Summary

This plan implements Option 3 from the design doc: restrict `Objective` to represent
a single (possibly scalarized) objective, move multi-objective representation to
`OptimizationConfig(objectives=list[Objective])`, add `threshold`/`relative_threshold`
fields to `Objective`, and deprecate `MultiObjectiveOptimizationConfig`, `MultiObjective`,
`ScalarizedObjective`, and `ObjectiveThreshold` (keeping them as deprecated shims,
removing all internal usage).

## Diff Stack

Each diff is backward-compatible. Summaries should reference this design doc.

---

### Diff 1: Add `objectives: list[Objective]` to `OptimizationConfig`

**Goal:** Enable the new `OptimizationConfig(objectives=[...])` construction path
without breaking any existing code.

**Files:**
- `ax/core/optimization_config.py`
- `ax/core/tests/test_optimization_config.py`

**Changes:**

1. `OptimizationConfig.__init__` accepts new kwarg `objectives: list[Objective] | None = None`
   - Mutually exclusive with `objective` -- raise `UserInputError` if both provided
   - If `objectives` is provided, store as `self._objectives: list[Objective]`
   - If `objective` is provided (existing path), wrap as `self._objectives = [objective]`
   - Validate: no duplicate metric names across objectives in the list
   - Validate: each objective in the list must not be multi-objective (no comma expressions)

2. `objectives` property: returns `self._objectives`

3. `objective` property: returns `self._objectives[0]` if `len == 1`, raises
   `UserInputError("Access individual objectives via `objectives` property for
   multi-objective configs.")` if `len > 1`

4. `is_moo_problem` property: `len(self._objectives) > 1`
   (replaces current `self.objective.is_multi_objective`)

5. `metric_names` property: aggregate across all objectives + constraints

6. `metric_name_to_signature` property: aggregate across all objectives + constraints

7. `metric_signatures` property: aggregate across all objectives + constraints

8. `objective_thresholds` property: filter constraints matching any objective metric name

9. `_validate_transformed_optimization_config`: drop the "does not support
   MultiObjective" error. Add validation that objectives don't share metrics
   and that objective metrics aren't constrained.

10. `clone_with_args`: support cloning with `objectives` list

11. Tests: add construction tests for `OptimizationConfig(objectives=[obj1, obj2])`,
    test `is_moo_problem`, test `objective` raises for MOO, test metric aggregation

---

### Diff 2: Update `MultiObjectiveOptimizationConfig` and `PreferenceOptimizationConfig`

**Goal:** Make MOOC use the new `objectives` list internally. Deprecate MOOC.

**Files:**
- `ax/core/optimization_config.py`
- `ax/core/tests/test_optimization_config.py`

**Changes:**

1. `MOOC.__init__`:
   - Accept `objectives: list[Objective] | None = None` kwarg (new path)
   - If legacy `objective` kwarg is used with a comma-separated multi-objective
     expression, decompose into individual objectives
   - If `objective_thresholds` list is provided, resolve each threshold onto
     the matching Objective (store on Objective for now as a temporary attribute,
     to be replaced by `threshold` field in Diff 5)
   - Emit `DeprecationWarning` for the class itself
   - Call `super().__init__(objectives=objectives_list, ...)`

2. `MOOC.objective_thresholds` property: return from stored `_objective_thresholds`
   OR synthesize from objectives (backward compat)

3. `PreferenceOptimizationConfig`:
   - Accept `objectives: list[Objective] | None` kwarg
   - Validate `len(objectives) > 1` instead of `isinstance(objective, MultiObjective)`
   - Deprecation warning for passing `objective` with multi-objective expression

4. Tests: update MOOC and PreferenceOptimizationConfig tests

---

### Diff 3: Restrict `Objective` to single/scalarized expressions

**Goal:** Make `Objective` reject comma-separated (multi-objective) expressions.
Safe because Diffs 1-2 provide the `objectives=[...]` alternative.

**Files:**
- `ax/core/objective.py`
- `ax/core/tests/test_objective.py`
- Any internal callers that construct comma-separated Objectives (update to use
  `OptimizationConfig(objectives=[...])`)

**Changes:**

1. `Objective.__init__`: after parsing the expression, if it contains commas
   (i.e., `parse_objective_expression` returns a tuple), raise `UserInputError`
   with migration guidance to use `OptimizationConfig(objectives=[...])`.

2. Remove `is_multi_objective` property (always False now, no longer meaningful).
   Add a deprecated shim that warns and returns False.

3. Remove `is_single_objective` property (redundant -- it's always `not is_scalarized`).
   Add a deprecated shim.

4. `MultiObjective.__init__`: raise `NotImplementedError` with message:
   "MultiObjective is removed. Use OptimizationConfig(objectives=[...]) instead."

5. Update all internal callers that construct `Objective(expression="acc, -loss")`
   to use `OptimizationConfig(objectives=[Objective("acc"), Objective("-loss")])`.
   Key files:
   - `ax/core/optimization_config.py` (MOOC validation)
   - `ax/core/experiment.py`
   - `ax/adapter/adapter_utils.py`
   - `ax/storage/json_store/decoder.py`
   - `ax/storage/sqa_store/encoder.py` / `decoder.py`

6. Tests: verify comma expressions raise, update multi-objective test construction

---

### Diff 4: Migrate `isinstance(_, MOOC)` checks to `is_moo_problem`

**Goal:** Mechanical replacement of isinstance checks. Large but safe.

**Files:** ~24 files across adapter/, service/, benchmark/, analysis/,
generators/, storage/, early_stopping/, global_stopping/, fb/

**Changes:**

Replace all `isinstance(opt_config, MultiObjectiveOptimizationConfig)` with
`opt_config.is_moo_problem`. Approximately 30 occurrences.

Key files (non-exhaustive):
- `ax/adapter/torch.py` (2 occurrences)
- `ax/adapter/adapter_utils.py` (1)
- `ax/adapter/transforms/objective_as_constraint.py` (2)
- `ax/adapter/transforms/standardize_y.py` (1)
- `ax/adapter/transforms/relativize.py` (1)
- `ax/adapter/transforms/derelativize.py` (1)
- `ax/adapter/transforms/power_transform_y.py` (1)
- `ax/adapter/transforms/stratified_standardize_y.py` (1)
- `ax/adapter/transforms/log_y.py` (1)
- `ax/service/utils/best_point.py` (2)
- `ax/benchmark/benchmark_problem.py` (3)
- `ax/benchmark/benchmark.py` (1)
- `ax/core/experiment.py` (1)
- `ax/storage/json_store/encoders.py` (1)
- `ax/storage/sqa_store/encoder.py` (1)
- `ax/global_stopping/strategies/improvement.py` (1)
- `ax/analysis/plotly/objective_p_feasible_frontier.py` (1)
- `ax/analysis/healthcheck/early_stopping_healthcheck.py` (1)
- `ax/early_stopping/dispatch.py` (1)
- `ax/fb/early_stopping/strategies/multi_objective.py` (1)

Also migrate remaining `isinstance(_, MultiObjective)` checks (~4 in production)
to `objective.is_multi_objective` (which is now deprecated) or the new
`opt_config.is_moo_problem`.

**Consider splitting** into sub-diffs by module if > 500 lines.

---

### Diff 5: Add `threshold` and `relative_threshold` to `Objective`

**Goal:** Co-locate objective thresholds with their objectives.

**Files:**
- `ax/core/objective.py`
- `ax/core/optimization_config.py`
- `ax/core/tests/test_objective.py`
- `ax/core/tests/test_optimization_config.py`
- `ax/storage/json_store/encoders.py`
- `ax/storage/json_store/decoder.py`
- `ax/storage/sqa_store/encoder.py`
- `ax/storage/sqa_store/decoder.py`
- `ax/storage/sqa_store/sqa_classes.py` (if SQA columns needed)

**Changes:**

1. `Objective.__init__` accepts `threshold: float | None = None` and
   `relative_threshold: float | None = None`
   - Store as `self._threshold` and `self._relative_threshold`
   - Properties with getters/setters

2. `OptimizationConfig.objective_thresholds` property: synthesize
   `OutcomeConstraint` objects from each objective's threshold/relative_threshold
   (for downstream compat with adapter layer's `extract_objective_thresholds`)

3. `MOOC.__init__`: when `objective_thresholds` list is provided, resolve each
   `OutcomeConstraint` to the matching `Objective.threshold` (or
   `relative_threshold`). Validate no conflicts.

4. `Objective.clone()`: preserve threshold fields

5. Storage:
   - JSON: add `"threshold"` and `"relative_threshold"` to `objective_to_dict`.
     Decoder: read these fields, defaulting to `None` for old data.
   - SQA: add nullable columns or store in `properties` dict.

6. Tests: construction, serialization round-trip, MOOC threshold resolution

---

### Diff 6: Cleanup -- Remove internal usage of deprecated classes

**Goal:** All internal Ax code uses the new patterns. Deprecated classes remain
as shims for external consumers.

**Files:** Broad -- all files that import/use `MultiObjectiveOptimizationConfig`,
`MultiObjective`, `ScalarizedObjective`, `ObjectiveThreshold`.

**Changes:**

1. Replace all internal construction of `MOOC(...)` with
   `OptimizationConfig(objectives=[...])`

2. Replace all internal construction of `MultiObjective([...])` with
   individual `Objective` instances in a list

3. Replace all internal construction of `ObjectiveThreshold(...)` with
   `Objective(..., threshold=...)` or `OutcomeConstraint(...)`

4. Remove internal imports of deprecated classes (keep re-exports for external compat)

5. Strengthen deprecation warnings (add removal timeline)

6. Clean up dead code paths, unused helper functions

**Consider splitting** into sub-diffs: core, adapter, service, storage, benchmark,
analysis, fb.

---

## Key Design Decisions

1. **`objectives` list on base `OptimizationConfig`** -- not a separate class.
   Multi-objective is a property (`is_moo_problem`), not a type.

2. **`objective` property raises for MOO** -- forces callers to use `objectives`
   for multi-objective, preventing silent bugs from accessing only the first objective.

3. **Thresholds on `Objective`** -- `threshold` (absolute) and `relative_threshold`
   (percent change from status quo). When both are set, the more stringent one
   is used after un-relativization.

4. **Deprecated classes kept as shims** -- `MultiObjective`, `ScalarizedObjective`,
   `ObjectiveThreshold`, `MultiObjectiveOptimizationConfig` remain importable but
   emit deprecation warnings. Internal usage is removed.

5. **Backward-compatible storage** -- old serialized data (without `objectives` list
   or `threshold` fields) deserializes correctly via fallback paths.

## Risks and Mitigations

- **Large surface area:** ~70 files reference these classes. Mitigated by splitting
  into 6+ focused diffs and running full test suites.
- **Storage backward compat:** Old experiments must still load. Mitigated by
  keeping decoder fallback paths and testing with existing fixtures.
- **External consumers:** Meta-internal code outside ax/ may use deprecated classes.
  Mitigated by keeping shims and using deprecation warnings before removal.
