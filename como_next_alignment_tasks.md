# COMO/DeepCOMO: Next Implementation Tasks for Paper Alignment

Generated: 2026-06-12

Scope of this plan: tighten alignment with the DeepCOMO paper for the non-REINVENT parts of the repository. The exact ChEMBL26 diverse R-group library is intentionally left as a later task, but the interfaces below should make it easy to plug in once available.

## Current status assumption

The repository is now closer than the initial version: it has a shared decomposition layer, a paper-mode close-in generator, a context-fixed Free-Wilson generator, score repeats, and a paper-style SVR class. The remaining work is mostly integration, strict paper-mode behavior, reporting, validation, and documentation.

## Paper anchors to preserve

Use these as the implementation contract:

1. **Diagnostic scores**: C, D, S, and P follow the paper equations. P is based on mean pairwise potency differences among EAs whose neighborhoods overlap through VAs. It is not inherently limited to `[0, 1]` unless the input potency scale/range makes it so.
2. **Chemical reference space**: the paper uses seven lead-optimization-relevant physicochemical descriptors calculated with RDKit.
3. **NBH radius**: the paper treats the neighborhood radius as a hyperparameter that may be fine-tuned by VA population and reference space. Do not describe the repo's automatic kNN radius as the paper default unless a source is found.
4. **Close-in VAs**: enumerate the AS core by randomly decorating all substitution sites with EA-derived substituents, with H/no-substituent sampled according to an AS-specific substitution probability, and restrict generated VA size to the EA size range.
5. **FW VAs**: derive from MMP/FW neighborhoods, contain only R-groups already present in the AS, and represent missing corners of qualifying local Free-Wilson matrices.
6. **Diagnostic reporting**: the example paper diagnostics used 1000 close-in VAs for 10 independent score calculations and reported mean +/- standard deviation.
7. **SVR potency model**: use folded 2048-bit ECFP4, a Tanimoto precomputed kernel, scikit-learn SVR, three-fold double cross-validation, ChEMBL target actives outside the AS plus 50% of the EAs for training, and the remaining 50% of EAs as external validation.
8. **AS1/AS2 sanity targets**: if exact curated data are available, verify the counts and scores against the published table: 219/158 EAs, 4/3 substitution sites, 70/133 unique close-in substituents, 907/3167 FW VAs, 183/45 EAs in FW NBHs, and the reported C/D/S/P means.

---

## Priority 0: Decide and expose `paper_mode` consistently

### Problem

The code appears to contain paper-aligned components, but they are not yet the single, explicit pipeline path. Users should not have to know which internal class uses paper behavior and which still uses legacy behavior.

### Implement

Add a single pipeline flag that controls all relevant behavior:

```python
score_series(
    ...,
    paper_mode: bool = False,
    random_state: int | None = 42,
)
```

CLI:

```text
--paper-mode
--random-state 42
```

When `paper_mode=True`:

- call `decompose_series(..., paper_mode=True)` exactly once;
- pass the resulting `SeriesDecomposition` object to close-in and FW generation;
- use exact EA heavy-atom-count range unless explicitly overridden;
- use close-in VAs as the diagnostic VA population;
- use `score_repeats=10` and `va_n=1000` by default for diagnostic scoring;
- require explicit warning if the core has no declared exit vectors and sites are inferred;
- require explicit warning if `nbh_radius="auto"` is used for a benchmark-style run.

### Acceptance criteria

- One integration test calls `score_series(..., paper_mode=True)` and verifies that decomposition is performed once and reused.
- README has a clear table: legacy/convenience mode vs paper mode.
- Generated output records `paper_mode`, `random_state`, `core_smiles`, `site_list`, `hac_range`, and `nbh_radius_source`.

---

## Priority 1: Wire `PaperSVRPredictor` into the main pipeline

### Problem

The paper-style SVR exists, but the main scoring path still needs to use it end-to-end. The CLI/API should make it impossible to request paper SVR without also supplying or validating the external ChEMBL-active data.

### Implement API changes

Add to `score_series`:

```python
svr_mode: Literal["legacy", "paper", "none"] = "legacy"
external_actives_csv: str | Path | None = None
external_smiles_col: str = "smiles"
external_activity_col: str = "pActivity"
ea_train_fraction: float = 0.5
outer_folds: int = 3
inner_folds: int = 3
svr_c_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0)
svr_epsilon_grid: Sequence[float] = (0.01, 0.05, 0.1, 0.2)
```

Add CLI flags:

```text
--svr-mode legacy|paper|none
--external-actives FILE
--external-smiles-col smiles
--external-activity-col pActivity
--ea-train-fraction 0.5
--outer-folds 3
--inner-folds 3
--svr-c-grid 0.1,1,10,100
--svr-epsilon-grid 0.01,0.05,0.1,0.2
```

### Implement behavior

For `svr_mode="paper"`:

1. Load external actives.
2. Canonicalize all SMILES.
3. Remove invalid molecules.
4. Remove any external molecule whose canonical SMILES appears in the AS.
5. Check activity values are finite and on the same potency scale as EAs.
6. Split EAs into train and external-validation halves with `random_state`.
7. Build the training pool as `external_non_as_actives + ea_train_half`.
8. Run three-fold double CV on the training pool.
9. Fit final production model(s) on the full training pool.
10. Predict the held-out EA validation half.
11. Predict all VA populations.

### Fix likely NumPy bug

Avoid patterns like:

```python
external_activities or []
```

because NumPy arrays do not have a scalar truth value. Use explicit `None` checks:

```python
if external_smiles is not None and len(external_smiles) > 0:
    if external_activities is None:
        raise ValueError("external_activities must be provided with external_smiles")
    ext_acts = np.asarray(external_activities, dtype=np.float64)
else:
    ext_acts = np.asarray([], dtype=np.float64)
```

### Output files

Add:

```text
svr_training_summary.csv
svr_outer_folds.csv
svr_external_validation.csv
svr_predictions.csv
```

Minimum columns:

`svr_training_summary.csv`

```text
svr_mode,n_external_loaded,n_external_valid,n_external_after_as_exclusion,n_ea_train,n_ea_validation,fp_type,kernel,outer_folds,inner_folds,random_state
```

`svr_outer_folds.csv`

```text
fold,best_C,best_epsilon,outer_r2,outer_mae,outer_rmse,n_train,n_test
```

`svr_external_validation.csv`

```text
smiles,observed_pActivity,predicted_pActivity,prediction_std,absolute_error,in_ea_train,in_ea_validation
```

### Acceptance criteria

- Requesting `--svr-mode paper` without `--external-actives` raises a clear error, unless an explicit `allow_no_external=True` testing flag is set.
- Paper SVR path predicts held-out EAs and all VAs.
- No external validation EA leaks into the SVR training pool.
- Invalid external molecules are reported, not silently ignored.
- Unit test passes external activities as both a Python list and a NumPy array.

---

## Priority 2: Make `decompose_series` strict enough for paper mode

### Problem

The decomposition object is the right architecture, but paper-mode behavior should be stricter and more transparent.

### Implement

Add parameters:

```python
decompose_series(
    core_smiles: str,
    ea_smiles: list[str],
    ea_activities: list[float] | None = None,
    paper_mode: bool = False,
    hac_padding: int | None = None,
    require_exit_vectors: bool = False,
)
```

Behavior:

- In `paper_mode=True`, default `hac_padding=0`.
- In legacy mode, keep current padding only if desired, for example `hac_padding=3`.
- If `paper_mode=True` and `require_exit_vectors=True`, raise an error when the core has no `*` dummy atoms.
- If `paper_mode=True` and exit vectors exist, reject EAs with substituents outside declared sites.
- Preserve `None` for H/no-substituent at all declared sites.
- Keep a complete rejection report.

### Important detail

The paper says H decoration is based on an **AS-specific substitution probability**. That sounds like a series-level probability, not a site-specific probability. Keep the default as global probability:

```python
probability_mode: Literal["global", "site_specific"] = "global"
```

Allow `site_specific` only as an optional sensitivity analysis.

### Output file

Add:

```text
decomposition_report.csv
```

Minimum columns:

```text
input_smiles,canonical_smiles,status,rejection_reason,activity,heavy_atom_count,site_1,site_2,...
```

Also add a JSON summary:

```text
decomposition_summary.json
```

with:

```json
{
  "n_input": 0,
  "n_decomposed": 0,
  "n_rejected": 0,
  "site_list": [],
  "n_sites": 0,
  "unique_substituent_count": 0,
  "substitution_probability_global": 0.0,
  "site_substitution_probability": {},
  "ea_hac_range": [0, 0],
  "paper_mode": true
}
```

### Acceptance criteria

- In paper mode, `ea_hac_range == (min_ea_hac, max_ea_hac)` unless overridden.
- Molecules with off-exit-vector substituents are rejected and reported.
- Core without exit vectors triggers either a warning or an error depending on `require_exit_vectors`.
- The same decomposition object feeds close-in, FW, scoring, and later diverse generation.

---

## Priority 3: Convert Free-Wilson to consume `SeriesDecomposition`

### Problem

The FW generator now has the right context-fixed logic, but it should not call legacy replacement-core helpers internally. It should consume the exact same decomposition object as close-in scoring.

### Implement API change

Support both paths for compatibility:

```python
class FreeWilsonVAGenerator:
    def generate_from_decomposition(
        self,
        decomp: SeriesDecomposition,
        n: int | None = None,
        hac_range: tuple[int, int] | None = None,
        paper_mode: bool = False,
    ) -> list[str]:
        ...

    def generate(...):
        # legacy shim: calls decompose_series then generate_from_decomposition
```

### FW matrix discovery contract

For each pair of varying sites `(site_a, site_b)`:

1. Define `rest_sites = all_sites - {site_a, site_b}`.
2. Group EAs by the complete fixed context over `rest_sites`.
3. Within each context, build a two-dimensional matrix over observed levels at `site_a` and `site_b`, including `None` for H/no-substituent when present.
4. If exactly three corners exist, create the missing FW VA.
5. If all four corners exist, create retrospective FW EA predictions by holding out each corner.
6. Count EAs in FW neighborhoods only if they participate in one of these qualifying matrices.

### Fix duplicate-corner behavior

Do not use the first EA at a matrix corner arbitrarily. If multiple EAs share the same site map/corner:

Option A, preferred for stability:

```python
corner_activity = mean(activity values for all EAs in that corner)
corner_activity_std = std(activity values for all EAs in that corner)
corner_n = number of EAs in that corner
```

Option B, optional advanced mode:

enumerate all combinations of one EA from each present corner and average the resulting FW predictions.

Record which option is used.

### Add output files

```text
fw_candidates.csv
fw_neighborhoods.csv
fw_ea_validation.csv
```

`fw_candidates.csv`:

```text
smiles,fw_pred_mean,fw_pred_std,fw_pred_n,varying_sites,n_supporting_neighborhoods,site_map_json
```

`fw_neighborhoods.csv`:

```text
neighborhood_id,site_a,site_b,context_json,present_corners_json,missing_corner_json,candidate_smiles,prediction
```

`fw_ea_validation.csv`:

```text
smiles,observed_pActivity,fw_pred_mean,fw_pred_std,fw_pred_n,absolute_error
```

### `n` handling

In paper mode, FW count should be determined by the series, not by a user-specified sample size. Implement:

```text
If paper_mode=True and n is provided, ignore n and log that FW VA count is series-determined.
```

### Acceptance criteria

- FW candidates are reproducibly generated from a `SeriesDecomposition` object.
- FW VAs use only substituents from `decomp.site_pools` or `None`.
- Context-mismatched 2-by-2 examples do not generate candidates.
- H/no-substituent can be a valid matrix level.
- Duplicate corner entries do not cause arbitrary first-row predictions.
- Retrospective FW EA validation metrics are available when all-four-corner matrices exist.

---

## Priority 4: Tighten close-in paper-mode generation

### Problem

Close-in generation is much improved, but it should be locked down as the canonical diagnostic VA generator.

### Implement

Add/ensure:

```python
class CloseInVAGenerator:
    def generate_from_decomposition(
        self,
        decomp: SeriesDecomposition,
        n: int = 1000,
        random_state: int | None = None,
        paper_mode: bool = False,
        probability_mode: Literal["global", "site_specific"] = "global",
        max_attempts: int | None = None,
    ) -> list[str]:
        ...
```

Sampling behavior in paper mode:

- Iterate over all `decomp.site_list`.
- At each site, choose H/no-substituent with probability `1 - p_sub_global` by default.
- Choose an organic substituent from that site's observed pool with probability `p_sub_global`.
- Do not choose from an empty site pool.
- Assemble core + chosen site fragments.
- Canonicalize.
- Reject invalid molecules.
- Reject EAs.
- Reject molecules outside exact `decomp.ea_hac_range`.
- Continue until `n` unique VAs or `max_attempts` is reached.

### Add sampling report

```text
close_in_generation_report.json
```

with:

```json
{
  "n_requested": 1000,
  "n_generated": 1000,
  "n_attempts": 0,
  "n_invalid": 0,
  "n_duplicate": 0,
  "n_existing_analog": 0,
  "n_outside_hac_range": 0,
  "substitution_probability": 0.0,
  "probability_mode": "global",
  "random_state": 42
}
```

### Acceptance criteria

- Running with the same seed produces the same VA set.
- Running with different seeds produces different VA sets but similar aggregate counts.
- No generated close-in VA is an EA.
- All generated close-in VAs are within the EA HAC range in paper mode.
- Sites with no observed substituent pool are treated as H-only.

---

## Priority 5: Make diagnostic scoring exactly reproducible

### Problem

The paper reports diagnostics as repeated calculations using close-in VA samples. The repo should make this one command and write the per-repeat details.

### Implement

Add:

```python
score_with_repeats(
    decomp: SeriesDecomposition,
    repeats: int = 10,
    va_n: int = 1000,
    nbh_radius: float | None = None,
    random_state: int | None = 42,
    reference_space: Literal["rdkit_7d"] = "rdkit_7d",
) -> ScoreRepeatResult
```

Behavior:

- For each repeat, generate or sample 1000 close-in VAs using independent but reproducible seeds.
- Compute seven-dimensional RDKit descriptors for EAs and VAs.
- Normalize/project EAs and VAs together using one scaler per repeat.
- Build EA neighborhoods with the chosen radius.
- Compute C, D, S, and P.
- Store mean and standard deviation across repeats.

### NBH radius behavior

Add explicit radius provenance:

```python
radius_mode: Literal["explicit", "auto"]
radius_value: float
radius_warning: str | None
```

In paper-mode benchmark runs, prefer explicit radius. If auto is used, include a warning:

```text
The paper treats NBH radius as a tunable hyperparameter. This run used the repository's auto-radius heuristic, so exact paper reproduction is not claimed.
```

### Fix score tests

Update tests so P is checked as finite and non-negative, not constrained to `<= 1`:

```python
assert np.isfinite(P)
assert P >= 0.0
```

### Output files

```text
score_repeats.csv
scores.csv
```

`score_repeats.csv`:

```text
repeat,seed,n_va,n_ea,nbh_radius,C,D,S,P,d_mean,n_covered_va,n_overlap_va
```

`scores.csv` should include:

```text
C_mean,C_std,D_mean,D_std,S_mean,S_std,P_mean,P_std,nbh_radius,nbh_radius_mode,repeats,va_n
```

### Acceptance criteria

- With `paper_mode=True`, default diagnostic output uses 10 repeats x 1000 close-in VAs.
- Per-repeat rows and final mean/std rows are written.
- P-score tests no longer assume an upper bound of 1.
- The run metadata makes radius choice auditable.

---

## Priority 6: Clarify stage labels and thresholds

### Problem

The repo exposes fixed stage labels and thresholds. The paper uses diagnostic scores to categorize the example ASs, but the repo's fixed threshold rule should not be presented as an authoritative DeepCOMO rule unless sourced.

### Implement

Rename or document threshold-based labels as a convenience interpretation:

```python
assign_stage(S, P, s_threshold=0.4, p_threshold=0.5, mode="heuristic")
```

Output both:

```text
stage_heuristic
stage_thresholds
```

Do not write simply `stage=late` without making the threshold rule clear.

### Documentation wording

Use wording like:

```text
The repository provides configurable heuristic stage labels based on S/P thresholds. The paper's AS1/AS2 stage assignments are reproduced only if the paper benchmark data and scoring protocol reproduce the published diagnostic values.
```

### Acceptance criteria

- README does not imply that the repo's default thresholds are explicitly defined by the paper.
- Output files include the thresholds used.
- Tests cover custom thresholds.

---

## Priority 7: Add AS1/AS2 benchmark harness

### Problem

Without the exact paper data and benchmark harness, the repository cannot honestly claim faithful reproduction.

### Implement structure

```text
benchmarks/
  deepcomo_as1_as2/
    README.md
    run_benchmark.py
    expected_values.yml
    data/
      as1.csv              # optional; only if redistribution is allowed
      as2.csv
      as1_external_actives.csv
      as2_external_actives.csv
```

`expected_values.yml`:

```yaml
as1:
  target: "P2X purinoreceptor 3"
  chembl_target_id: 2998
  potency_type: "IC50"
  n_ea: 219
  n_sites: 4
  n_unique_substituents: 70
  n_fw_va: 907
  n_ea_in_fw_nbh: 183
  scores:
    C: {mean: 0.43, std: 0.01, tolerance: 0.05}
    D: {mean: 0.90, std: 0.00, tolerance: 0.05}
    S: {mean: 0.58, std: 0.01, tolerance: 0.05}
    P: {mean: 0.55, std: 0.03, tolerance: 0.10}
as2:
  target: "Sodium channel protein type IX alpha subunit"
  chembl_target_id: 4296
  potency_type: "IC50"
  n_ea: 158
  n_sites: 3
  n_unique_substituents: 133
  n_fw_va: 3167
  n_ea_in_fw_nbh: 45
  scores:
    C: {mean: 0.18, std: 0.02, tolerance: 0.05}
    D: {mean: 0.73, std: 0.03, tolerance: 0.05}
    S: {mean: 0.29, std: 0.02, tolerance: 0.05}
    P: {mean: 0.95, std: 0.05, tolerance: 0.15}
```

### Benchmark script behavior

`run_benchmark.py` should:

1. Load AS data.
2. Run strict `paper_mode=True` with explicit core and exit vectors.
3. Verify decomposition counts.
4. Generate close-in diagnostics with 10 x 1000 VAs.
5. Generate FW VAs and FW EA validation metrics.
6. Optionally run paper SVR if external actives are present.
7. Write `benchmark_report.md` with pass/fail checks.

### Acceptance criteria

- If exact data are unavailable, benchmark README states what is missing and how to supply it.
- If exact data are supplied, benchmark reports pass/fail for all published counts and scores.
- Repository claim is downgraded from "faithful reproduction" to "paper-aligned implementation" until benchmark passes.

---

## Priority 8: Update README and user-facing claims

### Problem

The README should match the implementation and avoid overclaiming.

### Implement README sections

Add:

```text
## Reproduction status
```

Suggested wording:

```text
This repository implements a paper-aligned COMO workflow for diagnostic scoring, close-in VA generation, Free-Wilson VA generation, and SVR potency prediction. The REINVENT transfer-learning generator and the exact ChEMBL26 diverse R-group pool are not included. Exact reproduction of AS1/AS2 paper numbers requires the curated paper data and benchmark script.
```

Add:

```text
## Paper mode vs legacy mode
```

Table:

| Feature | Legacy/convenience mode | Paper mode |
|---|---|---|
| Decomposition | sites inferred | declared exit vectors preferred |
| HAC range | optionally padded | exact EA range |
| Close-in | may use legacy enumeration | random H-aware generation |
| Diagnostic scoring | one run unless requested | 1000 VAs x 10 repeats |
| NBH radius | auto allowed | explicit radius recommended |
| SVR | EA-only CV | external actives + 50% EA training + 50% EA validation |
| FW | compatibility path | decomposition-backed context-fixed matrices |

Add CLI examples:

```bash
python -m como \
  --series as1.csv \
  --core "[*]...[*]" \
  --paper-mode \
  --va close_in free_wilson \
  --va-n 1000 \
  --score-repeats 10 \
  --nbh-radius 0.35 \
  --svr-mode paper \
  --external-actives as1_external_actives.csv \
  --output results/as1_paper_mode
```

### Acceptance criteria

- README no longer describes paper-mode close-in as deterministic combinatorial enumeration.
- README explains that the bundled diverse fragments are not the paper's 44,636 ChEMBL26 R-group pool.
- README does not imply REINVENT transfer learning is implemented unless it is.
- API documentation includes `paper_mode`, `svr_mode`, `score_repeats`, and `external_actives`.

---

## Priority 9: Add output provenance and reproducibility metadata

### Problem

COMO scores depend on core choice, site definitions, VA sampling, radius, descriptors, and random seeds. Paper-aligned runs need enough metadata to be auditable.

### Implement

Write a `run_metadata.json` file for every pipeline run:

```json
{
  "como_version": "",
  "git_commit": "",
  "paper_mode": true,
  "random_state": 42,
  "input_series_file": "",
  "core_input": "",
  "core_used_after_exit_vector_stripping": "",
  "site_list": [],
  "n_ea_input": 0,
  "n_ea_decomposed": 0,
  "n_ea_rejected": 0,
  "va_strategies": [],
  "score_repeats": 10,
  "va_n": 1000,
  "nbh_radius": 0.0,
  "nbh_radius_mode": "explicit",
  "descriptor_space": "rdkit_7d",
  "svr_mode": "paper",
  "external_actives_file": "",
  "notes": []
}
```

### Acceptance criteria

- Every output directory contains `run_metadata.json`.
- Every CSV has enough IDs to link back to metadata.
- Results are exactly repeatable when run with the same seed, dependency versions, and input files.

---

## Priority 10: Prepare but do not finalize diverse generation

### Problem

The exact ChEMBL26 R-group derivation is not fully specified in the DeepCOMO paper. The current diverse generator can remain, but it must not be confused with the paper's 44,636-fragment external pool.

### Implement now

- Rename docs for the current implementation to `diverse_demo` or clearly mark it as a placeholder.
- Add a loader interface for future external R-group files:

```python
load_rgroup_library(
    path: str | Path,
    smiles_col: str = "smiles",
    attachment_format: Literal["dummy_atom", "atom_map"] = "dummy_atom",
    max_heavy_atoms: int = 13,
) -> list[RGroup]
```

- Require one explicit attachment point per R-group.
- Do not attach arbitrary atom 0.
- Exclude R-groups already present in the EA series.
- Use the same scaffold-assembly and H-aware sampling machinery as close-in.

### Acceptance criteria

- README states that diverse generation is not paper-exact yet.
- Loader rejects fragments without exactly one attachment point.
- Diverse paper-mode path can accept a future ChEMBL26-derived R-group file without code changes.

---

## Priority 11: Test plan

### Unit tests

Add or update:

```text
tests/test_decomposition_paper_mode.py
tests/test_close_in_paper_mode.py
tests/test_free_wilson_decomposition_backed.py
tests/test_free_wilson_duplicate_corners.py
tests/test_scoring_repeats.py
tests/test_svr_paper_protocol.py
tests/test_cli_paper_mode.py
tests/test_output_metadata.py
```

Specific tests:

1. **Exact HAC range**: paper mode uses `(min_hac, max_hac)`.
2. **Exit-vector rejection**: off-site substituted EA is rejected.
3. **Close-in H sampling**: at least one generated VA leaves a site as H when `p_sub < 1`.
4. **Close-in no EA leakage**: generated VAs exclude canonical EA SMILES.
5. **FW context isolation**: no candidate is produced when the fourth corner would require a different fixed context.
6. **FW H level**: `None` is accepted as a matrix level.
7. **FW duplicate corner**: duplicate corner activities are averaged or combinatorially handled, never first-row arbitrary.
8. **FW validation**: all-four-corner matrix yields retrospective FW EA predictions.
9. **P score bound**: P is finite and non-negative but not required to be <= 1.
10. **SVR no leakage**: held-out validation EAs are not in the training pool.
11. **SVR NumPy inputs**: external activities can be NumPy arrays.
12. **CLI paper mode**: writes all expected outputs.

### Integration tests

Create a small synthetic AS with known substitutions and activities:

- 3 sites.
- one missing FW corner with known additivity prediction.
- one all-four-corner matrix for FW EA validation.
- enough close-in combinations to sample >100 VAs.

Run:

```bash
python -m pytest tests/test_cli_paper_mode.py -v
```

### Acceptance criteria

- All tests pass locally and in CI.
- Tests cover both legacy and paper modes where backwards compatibility matters.

---

## Priority 12: Suggested implementation order

1. Add/standardize `paper_mode` and `random_state` in `score_series` and CLI.
2. Fix paper-mode HAC range in `decompose_series`.
3. Add decomposition reports and run metadata.
4. Convert close-in to `generate_from_decomposition` if not already done.
5. Convert FW to `generate_from_decomposition` and remove legacy helper dependence from the paper path.
6. Fix FW duplicate-corner handling.
7. Add FW candidate/neighborhood/validation output files.
8. Wire `PaperSVRPredictor` into `score_series` and CLI.
9. Fix NumPy truth-value bug in SVR external activity handling.
10. Add paper SVR output files and no-leakage tests.
11. Make score repeats the default paper diagnostic path.
12. Fix P-score tests and radius provenance.
13. Update README and examples.
14. Add AS1/AS2 benchmark harness with expected values.
15. Mark diverse as placeholder until the external R-group pool is solved.

---

## Definition of "aligned enough" for this pass

The repo can reasonably claim **paper-aligned COMO components excluding REINVENT and exact diverse pool** when all of the following are true:

- `paper_mode=True` runs decomposition, close-in diagnostics, FW generation, and optional paper SVR through one coherent pipeline.
- Close-in diagnostics use 1000 VAs x 10 repeats and report mean/std.
- FW VAs are context-fixed missing Free-Wilson matrix corners generated from the same decomposition used by close-in.
- FW EA validation predictions and metrics are reported when qualifying all-four-corner matrices exist.
- Paper SVR uses external non-AS target actives plus 50% EA training and held-out 50% EA validation.
- Run outputs include full provenance and enough intermediate files to audit counts.
- README clearly distinguishes implemented paper-aligned COMO components from missing DeepCOMO REINVENT transfer learning and non-exact diverse R-group generation.
- AS1/AS2 benchmark harness exists, even if exact data must be supplied by the user.

Do **not** claim full faithful DeepCOMO reproduction until the REINVENT transfer-learning generator, the exact or author-validated diverse R-group pool, and AS1/AS2 numerical benchmark reproduction are all present.
