# COMO/DeepCOMO Alignment Plan

Scope: align the existing repository with the DeepCOMO paper for the parts that do not require REINVENT transfer learning and do not depend on the final ChEMBL26 diverse R-group library.

In scope:

- Series decomposition and reusable R-group/site representation
- Close-in virtual analog (VA) generation
- Free-Wilson (FW) VA generation and local FW potency prediction
- Global SVR potency model
- Diagnostic scoring protocol and reproducibility hooks
- CLI/API changes, tests, and validation benchmarks

Out of scope for this pass:

- REINVENT 2.0 / transfer-learning sampled VAs
- Exact diverse ChEMBL26 external R-group pool, except for keeping interfaces ready for it

---

## 1. Target behavior from the paper

### 1.1 Close-in and diverse scaffold enumeration

The paper describes scaffold enumeration as decorating all substitution sites on the analog-series core with randomly selected terminal fragments according to predefined reactions. For multi-site series, it controls over-large products by restricting VA size to the existing-analog size range and by randomly replacing one or more organic substituents with hydrogen according to an analog-series-specific substitution probability.

Close-in VAs use only substituents extracted from the existing analogs (EAs). Diverse VAs use an external R-group pool. For this pass, implement the shared enumeration machinery and close-in mode; leave diverse pool loading as a later plug-in.

Paper sanity targets for AS1/AS2 if the exact paper data are available:

| Quantity | AS1 | AS2 |
|---|---:|---:|
| EAs | 219 | 158 |
| Substitution sites | 4 | 3 |
| Unique close-in substituents | 70 | 133 |
| FW VAs | 907 | 3167 |
| EAs in FW neighborhoods | 183 | 45 |
| Close-in diagnostic C | 0.43 +/- 0.01 | 0.18 +/- 0.02 |
| Close-in diagnostic D | 0.90 +/- 0.00 | 0.73 +/- 0.03 |
| Close-in diagnostic S | 0.58 +/- 0.01 | 0.29 +/- 0.02 |
| Close-in diagnostic P | 0.55 +/- 0.03 | 0.95 +/- 0.05 |

### 1.2 FW generation

The paper treats FW VAs as late-stage candidates generated from matched molecular pair (MMP) networks. FW VAs should be a subset of close-in analog space because they contain only R-groups already present in the analog series. Qualifying FW neighborhoods are local 2-by-2 substituent matrices: three observed EA corners can predict a missing fourth VA corner by Free-Wilson additivity. Four observed corners can be used for retrospective FW EA validation.

### 1.3 Global SVR potency prediction

The paper's global potency model uses:

- Folded 2048-bit ECFP4 fingerprints
- Tanimoto precomputed kernel
- Support vector regression in scikit-learn
- Three-fold double cross-validation
- Training compounds = ChEMBL target actives outside the analog series + 50% of the EAs
- External validation = remaining 50% of EAs
- Example external ChEMBL counts: 517 for AS1 and 1135 for AS2

---

## 2. Main gaps to close in the current repo

### 2.1 Close-in generator

Current behavior to replace:

- Uses deterministic `itertools.product(*pool_lists)` enumeration.
- Uses only sites observed in at least one extracted fragment, unless an explicit exit-vector core is supplied.
- Does not model H/no-substituent as a first-class site state.
- Returns the first `n` valid products, which biases VA populations toward sorted fragment order.

Target behavior:

- In paper mode, all declared substitution sites are used.
- Each site has either an organic substituent or H/no-substituent.
- Organic-vs-H sampling follows an analog-series-level substitution probability.
- Product generation is random and seedable.
- Product size is filtered to the EA heavy-atom-count range.
- EAs are excluded from the returned VA population.

### 2.2 FW generator

Current behavior to replace:

- Detects 2-by-2 submatrices over two sites without grouping by the complete fixed context at all other sites.
- Does not treat H/no-substituent consistently as a possible FW level.
- Counts EAs in FW neighborhoods as any EA with MMP graph degree greater than zero, which is broader than qualifying FW neighborhoods.
- Does not fully expose local FW validation predictions for FW EAs.

Target behavior:

- Build FW matrices separately for each pair of varying sites and each fixed context of all remaining sites.
- Allow `None`/H as a level when it is observed in the series.
- Generate missing corners only when exactly three EA corners exist in a fixed-context 2-by-2 matrix.
- Generate retrospective FW predictions when all four EA corners exist.
- Count EAs in FW neighborhoods only when they participate in at least one qualifying FW neighborhood.
- Store all local predictions per candidate, then report mean, standard deviation, and number of supporting local models.

### 2.3 SVR potency model

Current behavior to replace:

- Fits and cross-validates only on the provided EA series.
- Uses ordinary 3-fold K-fold CV, not three-fold double CV.
- Does not implement the paper's external ChEMBL-active + 50% EA training protocol.
- Does not keep the external validation half of EAs separate.

Target behavior:

- Provide a paper-protocol SVR mode while keeping the old EA-only mode as a convenience baseline.
- Implement nested three-fold double CV on the training pool.
- Use the held-out 50% EA set only for external validation.
- Predict all VA populations with the trained model or model ensemble.

---

## 3. Proposed repository structure

```text
como/
  series/
    __init__.py
    standardize.py          # molecule cleanup, canonical SMILES, activity alignment
    decomposition.py        # core matching, R-group extraction, SeriesDecomposition
    assembly.py             # assemble core + site map into a molecule
    schema.py               # dataclasses and typed records
  analogs/
    close_in.py             # paper-mode close-in random enumeration
    free_wilson.py          # context-fixed FW neighborhoods and predictions
    diverse.py              # keep API; later plug in real R-group pool
  potency/
    __init__.py
    fingerprints.py         # ECFP4 and Tanimoto matrices
    svr.py                  # paper-protocol SVR and legacy EA-only SVR
    fw_local.py             # optional helpers for FW validation metrics
  diagnostics/
    scoring_protocol.py     # 1000 VA x 10 repeats; radius handling
  tests/
    test_series_decomposition.py
    test_assembly.py
    test_close_in_paper_mode.py
    test_free_wilson_context.py
    test_free_wilson_predictions.py
    test_svr_paper_protocol.py
    test_scoring_protocol.py
```

Keep public imports backwards compatible where possible:

```python
from como.analogs.close_in import CloseInVAGenerator
from como.analogs.free_wilson import FreeWilsonVAGenerator
from como.potency import PaperSVRPredictor, SVRPredictor
```

---

## 4. Data model

Create one shared decomposition object and make close-in, FW, scoring, and SVR consume it.

```python
@dataclass(frozen=True)
class EARecord:
    input_smiles: str
    canonical_smiles: str
    activity: float
    site_map: dict[int, str | None]   # site -> fragment SMILES, None = H/no substituent
    heavy_atom_count: int

@dataclass(frozen=True)
class SeriesDecomposition:
    core_smiles: str
    core_mol: Chem.Mol
    site_list: tuple[int, ...]
    ea_records: tuple[EARecord, ...]
    ea_canonical_set: frozenset[str]
    site_pools: dict[int, frozenset[str]]
    unique_substituents: frozenset[str]
    substitution_probability: float
    site_substitution_probability: dict[int, float]
    ea_hac_range: tuple[int, int]
    rejected_records: tuple[RejectedRecord, ...]
```

Implementation rules:

- `site_map[site] = None` means that the site is H/no-substituent for that EA.
- `site_pools[site]` contains only organic substituent fragments, not `None`.
- `unique_substituents` excludes `None`.
- The default paper-mode substitution probability is:

```python
p_sub = number_of_non_null_site_entries / (number_of_EAs * number_of_sites)
```

- Also compute per-site probabilities for diagnostics and optional non-paper experiments, but use global `p_sub` by default in paper mode.

---

## 5. Phase 0: add explicit alignment modes

### Tasks

- [ ] Add a `paper_mode: bool = False` flag to the high-level pipeline.
- [ ] In paper mode, require an explicit core with exit-vector information or an explicit site list.
- [ ] Keep legacy behavior available, but label it as `legacy` or `convenience`, not `paper`.
- [ ] Add `random_state` to all generators and scoring protocols.
- [ ] Add `metadata.json` output with all paper-mode settings and random seeds.

### Acceptance criteria

- [ ] Existing tests still pass in legacy mode.
- [ ] Paper mode fails fast with a helpful error when no usable core/site definition is supplied.
- [ ] Runs are reproducible with the same random seed.

---

## 6. Phase 1: rebuild decomposition and assembly

### 6.1 Core and site handling

Preferred paper-mode input:

```text
core_smiles = core scaffold with explicit labelled exit vectors
```

Examples of acceptable representations:

- Dummy atoms attached to core atoms, with atom-map numbers used as site labels.
- A plain core plus a separate site-list argument.

Plan:

- [ ] Parse core.
- [ ] If dummy exit vectors are present, strip them before `ReplaceCore`, but preserve the mapping from dummy label to core attachment atom.
- [ ] Use declared sites as `site_list` in paper mode.
- [ ] If no dummy sites are provided in legacy mode, infer sites from observed decomposition as the current code does.
- [ ] Store a mapping of `site_id -> core_atom_idx` separately from atom indices so site IDs remain stable after sanitization and canonicalization.

### 6.2 EA decomposition

Plan:

- [ ] Canonicalize all EAs before activity alignment.
- [ ] For each EA, run core matching and R-group extraction.
- [ ] Reject molecules that do not contain the full core.
- [ ] Reject molecules whose substituents occur outside declared exit vectors in paper mode.
- [ ] Fill all missing declared sites with `None`.
- [ ] Store rejected molecules and reasons rather than silently skipping.

### 6.3 Fragment representation

Current code marks every fragment attachment atom with atom-map `:1`. Replace this with a site-specific representation.

Target fragment format:

```text
site_id -> fragment with exactly one dummy or attachment marker
```

Recommended internal representation:

```python
@dataclass(frozen=True)
class SiteFragment:
    site_id: int
    fragment_smiles: str       # canonical, attachment-normalized
    attachment_atom_idx: int | None
    heavy_atom_count: int
```

Rules:

- [ ] Do not attach at arbitrary atom 0.
- [ ] Do not reuse one global atom-map number for every fragment in persistent storage.
- [ ] Canonicalize fragments after normalizing the attachment marker.
- [ ] Deduplicate fragments per site and globally.
- [ ] Treat H/no-substituent as `None`, not as a fake fragment.

### 6.4 Assembly

Plan:

- [ ] Create `assemble_series_member(decomp, site_map)`.
- [ ] For each site with `None`, leave the core site undecorated so RDKit assigns implicit H where chemically valid.
- [ ] For each site with a fragment, connect the fragment attachment atom to the correct core atom.
- [ ] Sanitize the molecule.
- [ ] Canonicalize the assembled molecule.
- [ ] Return both SMILES and metadata, or a typed failure reason.

### Acceptance criteria

- [ ] Every decomposed EA can be reassembled to the same canonical SMILES, except for documented cases involving salts/stereo/tautomer standardization.
- [ ] Site maps contain every declared site.
- [ ] Missing sites are represented as `None`.
- [ ] Unique substituent counts are computed from organic fragments only.

---

## 7. Phase 2: close-in VA generation

### 7.1 API

```python
class CloseInVAGenerator:
    def generate(
        self,
        decomp: SeriesDecomposition,
        n: int,
        random_state: int | None = None,
        max_attempts: int = 1_000_000,
        paper_mode: bool = True,
        exclude_eas: bool = True,
        return_metadata: bool = True,
    ) -> list[VARecord]:
        ...
```

### 7.2 Algorithm

Paper-mode random generation:

```python
rng = np.random.default_rng(random_state)
results = {}

while len(results) < n and attempts < max_attempts:
    site_map = {}

    for site in decomp.site_list:
        pool = sorted(decomp.site_pools[site])
        if pool and rng.random() < decomp.substitution_probability:
            site_map[site] = rng.choice(pool)
        else:
            site_map[site] = None

    smi = assemble_series_member(decomp, site_map)
    if smi is invalid:
        continue
    if smi in decomp.ea_canonical_set:
        continue
    if heavy_atom_count(smi) outside decomp.ea_hac_range:
        continue

    results[smi] = VARecord(smiles=smi, strategy="close_in", site_map=site_map)
```

Notes:

- H/no-substituent is a first-class state.
- The generator should not return the first `n` products from lexicographic Cartesian enumeration.
- Add an optional exhaustive generator only for debugging and small test systems.
- Consider adding `min_organic_substituents=1` as a configurable guard, but keep it documented because the paper does not specify this exact guard.

### 7.3 Output metadata

Each VA record should include:

- Canonical SMILES
- Strategy: `close_in`
- Full site map, including `None` sites
- Heavy atom count
- Whether it was also an FW VA, if known later
- Generation seed and attempt index, optionally

### Acceptance criteria

- [ ] Running with the same seed returns the same ordered VA list.
- [ ] No EAs are returned as VAs.
- [ ] All VAs fall inside the EA heavy-atom-count range.
- [ ] At least some returned VAs contain H/no-substituent at one or more sites when `p_sub < 1`.
- [ ] Fragment frequencies are not deterministic lexicographic artifacts.

---

## 8. Phase 3: Free-Wilson VA generation

### 8.1 Definitions

For each EA, represent its full substitution pattern as:

```python
pattern = tuple(site_map[site] for site in decomp.site_list)
```

For a pair of sites `(site_a, site_b)`, define the fixed context as the ordered tuple of all other site values:

```python
context = tuple(site_map[s] for s in decomp.site_list if s not in (site_a, site_b))
```

A qualifying FW matrix is a fixed-context 2-by-2 matrix over two sites.

### 8.2 MMP graph

Build an MMP graph for reporting and debugging:

- Nodes: decomposed EAs.
- Edge between two EAs if they differ at exactly one site and are identical at all other sites.
- Edge label: changed site and old/new substituent values.
- Use `None`/H as a valid substituent level.

This graph is useful, but FW neighborhood membership should not be counted from graph degree alone.

### 8.3 Candidate discovery algorithm

```python
for site_a, site_b in combinations(decomp.site_list, 2):
    rest_sites = [s for s in decomp.site_list if s not in (site_a, site_b)]
    matrices = defaultdict(lambda: defaultdict(list))

    for ea_idx, ea in enumerate(decomp.ea_records):
        ra = ea.site_map[site_a]
        rb = ea.site_map[site_b]
        context = tuple(ea.site_map[s] for s in rest_sites)
        matrices[context][(ra, rb)].append(ea_idx)

    for context, matrix in matrices.items():
        ra_values = sorted(unique first coordinates, with None sorted first)
        rb_values = sorted(unique second coordinates, with None sorted first)

        for ra1, ra2 in combinations(ra_values, 2):
            for rb1, rb2 in combinations(rb_values, 2):
                corners = [(ra1, rb1), (ra1, rb2), (ra2, rb1), (ra2, rb2)]
                present = [corner for corner in corners if corner in matrix]
                missing = [corner for corner in corners if corner not in matrix]

                if len(present) == 3:
                    create_fw_va_from_missing_corner(...)

                if len(present) == 4:
                    create_retrospective_fw_ea_predictions(...)
```

### 8.4 Prediction formula

For a missing corner `(ra_missing, rb_missing)`:

```python
same_a = observed corner with ra == ra_missing
same_b = observed corner with rb == rb_missing
base   = observed corner with ra != ra_missing and rb != rb_missing
pred   = y[same_a] + y[same_b] - y[base]
```

For a 4-corner observed matrix, make four retrospective predictions by holding out each corner once.

### 8.5 Candidate records

```python
@dataclass
class FWVARecord(VARecord):
    fw_pred_mean: float
    fw_pred_std: float
    fw_pred_n: int
    supporting_ea_indices: tuple[int, ...]
    varying_sites: tuple[int, int]
    fixed_context: tuple[str | None, ...]
```

For duplicate FW VAs discovered from multiple qualifying neighborhoods:

- Deduplicate by canonical assembled SMILES.
- Keep all local FW predictions.
- Report mean, standard deviation, minimum, maximum, and number of local predictions.
- Keep all supporting EA triples.

### 8.6 EAs in FW neighborhoods

Define `n_ea_in_fw_nbh` as the number of unique EAs that participate in at least one qualifying FW neighborhood. Count an EA if it appears as:

- A support corner in a 3-present + 1-missing FW VA prediction, or
- A corner in a 4-present retrospective FW EA neighborhood.

Do not count an EA merely because it has MMP graph degree greater than zero.

### Acceptance criteria

- [ ] A toy 2-by-2 system with three EAs generates exactly one FW VA with the correct additivity prediction.
- [ ] A toy 2-by-2 system with four EAs generates no VA but four retrospective FW EA predictions.
- [ ] A three-site toy system where the third site differs does not create a false FW square.
- [ ] H/no-substituent participates correctly as a FW level.
- [ ] FW VAs use only fragments from the EA-derived pools or `None`.
- [ ] FW VAs are constructible by the same assembly code used for close-in VAs.

---

## 9. Phase 4: local FW prediction reports

Add a local FW prediction report independent of whether a VA was generated.

### Outputs

`fw_predictions.csv`:

```text
smiles,type,fw_pred_mean,fw_pred_std,fw_pred_n,observed_activity,abs_error,r2_group_id,varying_sites,context,supporting_eas
```

Where `type` is one of:

- `fw_va`: missing corner, predicted candidate
- `fw_ea`: observed corner, retrospective prediction

### Metrics

For FW EAs:

- R2
- MAE
- RMSE
- Number of retrospective predictions
- Number of unique EAs with predictions

For FW VAs:

- Prediction distribution summary
- Top-N candidates by predicted potency
- Prediction uncertainty summary using local-prediction standard deviation

### Acceptance criteria

- [ ] FW EAs and FW VAs are separated in outputs.
- [ ] FW VA predictions do not require global SVR.
- [ ] Multiple local predictions for the same VA are aggregated but not discarded.

---

## 10. Phase 5: global SVR potency model

### 10.1 Keep two modes

Legacy/convenience mode:

```python
SVRPredictor.fit_ea_only_cv(ea_smiles, ea_activities)
```

Paper-aligned mode:

```python
PaperSVRPredictor.fit_paper_protocol(
    ea_smiles=ea_smiles,
    ea_activities=ea_pic50,
    external_smiles=chembl_non_as_smiles,
    external_activities=chembl_non_as_pic50,
    ea_train_fraction=0.5,
    outer_folds=3,
    inner_folds=3,
    random_state=42,
)
```

### 10.2 Data preparation

Implement `TargetActivesLoader` as an optional helper, but keep the SVR model independent of ChEMBL file format.

Input requirements:

- Canonical SMILES
- pIC50/pActivity values
- Source label: `EA` or `external_target_active`
- Target ID and standard type, if available

Recommended ChEMBL filters for a reproducible approximation:

- Same `target_chembl_id`
- `standard_type == "IC50"`
- Numeric `pchembl_value` or convertible IC50 to pIC50
- Direct equality relation where available
- Exclude AS compounds by canonical parent structure
- Deduplicate by canonical parent SMILES, using median pActivity if needed

Paper-exact filters are not fully specified, so make these filters configurable and record them in metadata.

### 10.3 Fingerprints and kernel

```python
fp = MorganFingerprint(radius=2, n_bits=2048)  # ECFP4
K_train = tanimoto_matrix(train_fps, train_fps)
K_test = tanimoto_matrix(test_fps, train_fps)
model = SVR(kernel="precomputed", C=C, epsilon=epsilon)
```

Add unit tests for:

- Symmetric train kernel
- Diagonal similarity of 1.0
- Test kernel shape `(n_test, n_train)`
- No train/test leakage in precomputed matrices

### 10.4 Splitting protocol

```python
EA_train, EA_external_validation = split_eas_50_50(
    ea_records,
    random_state=random_state,
    stratify_by_potency_bins=True,
)

training_pool = external_target_actives + EA_train
external_validation_pool = EA_external_validation
```

Notes:

- Stratification is not stated in the paper, but it helps avoid pathological random splits. Make it configurable.
- Report the random seed and exact split membership.

### 10.5 Three-fold double CV

Implement nested CV over the training pool:

```python
for outer_train_idx, outer_test_idx in KFold(3, shuffle=True, seed).split(training_pool):
    for params in grid:
        inner_scores = []
        for inner_train_idx, inner_val_idx in KFold(3, shuffle=True, seed).split(outer_train):
            fit SVR on inner_train
            predict inner_val
            score by MAE or RMSE
        choose best params

    fit outer model on outer_train with best params
    predict outer_test
    store outer metrics and chosen params
```

Then produce final VA predictions in one of two documented ways:

Option A, preferred for uncertainty reporting:

- Keep the three outer models.
- Predict each VA with all outer models.
- Report mean and standard deviation.

Option B, preferred for a single final production model:

- Choose final hyperparameters by majority vote or best mean inner score.
- Refit on the full training pool.
- Predict validation EAs and VAs once.

Support both, but make Option A default for analysis because it exposes model uncertainty.

### 10.6 Outputs

`svr_cv_results.csv`:

```text
outer_fold,best_C,best_epsilon,outer_r2,outer_mae,outer_rmse,n_train,n_test
```

`svr_external_validation.csv`:

```text
smiles,observed_activity,pred_mean,pred_std,abs_error,split
```

`va_populations.csv` additions:

```text
svr_pred_mean,svr_pred_std,svr_percentile_vs_eas
```

### Acceptance criteria

- [ ] Paper-mode SVR refuses to run without external target actives unless `allow_ea_only=True` is explicitly set.
- [ ] EA external validation compounds are never included in fitting or inner/outer CV.
- [ ] ECFP4/Tanimoto/SVR implementation remains compatible with scikit-learn `SVR(kernel="precomputed")`.
- [ ] Results include nested-CV metrics and external-validation metrics separately.

---

## 11. Phase 6: diagnostic scoring protocol

The score formulas may remain mostly unchanged, but the run protocol should be paper-aligned.

### Tasks

- [ ] Add `score_with_repeats(decomp, close_in_generator, n_va=1000, n_repeats=10, random_state=...)`.
- [ ] For each repeat, randomly generate or randomly select 1000 close-in VAs.
- [ ] Compute C, D, S, and P for each repeat.
- [ ] Report mean and standard deviation.
- [ ] Use the same descriptor normalization policy for EAs and VAs in every repeat.
- [ ] Keep NBH radius as a tunable hyperparameter; do not hide it behind a single automatic heuristic in paper mode.

### NBH radius handling

Because the paper does not provide a universal numeric radius, support:

```text
--nbh-radius FLOAT          use explicit radius
--nbh-radius-grid values    sweep candidate radii and report sensitivity
--nbh-radius auto           legacy/convenience mode only, unless explicitly allowed
```

### Acceptance criteria

- [ ] Paper-mode score output contains 10 individual rows plus mean/std summary.
- [ ] Score output records `n_va`, `n_repeats`, radius, descriptor list, scaler, and seed.
- [ ] Scoring can use FW or diverse VAs experimentally, but close-in is the default diagnostic population.

---

## 12. Phase 7: CLI/API changes

### CLI sketch

```bash
python -m como \
  --series data/as1.csv \
  --smiles-col smiles \
  --activity-col pIC50 \
  --core data/as1_core.smi \
  --paper-mode \
  --va close_in free_wilson \
  --va-n 51200 \
  --score-va-n 1000 \
  --score-repeats 10 \
  --nbh-radius 0.35 \
  --external-actives data/chembl2998_non_as_actives.csv \
  --svr-mode paper \
  --random-state 42 \
  --output results/as1_paper_mode
```

### New options

```text
--paper-mode
--random-state INT
--score-va-n INT
--score-repeats INT
--nbh-radius FLOAT|auto
--nbh-radius-grid CSV_FLOATS
--svr-mode legacy|paper|off
--external-actives FILE
--ea-train-fraction FLOAT
--outer-folds INT
--inner-folds INT
--svr-c-grid CSV_FLOATS
--svr-epsilon-grid CSV_FLOATS
--require-explicit-sites / --no-require-explicit-sites
```

### Output files

```text
series_decomposition.csv
rejected_molecules.csv
close_in_vas.csv
free_wilson_vas.csv
fw_predictions.csv
scores_by_repeat.csv
scores_summary.csv
svr_cv_results.csv
svr_external_validation.csv
va_populations.csv
metadata.json
summary.md
```

---

## 13. Phase 8: tests

### 13.1 Unit tests

#### Decomposition and assembly

- [ ] Core with three explicit exit vectors produces exactly three sites.
- [ ] EA with no group at one site gets `None` for that site.
- [ ] Fragment extraction preserves site identity.
- [ ] Reassembling decomposed EAs reproduces canonical SMILES.
- [ ] Substituent outside declared exit vectors is rejected in paper mode.

#### Close-in generation

- [ ] Random seed reproducibility.
- [ ] HAC filtering.
- [ ] EA exclusion.
- [ ] H/no-substituent appears when expected.
- [ ] Generator stops cleanly if requested `n` exceeds feasible unique products.

#### FW generation

- [ ] Three-corner matrix creates one missing VA.
- [ ] Prediction formula is correct for each possible missing corner.
- [ ] Four-corner matrix creates retrospective FW EA predictions.
- [ ] Context mismatch blocks false FW candidates.
- [ ] `None`/H works as a substituent level.
- [ ] Duplicate candidates aggregate multiple local predictions.
- [ ] `n_ea_in_fw_nbh` counts qualifying participants, not graph-degree participants.

#### SVR

- [ ] Morgan radius 2, 2048-bit fingerprints are used.
- [ ] Tanimoto kernel shape and diagonal are correct.
- [ ] Nested CV never leaks external-validation EAs.
- [ ] Missing external actives in paper mode gives a clear error.
- [ ] Prediction output aligns with input SMILES order and uses `nan` for invalid molecules.

### 13.2 Integration tests

- [ ] Synthetic 2-site AS with known FW square.
- [ ] Synthetic 3-site AS with a context mismatch.
- [ ] Small ChEMBL-like CSV with external actives and 50/50 EA split.
- [ ] End-to-end paper-mode run produces all expected output files.

---

## 14. Phase 9: validation against paper-style targets

### Minimal validation without exact AS1/AS2 data

- [ ] Close-in VAs are random, H-aware, size-filtered, and EA-excluding.
- [ ] FW VAs are strict context-fixed missing corners.
- [ ] FW VAs are a subset of the close-in enumerability space.
- [ ] Local FW predictions are produced for FW VAs and retrospective FW EAs.
- [ ] SVR paper mode uses external actives + 50% EAs for training and 50% EAs for validation.
- [ ] Scoring reports 1000 close-in VAs x 10 repeats with mean/std.

### Strong validation with exact AS1/AS2 inputs

Try to reproduce, or at least explain deviations from:

| Quantity | AS1 | AS2 |
|---|---:|---:|
| EAs | 219 | 158 |
| Sites | 4 | 3 |
| Unique substituents | 70 | 133 |
| FW VAs | 907 | 3167 |
| EAs in FW neighborhoods | 183 | 45 |
| C/D/S/P | 0.43/0.90/0.58/0.55 | 0.18/0.73/0.29/0.95 |
| External ChEMBL actives for SVR | 517 | 1135 |

Deviations should be logged under one of these causes:

- Different ChEMBL standardization or activity filtering
- Different compound-core relationship extraction
- Different core/scaffold definition
- RDKit replacement for OpenEye/synthetic-rule extraction
- Different NBH radius or descriptor normalization
- Random seed / sampling variance

---

## 15. Implementation order

Recommended PR sequence:

1. `PR-1-series-schema`: add dataclasses, standardization, decomposition, assembly, and tests.
2. `PR-2-close-in-paper-mode`: replace deterministic close-in generator with H-aware random enumeration; keep legacy exhaustive mode.
3. `PR-3-fw-context`: rewrite FW discovery with fixed-context matrices and correct FW neighborhood counts.
4. `PR-4-fw-prediction-report`: add FW VA and FW EA prediction reports and metrics.
5. `PR-5-svr-paper-protocol`: add external-actives + 50% EA split and nested CV.
6. `PR-6-scoring-protocol`: add 1000 VA x 10 repeat diagnostic scoring and radius reporting.
7. `PR-7-cli-docs`: expose paper-mode CLI/API options and update README claims.
8. `PR-8-regression-benchmarks`: add paper-style benchmark fixtures if exact AS1/AS2 inputs are available.

---

## 16. README wording update

Until these changes and benchmarks pass, avoid calling the package a faithful reproduction. Suggested wording:

```markdown
This package implements COMO-inspired diagnostic scoring and virtual analog generation. It includes a paper-alignment mode for close-in VAs, Free-Wilson VAs, and SVR potency prediction. The REINVENT transfer-learning component and the exact ChEMBL26 diverse R-group library are not included.
```

After alignment and validation, use a more specific claim:

```markdown
This package reproduces the non-REINVENT COMO/DeepCOMO components implemented with RDKit/scikit-learn: close-in VA enumeration, Free-Wilson VA generation/local prediction, diagnostic scoring, and global ECFP4/Tanimoto SVR. The exact diverse R-group source and transfer-learning sampled VAs remain external.
```

---

## 17. Done criteria

The implementation is ready to call "paper-aligned, excluding REINVENT and the exact diverse library" when:

- [ ] Close-in generation is random, H-aware, EA-size-filtered, seedable, and based on all declared AS sites.
- [ ] FW generation uses fixed-context 2-by-2 matrices and produces local FW predictions with support metadata.
- [ ] FW neighborhood counts are based on qualifying FW neighborhoods.
- [ ] Global SVR paper mode uses external target actives + 50% EAs for fitting and the remaining 50% EAs only for validation.
- [ ] Three-fold double CV is implemented and reported separately from external validation.
- [ ] Diagnostic scoring uses close-in VAs with 1000 VAs x 10 repeats and reports mean/std.
- [ ] CLI/API output records all parameters needed to reproduce a run.
- [ ] Tests cover decomposition, assembly, close-in, FW, SVR, and end-to-end paper mode.
- [ ] README accurately distinguishes paper-aligned components from missing/deferred components.
