# COMO

A Python implementation of the **COMO/DeepCOMO** methodology for analog series analysis.

> Yonchev & Bajorath, *J. Comput.-Aided Mol. Des.* **2020**, 34, 1207–1218.

COMO scores a compound series against its virtual analog (VA) population to estimate
**chemical saturation** (S score) and **SAR progression** (P score), and assigns a
development stage label (early / early_mid / mid / late).

---

## Installation

```bash
cd /novo/users/vuln/como
pip install -e .
```

Requires RDKit, scikit-learn, polars, networkx (all available in the `misc` conda env).

---

## Quick Start — CLI

```bash
python -m como \
  --series tests/data/synthetic_egfr.csv \
  --va close_in diverse free_wilson \
  --output /tmp/como_results/
```

Output files written to `--output`:

| File | Contents |
|---|---|
| `scores.csv` | C, D, S, P, stage, NBH radius, SVR CV metrics |
| `va_populations.csv` | All VAs with predicted pActivity, percentile rank, NBH membership |
| `summary.txt` | Human-readable report with top-10 VA table and stage interpretation |

### All CLI options

```
python -m como [-h]
  --series FILE           CSV with existing analog (EA) SMILES + pActivity
  --smiles-col COL        Column name for SMILES [default: smiles]
  --activity-col COL      Column name for pActivity [default: pActivity]
  --core SMILES           Scaffold SMILES for R-group decomposition.
                          Auto-detected from Murcko scaffolds if omitted.
                          TIP: supply this explicitly for best Free-Wilson results.
  --va STRATEGY [...]     VA strategies: close_in, diverse, free_wilson
                          [default: close_in]
  --va-csv FILE           CSV from a generative model (plug-in hook).
                          Must have a 'smiles' column; optional pActivity column.
  --va-n N                Max VAs per strategy [default: 1000]
  --nbh-radius auto|FLOAT NBH radius in normalized 7D descriptor space.
                          'auto' = adaptive median k-NN distance [default: auto]
  --fragment-lib FILE     Custom fragment SMILES file for the diverse strategy.
                          One SMILES per line. Falls back to bundled ~200 fragments.
  --output DIR            Output directory [default: results]
  --s-threshold FLOAT     S score threshold for stage assignment [default: 0.4]
  --p-threshold FLOAT     P score threshold for stage assignment [default: 0.5]
  --svr-c FLOAT           SVR regularization C [default: 10.0]
  --svr-epsilon FLOAT     SVR epsilon [default: 0.1]
```

### Examples

```bash
# Minimal — close-in VAs only, auto-detect core
python -m como --series my_series.csv

# All three VA strategies with an explicit core scaffold
python -m como \
  --series my_series.csv \
  --core "c1cc2ncncc2cc1" \
  --va close_in diverse free_wilson \
  --va-n 2000

# Plug in a generative model: pass its SMILES output as --va-csv
python -m como \
  --series my_series.csv \
  --va-csv generated_mols.csv \
  --va-csv-activity-col pred_pAct \
  --output results_gen/

# Use a custom fragment library for the diverse strategy
python -m como \
  --series my_series.csv \
  --va diverse \
  --fragment-lib my_fragments.smi
```

---

## Python API

### One-call pipeline

```python
import como

result = como.score_series(
    series_csv="tests/data/synthetic_egfr.csv",
    smiles_col="smiles",
    activity_col="pActivity",
    core="c1cc2ncncc2cc1",          # optional; auto-detected if omitted
    va_strategies=["close_in", "diverse", "free_wilson"],
    va_n=500,
    output_dir="/tmp/como_out",
)

print(f"S={result.S:.3f}  C={result.C:.3f}  D={result.D:.3f}  P={result.P:.3f}")
print(f"Stage: {result.stage}")
print(f"SVR CV R²={result.cv_r2:.3f}  MAE={result.cv_mae:.3f}")

# VA populations as a Polars DataFrame
print(result.va_df.head(10))
```

`score_series` signature:

```python
como.score_series(
    series_csv: str | Path,
    smiles_col: str = "smiles",
    activity_col: str = "pActivity",
    core: str | None = None,            # scaffold SMILES
    va_strategies: list[str] = ("close_in",),
    va_csv: str | Path | None = None,             # generative model plug-in
    va_csv_activity_col: str | None = None,        # column name for model's predicted pActivity
    va_n: int = 1000,
    nbh_radius: float | None = None,    # None → adaptive
    output_dir: str | Path = "results",
    s_threshold: float = 0.4,
    p_threshold: float = 0.5,
    svr_c: float = 10.0,
    svr_epsilon: float = 0.1,
    fragment_lib: str | Path | None = None,
) -> ComoResult
```

### Individual score functions

```python
import numpy as np
from como import c_score, d_score, s_score, p_score, assign_stage

# membership: (n_va, n_ea) boolean array — True if VA j is in EA i's NBH
membership = np.array([
    [True, False, True],
    [False, True,  True],
    [True, True,   False],
])
ea_activities = np.array([7.2, 7.8, 6.9])

C = c_score(membership)
D, d_mean = d_score(membership)
S = s_score(C, D)
P = p_score(membership, ea_activities)
stage = assign_stage(S, P)          # "early" | "early_mid" | "mid" | "late"
```

### VA generators

Each generator is independently usable:

```python
import numpy as np
from como.analogs.close_in import CloseInVAGenerator
from como.analogs.free_wilson import FreeWilsonVAGenerator
from como.analogs.diverse import DiverseVAGenerator
from como.analogs.csv_plugin import CSVPluginVAGenerator

ea_smiles = [
    "COc1cc2ncnc(Nc3ccccc3)c2cc1OC",
    "COc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OC",
    "CCOc1cc2ncnc(Nc3ccccc3)c2cc1OCC",
]
ea_activities = np.array([7.2, 7.8, 6.9])
core = "c1cc2ncncc2cc1"

# Close-in: combinatorial R-group enumeration from observed EA substituents
gen = CloseInVAGenerator()
vas = gen.generate(ea_smiles, ea_activities, core, n=200, ea_hac_range=(10, 50))

# Free-Wilson: predict and generate missing 2x2 matrix corners
gen = FreeWilsonVAGenerator()
vas = gen.generate(ea_smiles, ea_activities, core, n=200, ea_hac_range=(10, 50))
print(gen.fw_predictions)   # {canonical_smi: predicted_pActivity}

# Diverse: substitute core sites with MedChem fragment library
gen = DiverseVAGenerator(fragment_lib_path=None)   # None = bundled library
vas = gen.generate(ea_smiles, ea_activities, core, n=500, ea_hac_range=(10, 50))

# CSV plug-in: pass any generative model's output directly
gen = CSVPluginVAGenerator("generated.csv", activity_col="pred_pAct")
vas = gen.generate(ea_smiles, ea_activities, core=None, n=1000, ea_hac_range=(5, 100))
print(gen.external_activities)   # {canonical_smi: float}
```

### Descriptors, NBH, and SVR

```python
from como.descriptors import compute_descriptors, normalize_descriptors
from como.nbh import build_nbh
from como.potency import SVRPredictor

# 7D physicochemical descriptors (MW, LogP, TPSA, HBD, HBA, RotBonds, Rings)
ea_raw, ea_valid_idx = compute_descriptors(ea_smiles)
va_raw, va_valid_idx = compute_descriptors(va_smiles)
ea_norm, va_norm, scaler = normalize_descriptors(ea_raw, va_raw)

# Adaptive NBH: radius = median of k=3 NN distances among EAs
membership, radius = build_nbh(ea_norm, va_norm, r=None, k=3)

# SVR potency model: ECFP4 + Tanimoto kernel
svr = SVRPredictor(C=10.0, epsilon=0.1)
cv_metrics = svr.fit(ea_smiles, ea_activities)   # {"cv_r2": ..., "cv_mae": ...}
va_preds = svr.predict(va_smiles)
```

---

## Methodology

### Scores (paper Equations 1–6)

**C score** (chemical coverage, Eq. 1):
```
C = #VAs in ≥1 EA neighborhood / #VAs total
```

**D score** (neighborhood density, Eqs. 2–3):
```
count_j = number of EA neighborhoods containing VA j
d_mean  = mean(count_j) over covered VAs only (count_j > 0)
D       = 1 − 1 / d_mean
```

**S score** (chemical saturation, Eq. 4):
```
S = 2CD / (C + D)   [harmonic mean; 0 if C + D = 0]
```

**P score** (SAR progression, Eqs. 5–6):
```
For each VA j with count_j ≥ 2:
  delta_j = mean pairwise |pActivity| difference across EAs in that NBH
  w_j     = 1 / count_j
P = Σ(w_j · delta_j) / Σ(w_j)
```

### Development Stage Assignment

|          | Low P       | High P       |
|----------|-------------|--------------|
| **Low S**  | early       | early_mid    |
| **High S** | mid         | late         |

Default thresholds: S = 0.4, P = 0.5 (configurable).

### VA Generation

| Strategy | Description |
|---|---|
| `close_in` | Combinatorial R-group enumeration using substituents observed in EAs |
| `diverse` | Substitute core sites with MedChem fragment library (~200 bundled) |
| `free_wilson` | Fill missing corners of 2×2 R-group submatrices; predict pActivity by additivity |
| CSV plug-in | Pass any generative model's SMILES output via `--va-csv` |

R-group decomposition uses `RDKit.ReplaceCore(..., labelByIndex=True)` + `GetMolFrags` so that substitution sites are labeled by their core atom index — consistent across all EAs regardless of canonical atom ordering.

---

## Running Tests

```bash
cd /novo/users/vuln/como
conda activate misc
python -m pytest tests/ -v
```

76 tests covering descriptors, NBH construction, all scoring equations, each VA strategy, SVR, and the full integration pipeline.
