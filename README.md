# CS 6850: AD/ADRD identification infrastructure

Shared Python infrastructure for the Cornell CS 6850 final research project comparing
ICD diagnosis-code classification, diagnosis + one LLM, diagnosis + multiple LLM
roles, and adaptive escalation. This repository currently contains **fully synthetic
fixtures and research scaffolds only**. No MIMIC data, clinical ICD definitions,
human-labeled HW1 records, or experimental findings are included.

The approximately 200 previously human-labeled HW1 patients are for development.
The test cohort must be sampled independently after excluding **all development
patient identities**, then labeled by human reviewers. Freeze prompts, code sets,
thresholds, and escalation rules before evaluating that cohort.

## Setup and quick start

Python 3.11+ is required. Runtime code uses only the standard library; pytest is the
only direct test dependency. Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python run_experiment.py --size 12 --seed 6850
```

The CLI always generates 24 invented patients, excludes the first four as mock
development patients, and runs all four approaches on the same sample. Its JSON
stdout contains aggregate diagnostics explicitly marked synthetic; those numbers
are infrastructure checks, not research results. Mock scores are deterministic
hash-derived values and do not implement clinical reasoning. Mock token counts
are zero because no inference occurs. Latency is measured and will vary.

## Architecture and contracts

| Module | Responsibility |
| --- | --- |
| `src/schemas.py` | Immutable patient, diagnosis, reference-label, prediction, and usage contracts |
| `src/data_loader.py` | Confirmed HW1 export reader, human labels, team exclusions, and diagnosis-resolution boundary |
| `src/synthetic.py` | Invented records and `SYNTH` codes, with no clinical meaning |
| `src/sampling.py` | Uniform sampling without replacement after development exclusion |
| `src/validation.py` | Duplicate, alignment, and development/test leakage checks |
| `src/llm_client.py` | Client protocol, deterministic mock, approved local HTTP adapter |
| `src/models/` | Common classifier protocol and four small reference implementations |
| `src/evaluation.py` | Metrics, confusion matrix, latency, usage, and review effort |
| `src/experiment.py` | Cohort selection, model execution, validation, and aggregate report |
| `run_experiment.py` | Synthetic-only command-line entry point |
| `tests/` | Synthetic unit and integration tests; no model server or source data needed |

These are **internal application schemas, not MIMIC schemas**:

- `Patient(patient_id, diagnoses, clinical_text)` carries a stable patient identity,
  a tuple of `DiagnosisCode(system, code)`, and selected text. No reference label or
  review metadata is passed to a classifier. The local prompt also omits patient ID.
- `ReferenceLabel(patient_id, label, human_review_seconds)` is a separate reviewed
  binary outcome, with class `1` meaning AD/ADRD under the team's finalized rubric.
  It is never inferred from the ICD baseline. Missing labels block evaluation.
- `Prediction(patient_id, model_name, score, predicted_label, latency_seconds,
  usage, route)` records a finite positive-class score in `[0,1]`, binary decision,
  full workflow wall time, cost, and route. Scores are not presumed calibrated.
- `Usage(input_tokens, output_tokens, llm_calls, human_review_seconds)` counts the
  entire workflow. Missing measured tokens or review effort are `None`, not zero;
  an unknown component makes its aggregate total unknown. Counts remain separate
  for input and output tokens. Human effort is elapsed review time in seconds.

Frozen dataclasses validate inputs at construction. All records for a person must
be aggregated upstream into exactly one `Patient`; duplicate identities are errors,
not silently deduplicated encounters. The exclusion list must use the same stable
patient identity across admissions and data sources.

## Four classification approaches

Every classifier implements `name`, `predict(patient) -> Prediction`, and
`describe() -> dict` for experiment configuration. All receive diagnoses and text;
the ICD baseline uses diagnoses only.

1. `ICDBaseline`: exact `(system, code)` membership in a required, nonempty,
   externally supplied code set. Its score is binary. There is no default clinical
   code set, prefix expansion, code normalization, or claimed clinical definition.
2. `SingleLLM`: one integrated diagnosis/text assessment. Default decision threshold
   is `0.5` (positive when score is greater than or equal to threshold).
3. `MultiAgent`: two sequential calls with different evidence-review perspectives;
   the final score is their arithmetic mean. This is a minimal role ensemble,
   not a validated debate/adjudication protocol. It uses no agent framework.
4. `AdaptiveAgent`: one initial assessment; scores in the inclusive default band
   `[0.35, 0.65]` trigger the two-role ensemble. The ensemble score replaces the
   initial score; all three calls and their costs are counted. Other cases use one
   call. `expensive_client` optionally selects a separately approved larger model.

The prompts, aggregation rule, thresholds, and ambiguity band are **placeholders
to tune on development patients**, not clinical recommendations. The band is a
score heuristic, not a statistically validated uncertainty estimate. The current
workflows do not perform human adjudication; prediction review effort is zero.
Future human-reviewed workflows should record actual effort in `Usage`.

## Approved local LLM backend

No model or hosting environment is approved by this repository. Obtain approval
through the team's existing institutional process, verify that model and server
are entirely local, and configure the approved model identifier explicitly:

```bash
python run_experiment.py --backend local \
  --local-model YOUR_APPROVED_MODEL_ID --approved-local \
  --local-url http://127.0.0.1:8000
```

This command still uses only synthetic data. `--approved-local` acknowledges
existing approval; it does not grant it. The adapter accepts only explicit IPv4 or
IPv6 loopback HTTP origins, disables proxies and redirects, and makes no retries.
There is no remote-cloud backend. A loopback endpoint alone cannot establish how a
server processes data; the team must verify its configuration and logging policy.

The expected server contract is `POST /v1/chat/completions`, accepting `model`,
`messages`, `temperature`, `seed`, and `max_tokens`. It must return
`choices[0].message.content` containing only JSON like `{"score": 0.7}`. Optional
`usage.prompt_tokens` and `usage.completion_tokens` are tracked; absent counts stay
unknown. Check this contract against the approved backend before connecting HW1.
Generation defaults are temperature `0`, seed `0`, and 128 output tokens; a seed
does not guarantee deterministic inference on every backend.

Use `LocalConfig` and `create_client("local", local_config=...)` in Python for
configuration. Client failures and malformed scores fail the run with a generic
error, without logging prompt or response bodies. Failed runs do not produce a
partial comparative report; their incurred cost is not currently persisted.

## Loading HW1 development exports

`load_hw1(paths)` reads the inspected HW1 CSV export format. The user confirmed
that the `is AD/ADRD? (type in 1,0, -1) ...` column contains human annotations.
The exact header is `HUMAN_LABEL_COLUMN` in `src/data_loader.py`; the loader does
not substitute `adrd_dx(final)` or `adrd_dx(icd_code)` for reviewed labels.

```python
from src.data_loader import load_hw1

# Run only in the approved data environment. Paths are illustrative.
development = load_hw1(["data/hw1/member_a.csv", "data/hw1/member_b.csv"])
development_ids = development.development_ids
references = development.references
uncertain_ids = development.uncertain_ids
unlabeled_ids = development.unlabeled_ids
```

Supply the files you actually have, then add all teammates' exports when available.
The returned exclusion set covers **every patient in the supplied files**, including
uncertain (`-1`) and blank annotations. It cannot establish that all team development
patients have been supplied. Complete the team exclusion set before test sampling.
Duplicate patient identities within or across files are errors, even across
different admissions or matching labels; resolve overlap explicitly with the team.

`references` contains only binary human labels (`0` and `1`). Uncertain and blank
labels remain separately accessible, never converted to negative labels. Review
time is unknown because this export does not record it. For development analysis,
explicitly select the reviewed subset and report that selection; for final test
evaluation, resolve labels under the agreed adjudication policy before running.

The loader preserves `subject_id`, `hadm_id`, and the existing discharge `text`.
It does not rerun the notebook, select a new encounter, append reviewer reasoning,
or copy any input files. Only discharge text is currently passed as clinical text;
radiology, medication, and other structured features require a separately agreed
input design. Source clinical text still needs review under the observation policy.

The notebook builds `all_icd_codes` as a pipe-separated list, dropping ICD version.
Consequently, the export alone cannot construct versioned diagnosis inputs. Supply
a source-verified mapping from `(subject_id, hadm_id)` to tuples of `DiagnosisCode`
objects, using systems consistent with the reviewed baseline code set:

```python
# Resolve this mapping from the original diagnosis source in the approved environment.
# Do not infer versions from code spelling or assign one version to the whole export.
patients = development.to_patients(diagnoses_by_encounter)
```

Conversion requires a mapping for each exported encounter and checks that its
codes match the exported code list exactly. Missing diagnoses cannot silently
become negative cases. All patients, including unresolved cases, remain in the
converted inputs, so full-cohort evaluation rejects missing reference labels.

The HW1 notebook uses prefix matching for its ICD-selected strata, whereas the
current baseline uses exact `(system, code)` membership. Reconcile this difference
with the reviewed code dictionary before claiming to reproduce HW1. The HW1
development sample is selected from ICD-positive/negative groups; it is not an
independent population-random test sample.

## Remaining integration before research use

The CSV reader establishes the confirmed development-export mapping. Supply and
review the remaining research inputs in the approved environment:

1. Actual source tables/columns and stable patient-identity mapping; verify consent,
   access restrictions, and the allowable data environment through existing policy.
2. Encounter aggregation, observation/index windows, note selection and truncation,
   and safeguards against text revealing labels or information after the cutoff.
3. The reviewed AD/ADRD target definition and versioned ICD systems/code set,
   including any normalization or expansion. No clinical codes are supplied here.
4. The complete development ID list, including patients used for HW1 and later
   tuning, and a versioned eligible sampling frame independent of reference labels.
5. Reference-label mapping and the human review/adjudication rubric. Record effort
   and any reviewer disagreement in the approved annotation workflow. Prefer
   reviewers blinded to model outputs; unresolved cases need an explicit policy.

Sample before obtaining test labels; save the selected cohort only in approved,
ignored storage. The library can select the cohort before labeling:

```python
from src.sampling import sample_test_cohort

# eligible_patients is a separately prepared, versioned sampling frame.
# development_ids includes ALL teammate exports and any other tuning patients.
cohort = sample_test_cohort(eligible_patients, development_ids, size=100, seed=6850)
# Obtain independent reviewed labels for cohort in the approved environment.
```

After labeling, call `src.experiment.run_experiment` with the **same versioned
sampling frame**, full development exclusion set, sample size, and seed, plus
reviewed labels and all four configured models. It reproduces the selection before
checking labels; it never samples only the patients whose labels happen to exist.
The returned `ExperimentResult.cohort_ids` should match the saved cohort manifest.
Labels may include other patients, but every sampled patient needs a label.

The manifest/report records data version, Python version, seed, sample size, and
model configurations. Retain the exact source commit, environment/package versions,
approved model revision/weights, and the patient cohort manifest in approved
experiment storage. Use the same Python version for strict sampler replay.
`ExperimentResult` holds patient-level predictions in memory; no patient-level
files are written automatically. Do not print or commit that object.

## Evaluation conventions

Evaluation aligns by patient ID, rejects duplicate/missing/extra predictions, and
evaluates one model at a time. AUROC uses continuous scores and average ranks for
ties; a binary baseline's AUROC provides only a coarse ranking. Sensitivity, PPV,
specificity, accuracy, and F1 use the predicted binary decisions. Confusion matrices
use true labels as rows and predicted labels as columns: `[[TN, FP], [FN, TP]]`.
Undefined ratios and single-class AUROC are `null`/`None`, never silently zero.
An empty evaluation cohort is an error.

Reports include total/mean patient prediction latency, token usage, call count,
route counts, and unknown-cost counts. Prediction latency excludes loading,
sampling, and evaluation. Models run sequentially; local runtime comparisons will
need a predeclared warm-up and order policy. Reference-label review cost is the
**same shared annotation cost** repeated for context in each report; do not sum it
across models. Model-specific human effort is recorded separately. Confidence
intervals and statistical comparisons are future research work.

## Team responsibilities

Assign people to these roles before integrating restricted data:

| Role | Owned work and handoff |
| --- | --- |
| Data/HW1 integration owner | Reviewed adapter, observation windows, stable identity, development list, versioned sampling frame |
| Clinical definition/annotation owner | Target rubric, sourced ICD code set, independent labels, adjudication, review-effort records |
| Model owners | Baseline, single, multi-role, and adaptive implementations; frozen prompts/configurations; approved backend integration |
| Evaluation/reproducibility owner | Shared cohort manifest, metric conventions, cost accounting, frozen runs and later statistical comparisons |
| All contributors/reviewers | Synthetic tests, code review, data handling, documentation of assumptions and limitations |

## Repository hygiene

Keep restricted inputs under `data/`, `private/`, or `restricted/` and patient-level
outputs under `outputs/`, `results/`, or `artifacts/`, all ignored. Common data and
output formats, notebooks (including cell outputs), logs, environment files, and
credentials are ignored throughout the tree. Synthetic fixtures live in Python
source so they need no data-file exception. Port reviewed notebook logic into
modules instead of committing executed notebooks.

Ignore rules do not inspect content or remove previously tracked files. Never
force-add restricted artifacts, paste patient text into code/docs/PRs, or commit
credentials. Inspect `git diff --cached` before every commit. Real aggregate
reports still need review under the team's applicable data-release rules.
