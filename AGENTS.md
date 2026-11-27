# Project objectives

Study AD/ADRD identification in MIMIC electronic health records for Cornell CS 6850.
Compare diagnosis-code classification, diagnosis + a single LLM, diagnosis +
multiple LLM roles, and adaptive escalation on the same independently sampled,
human-reviewed test patients. Approximately 200 HW1-labeled patients are for
development and must be excluded from the test cohort at the patient level.

# Conventions

- Use Python 3.11+, standard-library runtime components, dataclasses, type hints,
  and small modules. Keep pytest as the only direct test dependency unless a real
  requirement justifies more. Do not add frontends, databases, deployment services,
  or agent frameworks for this infrastructure task.
- Preserve the shared contracts in `src/schemas.py`. Keep reference labels and
  annotation metadata separate from classifier inputs. All classifiers implement
  `src.models.base.Classifier` and return exactly one prediction per patient.
- Never silently deduplicate patients or drop missing predictions. Fail closed on
  duplicate identities, patient alignment errors, and development/test leakage.
- Use a local seeded RNG and sorted eligible patient identities for sampling.
  Sample independently of whether labels are already available. Never tune on test
  labels. Freeze configurations before final evaluation.
- Track total workflow latency, input/output tokens, LLM calls, and actual human
  review seconds. Preserve unknown values as `None`; never fabricate measurements.
  Reference-label annotation effort is shared across models, not multiplied by them.
- Use finite scores in `[0,1]`, positive class `1`, and explicit decision thresholds.
  Document undefined metrics, confusion-matrix orientation, and score semantics.
- Keep prompts and local backend configuration reviewable and versioned. Local
  approval acknowledgement does not grant institutional approval. Do not add a
  remote endpoint or transmit restricted data through development tools.

# Data and evidence boundaries

- Use only fully synthetic patients in tests, examples, and repository artifacts.
  `SYNTH` fixture codes are invented and have no clinical meaning.
- Do not invent MIMIC schemas, ICD definitions, patient observations, annotation
  decisions, or experimental findings. `src/data_loader.py` is the HW1 notebook
  integration boundary; leave unsupported mappings explicitly unimplemented.
- The user confirmed the manual `is AD/ADRD?` export column contains human labels.
  Keep `-1` and blank annotations unresolved and include their patients in the
  development exclusion set. Combine all teammates' exports before test sampling;
  reject overlapping identities instead of silently picking an annotation.
- HW1 `all_icd_codes` omits ICD versions. Require source-resolved diagnoses keyed
  by patient and admission before creating model inputs. Never guess versions or
  turn missing diagnoses into negative baseline classifications. Reconcile HW1
  prefix matching with the exact-match baseline before claiming reproduction.
- Review the existing HW1 notebook within the approved data environment to establish
  identity mapping, source fields, encounter aggregation, observation windows,
  note selection, code definitions, labels, and the full development exclusion list.
- Keep restricted data, patient-level outputs, credentials, and notebook outputs
  out of Git, logs, terminal output, issues, and PRs. Store sensitive files in ignored
  directories. Never bypass ignore rules to commit data. Aggregate outputs are not
  automatically suitable for release.

# Verification and review

- Run `python -m pytest -q` and `python run_experiment.py --size 12 --seed 6850`
  from the repository root after changes. Tests must not need MIMIC, credentials,
  external network access, or a model server; inject synthetic client responses.
- Test behavior: hand-calculated metrics and ties, undefined metrics, duplicate and
  missing records, patient exclusion, reproducibility, escalation boundaries, and
  full workflow cost accounting. Avoid asserting wall-clock timings exactly.
- Update README usage and limitations when contracts or workflow policies change.
- Inspect staged changes for sensitive content. PRs should state the concrete
  behavior, synthetic validation, limitations, and remaining HW1 integration points.
- Team responsibilities are defined in README; do not invent member assignments.
