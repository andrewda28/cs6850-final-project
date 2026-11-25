"""Synthetic-only CLI. Restricted HW1 workflows must integrate the library explicitly."""

import argparse
import json

from src.experiment import run_experiment
from src.llm_client import LocalConfig, create_client
from src.models import AdaptiveAgent, ICDBaseline, MultiAgent, SingleLLM
from src.synthetic import SYNTHETIC_POSITIVE, synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run infrastructure smoke experiment on fully synthetic fixtures")
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=6850)
    parser.add_argument("--backend", choices=("mock", "local"), default="mock")
    parser.add_argument("--local-model")
    parser.add_argument("--local-url", default="http://127.0.0.1:8000")
    parser.add_argument("--approved-local", action="store_true", help="Acknowledge prior approval of model and environment")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ambiguity-lower", type=float, default=0.35)
    parser.add_argument("--ambiguity-upper", type=float, default=0.65)
    args = parser.parse_args()
    if args.backend == "local" and not args.local_model:
        parser.error("--local-model is required for the local backend")
    if args.backend == "mock" and (args.local_model or args.approved_local):
        parser.error("Local model settings require --backend local")
    config = LocalConfig(args.local_model, args.approved_local, args.local_url) if args.backend == "local" else None
    client = create_client(args.backend, local_config=config)
    patients, labels = synthetic_dataset()
    models = [
        ICDBaseline(frozenset({SYNTHETIC_POSITIVE}), code_set_version="synthetic-demo-v1"),
        SingleLLM(client, threshold=args.threshold), MultiAgent(client, threshold=args.threshold),
        AdaptiveAgent(client, lower=args.ambiguity_lower, upper=args.ambiguity_upper, threshold=args.threshold),
    ]
    result = run_experiment(patients, labels, {p.patient_id for p in patients[:4]}, models,
                            size=args.size, seed=args.seed, data_version="fully-synthetic-v1")
    result.summary["notice"] = "SYNTHETIC INFRASTRUCTURE CHECK ONLY; not clinical research results"
    print(json.dumps(result.summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
