#!/usr/bin/env python3
"""Run the pipeline, or a single stage of it.

Stages: scenarios, precompute, simulate, summarize, compare (or all).

Example:
    python run_pipeline.py simulate --model baseline --limit 1 --processes 1
"""

import argparse
import sys

from rsa_planning.paths import MODEL_ALIASES, ensure_output_dirs

MODELS = list(MODEL_ALIASES)


def cmd_scenarios(args):
    from rsa_planning.scenarios import create_scenarios, seed_scenario_generation
    seed_scenario_generation(args.seed)
    create_scenarios(mode=args.mode, version=args.version, per_type=args.per_type,
                     start_id=args.start_id, overwrite=args.overwrite)


def cmd_precompute(args):
    from rsa_planning.precompute import build_trees
    build_trees(max_horizon=args.horizon, overwrite=args.overwrite)


def check_test_cases(args):
    import os
    from rsa_planning.paths import DEFAULT_TEST_CASES_CSV
    path = args.test_cases or DEFAULT_TEST_CASES_CSV
    if not os.path.exists(path):
        sys.exit(f"No test cases at {path}\n"
                 f"Generate some first, e.g.:\n"
                 f"  python run_pipeline.py scenarios --per-type 1 --seed 1 --version demo\n"
                 f"and pass --test-cases data/scenarios/dynamic_test_cases_vdemo.csv")


def cmd_simulate(args):
    from rsa_planning.precompute import require_tree
    from rsa_planning.runner import run_model

    check_test_cases(args)
    require_tree(args.horizon)
    for model in (MODELS if args.model == "all" else [args.model]):
        run_model(model,
                  test_cases_csv=args.test_cases,
                  tc_ids=args.tc_id,
                  limit=args.limit,
                  processes=args.processes,
                  time_horizon=args.horizon,
                  overwrite=args.overwrite)


def cmd_summarize(args):
    from rsa_planning.summarize import build_summary
    from rsa_planning.trajectories import build_trajectories

    check_test_cases(args)
    for model in (MODELS if args.model == "all" else [args.model]):
        build_trajectories(model, test_cases_csv=args.test_cases,
                           processes=args.processes)
        build_summary(model, test_cases_csv=args.test_cases)


def cmd_compare(args):
    from rsa_planning.compare import build_comparison
    build_comparison()


def cmd_all(args):
    args.model = "all"
    cmd_simulate(args)
    cmd_summarize(args)
    cmd_compare(args)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, with_model=True):
        if with_model:
            p.add_argument("--model", choices=MODELS + ["all"], default="all",
                           help="model variant to run (default: all four)")
        p.add_argument("--test-cases", default=None,
                       help="test-case CSV (default: data/scenarios/dynamic_test_cases_v5_0725.csv)")
        p.add_argument("--processes", type=int, default=4,
                       help="worker processes (default: 4)")
        p.add_argument("--horizon", type=int, default=7,
                       help="planning horizon H (default: 7, as in the paper)")
        p.add_argument("--limit", type=int, default=None,
                       help="run only the first N scenarios (for smoke tests)")
        p.add_argument("--tc-id", action="append", default=None,
                       help="run only this scenario id; repeatable")
        p.add_argument("--overwrite", action="store_true",
                       help="recompute results that already exist")

    p = sub.add_parser("scenarios", help="generate scenarios and test cases")
    p.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    p.add_argument("--version", default="custom",
                   help="suffix for the output filenames (default: custom). "
                        "'5_0725' is the set used in the paper; do not reuse it.")
    p.add_argument("--per-type", type=int, default=25, dest="per_type",
                   help="scenarios per critical-property condition (2, 3, 4). "
                        "Each scenario yields 4 test cases. Default 25 -> 800.")
    p.add_argument("--start-id", type=int, default=101, dest="start_id",
                   help="first scenario number, kept clear of the shipped set "
                        "(which uses 1-75). Default 101.")
    p.add_argument("--seed", type=int, default=None,
                   help="seed the generator so the set is reproducible")
    p.add_argument("--overwrite", action="store_true",
                   help="allow overwriting existing scenario files")
    p.set_defaults(func=cmd_scenarios)

    p = sub.add_parser("precompute", help="enumerate legal utterance sequences")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_precompute)

    p = sub.add_parser("simulate", help="run model variants over the scenarios")
    add_common(p)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("summarize", help="trajectories and per-run metrics")
    add_common(p)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("compare", help="build the wide and long comparison tables")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("all", help="simulate, summarize and compare, all variants")
    add_common(p, with_model=False)
    p.set_defaults(func=cmd_all)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ensure_output_dirs()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
