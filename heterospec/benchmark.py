"""Benchmark CLI.

    python -m heterospec.benchmark --workload mixed_50_50 \\
        --policy sglang_adaptive --num-requests 500

Runs on the Mac today with ``--mock`` (no GPU, no SGLang):

    python -m heterospec.benchmark --workload mixed_50_50 \\
        --policy static_k3 --num-requests 120 --mock

and emits the exact server command for a rented GPU with ``--print-commands``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from heterospec.config import load_launch_configs
from heterospec.mockserver import MockSGLangServer
from heterospec.runner import RunConfig, run_benchmark
from heterospec.workloads import workload_names

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCH_CONFIG = REPO_ROOT / "configs" / "models" / "llama31_8b_eagle3.json"
DEFAULT_SGLANG_PATH = REPO_ROOT.parent / "sglang"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m heterospec.benchmark",
        description="Run one HeteroSpec benchmark against an SGLang server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--workload", help=f"one of: {', '.join(workload_names())}")
    p.add_argument("--policy", help="launch config id, e.g. static_k3, sglang_adaptive")
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--dispatch",
        choices=("waves", "continuous"),
        default="waves",
        help="waves = controlled mixtures (oracle study); "
        "continuous = realistic sliding window",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=900.0)

    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--launch-config", type=Path, default=DEFAULT_LAUNCH_CONFIG)
    p.add_argument("--sglang-path", type=Path, default=DEFAULT_SGLANG_PATH)
    p.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results",
        help="where to write the run directory",
    )

    p.add_argument(
        "--mock",
        action="store_true",
        help="start a mock SGLang server and run against it (no GPU). "
        "Mock runs are never citable.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="build the plan, send nothing"
    )
    p.add_argument("--no-write", action="store_true", help="do not persist results")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--list-workloads", action="store_true")
    p.add_argument("--list-policies", action="store_true")
    p.add_argument(
        "--print-commands",
        action="store_true",
        help="print the SGLang launch command for each policy and exit",
    )
    return p


def _load_launches(path: Path):
    if not path.is_file():
        raise SystemExit(f"launch config not found: {path}")
    return {c.id: c for c in load_launch_configs(path)}


def cmd_list_workloads() -> int:
    from heterospec.workloads import get_workload

    print("workloads:")
    for name in workload_names():
        w = get_workload(name)
        phases = " | ".join(
            f"{p.label or '-'}:"
            + ",".join(f"{k}={v:g}" for k, v in p.composition.items())
            + f" x{p.n_requests}"
            for p in w.phases
        )
        print(f"  {name:14s} {w.description}")
        print(f"  {'':14s}   {phases}")
    return 0


def cmd_list_policies(path: Path) -> int:
    launches = _load_launches(path)
    print(f"policies in {path}:")
    for cid, c in launches.items():
        if c.spec is None:
            kind = "no speculation"
        else:
            kind = (
                f"K={c.spec.num_steps} draft={c.spec.num_draft_tokens} "
                f"{c.spec.algorithm}" + (" adaptive" if c.spec.adaptive else " static")
            )
        print(f"  {cid:18s} {kind}")
    return 0


def cmd_print_commands(path: Path) -> int:
    launches = _load_launches(path)
    print("# SGLang launch commands (run on the GPU host, from the sglang checkout)\n")
    for cid, c in launches.items():
        print(f"# ---- {cid}: {c.description}")
        print(c.command())
        print()
    return 0


def _fmt_float(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_workloads:
        return cmd_list_workloads()
    if args.list_policies:
        return cmd_list_policies(args.launch_config)
    if args.print_commands:
        return cmd_print_commands(args.launch_config)

    if not args.workload:
        print(
            "error: --workload is required (or use --list-workloads)", file=sys.stderr
        )
        return 2
    if not args.policy:
        print("error: --policy is required (or use --list-policies)", file=sys.stderr)
        return 2
    if args.workload not in workload_names():
        print(
            f"error: unknown workload {args.workload!r}; "
            f"choose from {', '.join(workload_names())}",
            file=sys.stderr,
        )
        return 2

    launches = _load_launches(args.launch_config)
    if args.policy not in launches:
        print(
            f"error: unknown policy {args.policy!r}; choose from {', '.join(launches)}",
            file=sys.stderr,
        )
        return 2
    launch = launches[args.policy]
    launch.validate()

    # Static K is known exactly; adaptive K is not, and the analysis must know
    # the difference (see heterospec/telemetry.py).
    static_k = None
    if launch.spec is not None and not launch.spec.adaptive:
        static_k = launch.spec.num_steps

    progress = None if args.quiet else print

    server: MockSGLangServer | None = None
    base_url = args.base_url
    if args.mock:
        # Mirror the policy's K so the mock exercises the same shape.
        k = static_k if static_k and static_k > 0 else 4
        server = MockSGLangServer(k=k, seed=args.seed).start()
        base_url = server.base_url
        if not args.quiet:
            print(f"started mock SGLang at {base_url} (k={k})")

    try:
        cfg = RunConfig(
            workload=args.workload,
            policy_id=args.policy,
            base_url=base_url,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            dispatch=args.dispatch,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
            results_root=args.results_root,
            sglang_path=args.sglang_path,
            launch=launch.to_dict(),
            static_k=static_k,
            dry_run=args.dry_run,
            write_results=not args.no_write,
            progress=progress,
        )
        result = run_benchmark(cfg)
    finally:
        if server is not None:
            server.stop()

    if not args.quiet:
        print()
        print(f"policy       : {args.policy}")
        print(f"workload     : {args.workload}")
        print(f"requests     : {result.plan_summary.get('total', 0)}")
        print(f"dispatch     : {args.dispatch} (concurrency {args.concurrency})")
        print(f"wall time    : {result.wall_time_s:.1f}s")
        if result.records:
            a = result.aggregate
            print(f"ok / failed  : {a['n_ok']} / {a['n_failed']}")
            print(f"speculative  : {a['n_speculative']}")
            print(
                f"mean accepted: {_fmt_float(a['mean_accepted_drafts'])} drafts/round"
            )
            print(f"draft waste  : {_fmt_float(a['draft_waste'])}")
            if not a.get("draft_waste_covers_all_speculative"):
                print(
                    "               (partial: K known for "
                    f"{a['n_with_known_k']}/{a['n_speculative']} speculative requests)"
                )
            print(f"mean latency : {_fmt_float(a['mean_latency_s'])}s")
            if a.get("histogram_inconsistencies"):
                print(
                    f"WARNING      : {len(a['histogram_inconsistencies'])} histogram/"
                    f"verify-count mismatches: {a['histogram_inconsistencies'][:5]}"
                )
            print("by class     :")
            for cls, s in sorted(a.get("by_class", {}).items()):
                print(
                    f"  {cls:12s} n={s['n']:<4d} "
                    f"mean_accepted={_fmt_float(s['mean_accepted_drafts'])} "
                    f"mean_tokens={_fmt_float(s['mean_completion_tokens'])}"
                )
        ok, reason = result.citable
        print(f"citable      : {ok} ({reason})")
        if result.run_dir:
            print(f"results      : {result.run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
